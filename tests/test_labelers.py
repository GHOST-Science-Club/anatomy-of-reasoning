import asyncio
import json
import re
from types import SimpleNamespace

import httpx
import openai
import pytest

from labeling import labelers as m


def response(rows):
    content = rows if isinstance(rows, str) else json.dumps({'sentences': rows})
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def rows(n, label='Read'):
    return [{'index': i, 'reason': f'reason {i}', 'category': label} for i in range(1, n + 1)]


@pytest.fixture
def setup(monkeypatch, tmp_path):
    calls, sleeps, options = [], [], []
    guide = tmp_path / 'guide.md'
    guide.write_text('Guide {x}')
    active = peak = 0

    async def default(**kw):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        n = int(re.search(r'label for (\d+) sentences', kw['messages'][0]['content'])[1])
        return response(rows(n))

    handler = [default]

    async def create(**kw):
        calls.append(kw)
        return await handler[0](**kw)

    def client(**kw):
        options.append(kw)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    async def sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(m, 'AsyncOpenAI', client)
    monkeypatch.setattr(m, 'sleep', sleep)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('LABELER_API_KEY', raising=False)
    config = {'model': 'test', 'base_url': 'http://localhost:4000', 'guidebook': str(guide)}
    return SimpleNamespace(config=config, calls=calls, sleeps=sleeps, options=options,
                           handler=handler, peak=lambda: peak)


def test_context_and_prediction(setup):
    labeler = m.OpenAICompatLabeler(setup.config)
    sentences = [f'Sentence {i}.' for i in range(25)]
    predictions = asyncio.run(labeler.label('Problem', sentences))
    assert len(predictions) == 25
    assert predictions[20] == m.Prediction('Read', 'reason 1')
    assert all(p.probs is None for p in predictions)
    prompt = setup.calls[1]['messages'][0]['content']
    context = prompt.split('[Previous Context]\n')[1].split('[End of the Previous Context]')[0]
    assert '\n'.join(sentences[:20]) in context
    assert sentences[20] not in context
    assert '[1] Sentence 20.' in prompt
    assert 'Guide {x}' in prompt
    assert setup.calls[0]['messages'][0]['role'] == 'user'


@pytest.mark.parametrize('count,requests', [(0, 0), (20, 1), (21, 2)])
def test_batch_alignment(setup, count, requests):
    assert len(asyncio.run(m.OpenAICompatLabeler(setup.config).label('P', ['s'] * count))) == count
    assert len(setup.calls) == requests


@pytest.mark.parametrize('bad', [
    '{', {}, [], {'sentences': []},
    {'sentences': rows(2)[:1]},
    {'sentences': [rows(2)[0], rows(2)[0]]},
    {'sentences': [*rows(1), {'index': 3, 'reason': 'r', 'category': 'Read'}]},
    {'sentences': rows(2, 'Unknown')},
    {'sentences': [{'index': True, 'reason': 'r', 'category': 'Read'}, rows(2)[1]]},
    {'sentences': [{'index': '1', 'reason': 'r', 'category': 'Read'}, rows(2)[1]]},
    {'sentences': [{'index': 1, 'category': 'Read'}, rows(2)[1]]},
])
def test_persistently_invalid_response_fails(setup, bad):
    async def invalid(**kw):
        return response(bad if isinstance(bad, str) else json.dumps(bad))
    setup.handler[0] = invalid
    with pytest.raises(ValueError):
        asyncio.run(m.OpenAICompatLabeler(setup.config).label('P', ['a', 'b']))
    assert len(setup.calls) == 5
    assert setup.sleeps == []


def test_case_normalization_and_index_order(setup):
    async def mixed(**kw):
        return response([{'index': 2, 'reason': 'second', 'category': 'aNsWeR'},
                         {'index': 1, 'reason': 'first', 'category': 'READ'}])
    setup.handler[0] = mixed
    result = asyncio.run(m.OpenAICompatLabeler(setup.config).label('P', ['a', 'b']))
    assert result == [m.Prediction('Read', 'first'), m.Prediction('Answer', 'second')]


@pytest.mark.parametrize('mode', ['json_schema', 'json_object'])
@pytest.mark.parametrize('set_options', [False, True])
def test_request_shape(setup, mode, set_options):
    setup.config.update(response_format=mode, temperature=0.3 if set_options else None,
                        max_tokens=100 if set_options else None)
    asyncio.run(m.OpenAICompatLabeler(setup.config).label('P', ['a']))
    request = setup.calls[0]
    if mode == 'json_object':
        assert request['response_format'] == {'type': 'json_object'}
    else:
        schema = request['response_format']['json_schema']
        assert schema['strict'] is True
        item = schema['schema']['properties']['sentences']['items']
        assert item['properties']['category']['enum'] == m.LABELS
        assert item['additionalProperties'] is False
        assert set(item['required']) == {'index', 'reason', 'category'}
    for key, value in [('temperature', 0.3), ('max_tokens', 100)]:
        if set_options:
            assert request[key] == value
        else:
            assert key not in request


def test_client_setup_and_keys(setup, monkeypatch):
    m.OpenAICompatLabeler(setup.config)
    assert setup.options[-1] == {'base_url': 'http://localhost:4000', 'api_key': 'unused', 'max_retries': 0}
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-test')
    m.OpenAICompatLabeler(setup.config)
    assert setup.options[-1]['api_key'] == 'openai-test'
    monkeypatch.setenv('LABELER_API_KEY', 'local-test')
    m.OpenAICompatLabeler(setup.config)
    assert setup.options[-1]['api_key'] == 'local-test'


def error(status):
    return openai.APIStatusError('request failed', response=httpx.Response(status, request=httpx.Request('POST', 'http://local')), body=None)


@pytest.mark.parametrize('status', [400, 401, 403, 404])
def test_permanent_error_not_retried(setup, status):
    async def fail(**kw):
        raise error(status)
    setup.handler[0] = fail
    with pytest.raises(openai.APIStatusError):
        asyncio.run(m.OpenAICompatLabeler(setup.config).label('P', ['a']))
    assert len(setup.calls) == 1
    assert setup.sleeps == []


@pytest.mark.parametrize('status', [429, 500, 'timeout', 'connection'])
def test_transient_retry_then_success(setup, status):
    async def retry(**kw):
        if len(setup.calls) == 1:
            if status == 'timeout':
                raise openai.APITimeoutError(request=httpx.Request('POST', 'http://local'))
            if status == 'connection':
                raise openai.APIConnectionError(request=httpx.Request('POST', 'http://local'))
            raise error(status)
        return response(rows(1))
    setup.handler[0] = retry
    assert asyncio.run(m.OpenAICompatLabeler(setup.config).label('P', ['a']))[0].label == 'Read'
    assert len(setup.sleeps) == 1
    assert 1 <= setup.sleeps[0] <= 2


def test_transient_exhaustion_and_backoff(setup):
    async def fail(**kw):
        raise error(500)
    setup.handler[0] = fail
    with pytest.raises(openai.APIStatusError):
        asyncio.run(m.OpenAICompatLabeler(setup.config).label('P', ['a']))
    assert len(setup.calls) == 5
    assert len(setup.sleeps) == 4
    assert setup.sleeps == sorted(setup.sleeps)
    assert all(0 < s <= 30 for s in setup.sleeps)
    assert m.backoff(20) <= 30


def test_validation_retry_is_immediate(setup):
    async def retry(**kw):
        return response('bad' if len(setup.calls) == 1 else rows(1))
    setup.handler[0] = retry
    assert asyncio.run(m.OpenAICompatLabeler(setup.config).label('P', ['a']))
    assert len(setup.calls) == 2
    assert not setup.sleeps


def test_shared_concurrency(setup):
    setup.config.update(concurrency=2, batch_size=1)
    labeler = m.OpenAICompatLabeler(setup.config)
    async def both():
        return await asyncio.gather(labeler.label('P', ['a'] * 5), labeler.label('Q', ['b'] * 5))
    asyncio.run(both())
    assert setup.peak() == 2


def test_out_of_order_batches(setup):
    setup.config['batch_size'] = 1
    finished = []
    async def delayed(**kw):
        prompt = kw['messages'][0]['content']
        first = '[1] first' in prompt
        await asyncio.sleep(0.01 if first else 0)
        finished.append(first)
        return response(rows(1, 'Read' if first else 'Answer'))
    setup.handler[0] = delayed
    result = asyncio.run(m.OpenAICompatLabeler(setup.config).label('P', ['first', 'second']))
    assert finished == [False, True]
    assert [p.label for p in result] == ['Read', 'Answer']


@pytest.mark.parametrize('kwargs', [
    {'enable_thinking': False}, {'thinking': False}, {'enable_thinking': True},
])
def test_chat_template_kwargs_are_forwarded(setup, kwargs):
    setup.config['chat_template_kwargs'] = kwargs
    asyncio.run(m.OpenAICompatLabeler(setup.config).label('P', ['a']))
    assert setup.calls[0]['extra_body'] == {'chat_template_kwargs': kwargs}


@pytest.mark.parametrize('explicit_null', [False, True])
def test_unset_chat_template_kwargs_are_omitted(setup, explicit_null):
    if explicit_null:
        setup.config['chat_template_kwargs'] = None
    asyncio.run(m.OpenAICompatLabeler(setup.config).label('P', ['a']))
    assert 'extra_body' not in setup.calls[0]


@pytest.mark.parametrize('kwargs', [False, 'false', [], 1])
def test_invalid_chat_template_kwargs_fail_at_startup(setup, kwargs):
    setup.config['chat_template_kwargs'] = kwargs
    with pytest.raises(ValueError, match='chat_template_kwargs must be a mapping'):
        m.OpenAICompatLabeler(setup.config)
    assert not setup.calls
