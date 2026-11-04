import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from labeling import labelers as m


def envelope(choice, probs=None):
    """Copy of a real basal-serve /v1/systemone response (basal 2.5.3, basal-2.0-4.5B)."""
    probs = probs or {label: (0.93 if label == choice else 0.01) for label in m.LABELS}
    return {
        'model': 'basal-2.0-4.5B',
        'answers': {
            'label': {
                'type': 'choice',
                'choice': choice,
                'probabilities': probs,
                'confidence': max(probs.values()),
            }
        },
        'usage': {
            'input_tokens': 127,
            'output_tokens': 0,
            'questions': 1,
            'branches': 1,
            'latency_ms': 33.22,
            'longest_prompt_tokens': 127,
        },
    }


@pytest.fixture
def server(monkeypatch):
    """Fake basal-serve behind a real httpx client; `handler` maps a request body to a response."""
    bodies, sleeps = [], []
    state = SimpleNamespace(bodies=bodies, sleeps=sleeps, handler=lambda body: envelope('Plan'))

    def transport(request):
        body = json.loads(request.content)
        bodies.append(body)
        result = state.handler(body)
        return result if isinstance(result, httpx.Response) else httpx.Response(200, json=result)

    async def async_transport(request):
        return transport(request)

    real = httpx.AsyncClient
    monkeypatch.setattr(
        m,
        'AsyncClient',
        lambda **kw: real(transport=httpx.MockTransport(async_transport), **kw),
    )

    async def sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(m, 'sleep', sleep)
    return state


def label(sentences, config=None):
    labeler = m.REGISTRY['basal']({'model': 'basal', **(config or {})})
    return asyncio.run(labeler.label('What is 2+2?', sentences))


def test_registered():
    assert m.REGISTRY['basal'] is m.BasalLabeler


def test_request_shape(server):
    label(['First.', 'Second.', 'Third.'])
    body = server.bodies[2]
    assert set(body['questions']) == {'label'}
    question = body['questions']['label']
    assert question['type'] == 'choice'
    assert list(question['criteria']) == m.LABELS
    assert all(question['criteria'].values())
    assert question['instructions']
    state = body['state']
    assert 'What is 2+2?' in state
    context, target = state.split(m.BASAL_TARGET)
    assert 'First.\nSecond.' in context
    assert 'Third.' in target and 'Third.' not in context
    first = server.bodies[0]['state']
    assert 'First.' in first.split(m.BASAL_TARGET)[1]


def test_valid_response_becomes_prediction(server):
    probs = dict.fromkeys(m.LABELS, 0.0) | {'Verify': 0.7, 'Answer': 0.3}
    server.handler = lambda body: envelope('Verify', probs)
    assert label(['The answer is 4, check.']) == [m.Prediction('Verify', None, probs)]


@pytest.mark.parametrize(
    'reply',
    [
        envelope('Summarize'),
        {'model': 'basal-2.0-4.5B', 'answers': {}, 'usage': {}},
        {'error': 'oops'},
    ],
)
def test_unknown_or_missing_choice_raises(server, reply):
    server.handler = lambda body: reply
    with pytest.raises(ValueError):
        label(['Sentence.'])


@pytest.mark.parametrize('failure', ['503', 'connect'])
def test_transient_error_retried_then_succeeds(server, failure):
    calls = []

    def handler(body):
        calls.append(body)
        if len(calls) < 3:
            if failure == 'connect':
                raise httpx.ConnectError('refused')
            return httpx.Response(503, json={'error': 'out of memory'})
        return envelope('Monitor')

    server.handler = handler
    assert [p.label for p in label(['Wait.'])] == ['Monitor']
    assert len(calls) == 3
    assert len(server.sleeps) == 2


@pytest.mark.parametrize('status', [400, 422])
def test_client_error_not_retried(server, status):
    server.handler = lambda body: httpx.Response(status, json={'error': 'bad request'})
    with pytest.raises(httpx.HTTPStatusError):
        label(['Sentence.'])
    assert len(server.bodies) == 1
    assert server.sleeps == []


def test_output_order_when_requests_finish_out_of_order(server, monkeypatch):
    labels = ['Read', 'Analyze', 'Plan', 'Implement']
    finished = []

    async def post(self, url, json):
        index = int(json['state'].rsplit('Sentence ', 1)[1].rstrip('.'))
        await asyncio.sleep(0.01 * (len(labels) - index))
        finished.append(index)
        return httpx.Response(200, json=envelope(labels[index]), request=httpx.Request('POST', url))

    monkeypatch.setattr(httpx.AsyncClient, 'post', post)
    predictions = label([f'Sentence {i}.' for i in range(len(labels))])
    assert finished == [3, 2, 1, 0]
    assert [p.label for p in predictions] == labels


def test_prompt_file_must_describe_all_labels(tmp_path):
    prompt = tmp_path / 'basal.md'
    prompt.write_text('Which category?\n\nRead: restates.\nAnalyze: infers.\n')
    with pytest.raises(ValueError, match='labels'):
        m.BasalLabeler({'guidebook': str(prompt)})


def test_failure_cancels_other_requests(server, monkeypatch):
    completed = []

    async def post(self, url, json):
        request = httpx.Request('POST', url)
        if json['state'].endswith('Bad.'):
            return httpx.Response(400, request=request)
        await asyncio.sleep(0.2)
        completed.append(json['state'])
        return httpx.Response(200, json=envelope('Read'), request=request)

    async def run():
        labeler = m.BasalLabeler({'model': 'basal'})
        with pytest.raises(httpx.HTTPStatusError):
            await labeler.label('P', ['Slow.', 'Bad.', 'Slow too.'])
        await asyncio.sleep(0.3)

    monkeypatch.setattr(httpx.AsyncClient, 'post', post)
    asyncio.run(run())
    assert completed == []
