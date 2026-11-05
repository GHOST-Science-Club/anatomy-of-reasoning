import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from labeling import labelers as m


def answer(choice, probs=None):
    probs = probs or {label: (0.93 if label == choice else 0.01) for label in m.LABELS}
    return {
        'type': 'choice',
        'choice': choice,
        'probabilities': probs,
        'confidence': max(probs.values()),
    }


def envelope(answers):
    """Copy of a real basal-serve /v1/systemone response (basal 2.5.3, basal-2.0-4.5B)."""
    return {
        'model': 'basal-2.0-4.5B',
        'answers': answers,
        'usage': {
            'input_tokens': 127,
            'output_tokens': 0,
            'questions': len(answers),
            'branches': 1,
            'latency_ms': 33.22,
            'longest_prompt_tokens': 127,
        },
    }


def quoted(question):
    return question['instructions'].rsplit('Sentence: ', 1)[1]


def by_position(body):
    """Answer each question `sK` with the K-th label in LABELS."""
    return envelope({name: answer(m.LABELS[int(name[1:]) - 1]) for name in body['questions']})


def same_label(choice):
    return lambda body: envelope({name: answer(choice) for name in body['questions']})


@pytest.fixture
def server(monkeypatch):
    """Fake basal-serve behind a real httpx client; `handler` maps a request body to a response."""
    bodies, sleeps = [], []
    state = SimpleNamespace(bodies=bodies, sleeps=sleeps, handler=by_position)

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


def label(sentences, batch_size=2):
    labeler = m.REGISTRY['basal']({'model': 'basal', 'batch_size': batch_size})
    return asyncio.run(labeler.label('What is 2+2?', sentences))


def test_registered():
    assert m.REGISTRY['basal'] is m.BasalLabeler


def test_request_shape(server):
    label(['First.', 'Second.', 'Third.'])
    assert len(server.bodies) == 2
    state = server.bodies[1]['state']
    assert Path(m.DEFAULT_GUIDEBOOK).read_text(encoding='utf-8') in state
    assert 'What is 2+2?' in state
    assert all(sentence not in state for sentence in ['First.', 'Second.', 'Third.'])
    assert server.bodies[0]['state'] == state
    questions = server.bodies[1]['questions']
    assert list(questions) == ['s1']
    assert questions['s1']['type'] == 'choice'
    assert questions['s1']['criteria'] == m.LABELS
    assert quoted(questions['s1']) == 'Third.'
    first = server.bodies[0]['questions']
    assert [quoted(first[name]) for name in ['s1', 's2']] == ['First.', 'Second.']


def test_answers_map_to_sentences_in_order(server):
    predictions = label(['a', 'b', 'c', 'd', 'e'], batch_size=3)
    assert [p.label for p in predictions] == ['Read', 'Analyze', 'Plan', 'Read', 'Analyze']
    assert all(p.probs and p.reason is None for p in predictions)


def test_valid_response_becomes_prediction(server):
    probs = dict.fromkeys(m.LABELS, 0.0) | {'Verify': 0.7, 'Answer': 0.3}
    server.handler = lambda body: envelope({'s1': answer('Verify', probs)})
    assert label(['The answer is 4, check.']) == [m.Prediction('Verify', None, probs)]


@pytest.mark.parametrize(
    'reply',
    [
        envelope({'s1': answer('Summarize'), 's2': answer('Read')}),
        envelope({'s1': answer('Read')}),
        {'error': 'oops'},
    ],
)
def test_unknown_or_missing_choice_raises(server, reply):
    server.handler = lambda body: reply
    with pytest.raises(ValueError):
        label(['a', 'b'])


@pytest.mark.parametrize('failure', ['503', 'connect'])
def test_transient_error_retried_then_succeeds(server, failure):
    calls = []

    def handler(body):
        calls.append(body)
        if len(calls) < 3:
            if failure == 'connect':
                raise httpx.ConnectError('refused')
            return httpx.Response(503, json={'error': 'out of memory'})
        return same_label('Monitor')(body)

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
        index = int(quoted(json['questions']['s1']).split()[1].rstrip('.'))
        await asyncio.sleep(0.01 * (len(labels) - index))
        finished.append(index)
        reply = envelope({'s1': answer(labels[index])})
        return httpx.Response(200, json=reply, request=httpx.Request('POST', url))

    monkeypatch.setattr(httpx.AsyncClient, 'post', post)
    predictions = label([f'Sentence {i}.' for i in range(len(labels))], batch_size=1)
    assert finished == [3, 2, 1, 0]
    assert [p.label for p in predictions] == labels


def test_criteria_file_must_describe_all_labels(tmp_path):
    prompt = tmp_path / 'basal.md'
    prompt.write_text('Which category?\nSentence: {sentence}\n\nRead\nAnalyze\n')
    with pytest.raises(ValueError, match='labels'):
        m.BasalLabeler({'criteria': str(prompt)})


def test_criteria_question_must_contain_sentence(tmp_path):
    prompt = tmp_path / 'basal.md'
    lines = '\n'.join(m.LABELS)
    prompt.write_text(f'Which category?\n\n{lines}\n')
    with pytest.raises(ValueError, match='sentence'):
        m.BasalLabeler({'criteria': str(prompt)})


def test_failure_cancels_other_requests(server, monkeypatch):
    completed = []

    async def post(self, url, json):
        request = httpx.Request('POST', url)
        if quoted(json['questions']['s1']) == 'Bad.':
            return httpx.Response(400, request=request)
        await asyncio.sleep(0.2)
        completed.append(json['state'])
        return httpx.Response(200, json=envelope({'s1': answer('Read')}), request=request)

    async def run():
        labeler = m.BasalLabeler({'model': 'basal', 'batch_size': 1})
        with pytest.raises(httpx.HTTPStatusError):
            await labeler.label('P', ['Slow.', 'Bad.', 'Slow too.'])
        await asyncio.sleep(0.3)

    monkeypatch.setattr(httpx.AsyncClient, 'post', post)
    asyncio.run(run())
    assert completed == []
