import asyncio
import json
from pathlib import Path

import pytest
import yaml

from labeling import run as m
from labeling.labelers import Prediction
from labeling.__main__ import main


def write_jsonl(path, records):
    path.write_text(''.join(json.dumps(row) + '\n' for row in records))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(m, 'load_dotenv', lambda: None)
    records = [{'id': 'good', 'problem': 'P', 'trace': 'First. Second.'},
               {'id': 'bad', 'problem': 'bad', 'sentences': ['Given.']},
               {'id': 'empty', 'problem': 'P', 'sentences': []}]
    write_jsonl(Path('input.jsonl'), records)
    config = {'input': 'input.jsonl', 'labeler': 'fake', 'model': 'test', 'concurrency': 2, 'traces_in_flight': 2}
    config_path = Path('config.yaml')
    config_path.write_text(yaml.safe_dump(config))
    calls = []
    fail = [True]

    class Fake:
        async def label(self, problem, sentences):
            calls.append((problem, sentences))
            await asyncio.sleep(0)
            if problem == 'bad' and fail[0]:
                raise ValueError('batch failed')
            return [Prediction('Read', 'reason', {'Read': 1.0}) for _ in sentences]

    monkeypatch.setitem(m.REGISTRY, 'fake', lambda config: Fake())
    return config_path, config, calls, fail, records


def test_failure_resume_metadata_and_idempotence(setup, capsys):
    config_path, config, calls, fail, records = setup
    assert asyncio.run(m.run(config_path, 'trial')) == 1
    folder = Path('runs/trial')
    output = folder / 'labels.jsonl'
    saved = [json.loads(line) for line in output.read_text().splitlines()]
    assert {row['id'] for row in saved} == {'good', 'empty'}
    assert (folder / 'config.yaml').read_bytes() == config_path.read_bytes()
    assert 'commit' in json.loads((folder / 'meta.json').read_text())
    assert saved[0]['sentences'][0] == {'index': 0, 'text': 'First.', 'label': 'Read', 'reason': 'reason', 'probs': {'Read': 1.0}}
    assert saved[1]['sentences'] == []
    with output.open('a') as f:
        f.write('{"id": "torn')
    calls.clear()
    fail[0] = False
    assert asyncio.run(m.run(config_path, 'trial')) == 0
    assert calls == [('bad', ['Given.'])]
    assert {json.loads(line)['id'] for line in output.read_text().splitlines()} == {'good', 'bad', 'empty'}
    before = output.read_bytes()
    calls.clear()
    assert asyncio.run(m.run(config_path, 'trial')) == 0
    assert calls == []
    assert output.read_bytes() == before
    assert 'bad' in capsys.readouterr().out


def test_resume_configuration_guard(setup):
    path, config, calls, fail, _ = setup
    asyncio.run(m.run(path, 'trial'))
    config.update(concurrency=3, traces_in_flight=1)
    path.write_text(yaml.safe_dump(config))
    asyncio.run(m.run(path, 'trial'))
    config['model'] = 'different'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='new run'):
        asyncio.run(m.run(path, 'trial'))


def test_presplit_bypasses_splitter(setup, monkeypatch):
    path, _, _, fail, records = setup
    write_jsonl(Path('input.jsonl'), [records[1]])
    monkeypatch.setattr(m, 'process_response_to_sentences', lambda _: pytest.fail('splitter called'))
    fail[0] = False
    assert asyncio.run(m.run(path, 'trial')) == 0


@pytest.mark.parametrize('records', [
    [{'id': 'x', 'problem': 'P', 'trace': 'a', 'sentences': ['a']}],
    [{'id': 'x', 'problem': 'P'}],
    [{'id': 1, 'problem': 'P', 'trace': 'a'}],
    [{'id': 'x', 'problem': 2, 'trace': 'a'}],
    [{'id': 'x', 'problem': 'P', 'trace': []}],
    [{'id': 'x', 'problem': 'P', 'sentences': [1]}],
    [{'id': 'x', 'problem': 'P', 'trace': 'a'}] * 2,
])
def test_invalid_input_rejected_before_calls(setup, records):
    path, _, calls, _, _ = setup
    write_jsonl(Path('input.jsonl'), records)
    with pytest.raises(ValueError):
        asyncio.run(m.run(path, 'trial'))
    assert not calls


def test_unknown_labeler_lists_registry(setup):
    path, config, calls, _, _ = setup
    config['labeler'] = 'unknown'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='openai'):
        asyncio.run(m.run(path, 'trial'))
    assert not calls


@pytest.mark.parametrize('change', [{'batch_size': 0}, {'concurrency': True}, {'traces_in_flight': -1}, {'response_format': 'invalid'}])
def test_invalid_config(setup, change):
    path, config, _, _, _ = setup
    config.update(change)
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        asyncio.run(m.run(path, 'trial'))


def test_run_required():
    with pytest.raises(SystemExit) as exc:
        main(['label', 'config.yaml'])
    assert exc.value.code == 2


@pytest.mark.parametrize('name', ['../escape', '/absolute', '.', 'nested/name'])
def test_invalid_run_name(setup, name):
    with pytest.raises(ValueError, match='run name'):
        asyncio.run(m.run(setup[0], name))


def test_prediction_count_guard(setup, monkeypatch):
    class Short:
        async def label(self, *args):
            return []
    monkeypatch.setitem(m.REGISTRY, 'fake', lambda c: Short())
    assert asyncio.run(m.run(setup[0], 'trial')) == 1
    assert [json.loads(line)['id'] for line in Path('runs/trial/labels.jsonl').read_text().splitlines()] == ['empty']


def test_trace_limit_and_completion_order(setup, monkeypatch):
    path, config, _, _, _ = setup
    write_jsonl(Path('input.jsonl'), [{'id': str(i), 'problem': str(i), 'sentences': ['s']} for i in range(8)])
    active = peak = 0
    started = []
    class Limited:
        async def label(self, problem, sentences):
            nonlocal active, peak
            started.append(problem)
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02 if problem == '0' else 0)
            active -= 1
            return [Prediction('Read')]
    monkeypatch.setitem(m.REGISTRY, 'fake', lambda c: Limited())
    assert asyncio.run(m.run(path, 'trial')) == 0
    assert peak == 2
    assert started == list(map(str, range(8)))
    output = [json.loads(s)['id'] for s in Path('runs/trial/labels.jsonl').read_text().splitlines()]
    assert output[0] == '1'


def test_malformed_middle_line_is_not_ignored(setup):
    asyncio.run(m.run(setup[0], 'trial'))
    Path('runs/trial/labels.jsonl').write_text('{bad}\n{"id":"good","sentences":[]}\n')
    with pytest.raises(ValueError):
        asyncio.run(m.run(setup[0], 'trial'))


def test_cli_failure_then_resume_and_secret_free_metadata(setup, monkeypatch):
    path, config, calls, fail, _ = setup
    monkeypatch.setenv('OPENAI_API_KEY', 'private-test-openai-key')
    monkeypatch.setenv('LABELER_API_KEY', 'private-test-labeler-key')
    assert main(['label', str(path), '--run', 'cli']) == 1
    calls.clear()
    fail[0] = False
    assert main(['label', str(path), '--run', 'cli']) == 0
    assert calls == [('bad', ['Given.'])]
    calls.clear()
    before = Path('runs/cli/labels.jsonl').read_bytes()
    assert main(['label', str(path), '--run', 'cli']) == 0
    assert not calls
    assert Path('runs/cli/labels.jsonl').read_bytes() == before
    for name in ['config.yaml', 'meta.json']:
        text = Path('runs/cli', name).read_text()
        assert 'private-test' not in text


def test_failure_errors_do_not_expose_environment_keys(setup, monkeypatch, capsys):
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-openai-secret')
    monkeypatch.setenv('LABELER_API_KEY', 'synthetic-local-secret')
    class EchoingError:
        async def label(self, problem, sentences):
            raise ValueError('request rejected synthetic-openai-secret synthetic-local-secret')
    monkeypatch.setitem(m.REGISTRY, 'fake', lambda config: EchoingError())
    assert asyncio.run(m.run(setup[0], 'secret-error')) == 1
    captured = capsys.readouterr()
    assert 'synthetic-openai-secret' not in captured.out + captured.err
    assert 'synthetic-local-secret' not in captured.out + captured.err
    assert 'request rejected' in captured.out


def test_cli_startup_errors_redact_keys(setup, monkeypatch, capsys):
    monkeypatch.setenv('LABELER_API_KEY', 'synthetic-startup-secret')
    def fail(config):
        raise ValueError('startup rejected synthetic-startup-secret')
    monkeypatch.setitem(m.REGISTRY, 'fake', fail)
    with pytest.raises(SystemExit) as exc:
        main(['label', str(setup[0]), '--run', 'startup'])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert 'synthetic-startup-secret' not in captured.out + captured.err
