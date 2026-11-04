import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('convert_thinkarm', Path(__file__).resolve().parents[1] / 'scripts/convert_thinkarm.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(root, model, annotated=(1, 2)):
    raw = root / 'ThinkARM/data/raw'
    ground = root / 'ThinkARM/data/ground_truth' / model
    raw.mkdir(parents=True, exist_ok=True)
    ground.mkdir(parents=True)
    (raw / f'{model}.json').write_text(json.dumps([
        {'Instruction': f'Problem {i}', 'Response': f'Full reasoning {i}. Answer.'}
        for i in range(1, 13)]))
    for i in annotated:
        (ground / f'{i}.json').write_text(json.dumps([
            {'index': 1, 'sentence': f'Text {i}', 'sentence-type': 'think', 'human_label': 'Read'},
            {'index': 2, 'sentence': 'Answer', 'sentence-type': 'answer', 'human_label': 'Answer'}]))


def read_output(folder, name):
    return [json.loads(line) for line in (Path(folder) / f'{name}.jsonl').read_text().splitlines()]


def test_default_exports_all_full_traces_and_available_human_labels(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for model in ['m1', 'm2']:
        fixture(tmp_path, model)
    module.main(['m1', 'm2', '--out', 'converted'])
    traces, human = read_output('converted', 'traces'), read_output('converted', 'human')
    assert len(traces) == 24 and len(human) == 4
    assert traces[0] == {'id': 'm1-1', 'problem': 'Problem 1', 'trace': 'Full reasoning 1. Answer.'}
    assert traces[-1]['id'] == 'm2-12'
    assert human[0] == {'id': 'm1-1', 'sentences': [
        {'index': 0, 'text': 'Text 1', 'label': 'Read'}, {'index': 1, 'text': 'Answer', 'label': 'Answer'}]}
    assert '24 traces' in capsys.readouterr().out


def test_human_only_exports_annotated_prefixes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fixture(tmp_path, 'm1', annotated=(1, 4, 12))
    module.main(['m1', '--human-only', '--out', 'converted'])
    traces = read_output('converted', 'traces')
    assert [row['id'] for row in traces] == ['m1-1', 'm1-4', 'm1-12']
    assert traces[0] == {'id': 'm1-1', 'problem': 'Problem 1', 'sentences': ['Text 1', 'Answer']}
    assert [r['id'] for r in traces] == [r['id'] for r in read_output('converted', 'human')]


@pytest.mark.parametrize('human_only', [False, True])
def test_limit_is_per_model_after_human_filter(tmp_path, monkeypatch, human_only):
    monkeypatch.chdir(tmp_path)
    for model in ['m1', 'm2']:
        fixture(tmp_path, model, annotated=(3, 5, 8))
    args = ['m1', 'm2', '--limit', '2', '--out', 'converted']
    if human_only:
        args.append('--human-only')
    module.main(args)
    traces, human = read_output('converted', 'traces'), read_output('converted', 'human')
    expected = ['m1-3', 'm1-5', 'm2-3', 'm2-5'] if human_only else ['m1-1', 'm1-2', 'm2-1', 'm2-2']
    assert [row['id'] for row in traces] == expected
    assert [row['id'] for row in human] == (expected if human_only else [])


def test_unannotated_source_can_be_converted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fixture(tmp_path, 'm1', annotated=())
    Path('ThinkARM/data/ground_truth/m1').rmdir()
    module.main(['m1', '--out', 'converted'])
    assert len(read_output('converted', 'traces')) == 12
    assert read_output('converted', 'human') == []


def test_human_only_without_annotations_is_clear_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    fixture(tmp_path, 'm1', annotated=())
    with pytest.raises(SystemExit) as exc:
        module.main(['m1', '--human-only', '--out', 'out'])
    assert exc.value.code == 2
    assert 'No human labels' in capsys.readouterr().err
    assert not Path('out/traces.jsonl').exists()


@pytest.mark.parametrize('limit', ['0', '-1', 'abc'])
def test_invalid_limit_is_rejected(limit):
    with pytest.raises(SystemExit) as exc:
        module.main(['m1', '--limit', limit, '--out', 'out'])
    assert exc.value.code == 2


def test_missing_raw_source_does_not_write_partial_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fixture(tmp_path, 'm1')
    with pytest.raises(FileNotFoundError):
        module.main(['m1', 'missing', '--out', 'out'])
    assert not Path('out').exists()
