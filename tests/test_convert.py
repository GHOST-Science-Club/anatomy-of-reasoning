import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('convert_thinkarm', Path(__file__).resolve().parents[1] / 'scripts/convert_thinkarm.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(root, model):
    raw = root / 'ThinkARM/data/raw'
    ground = root / 'ThinkARM/data/ground_truth' / model
    raw.mkdir(parents=True, exist_ok=True)
    ground.mkdir(parents=True)
    (raw / f'{model}.json').write_text(json.dumps([{'Instruction': f'Problem {i}', 'Response': 'unused full trace'} for i in range(1, 10)]))
    for i in range(1, 10):
        (ground / f'{i}.json').write_text(json.dumps([
            {'index': 1, 'sentence': f'Text {i}', 'sentence-type': 'think', 'human_label': 'Read'},
            {'index': 2, 'sentence': 'Answer', 'sentence-type': 'answer', 'human_label': 'Answer'}]))


def test_converter_multiple_models(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for model in ['m1', 'm2']:
        fixture(tmp_path, model)
    module.main(['m1', 'm2', '--out', 'converted'])
    traces = [json.loads(line) for line in Path('converted/traces.jsonl').read_text().splitlines()]
    human = [json.loads(line) for line in Path('converted/human.jsonl').read_text().splitlines()]
    assert len(traces) == len(human) == 18
    assert traces[0] == {'id': 'm1-1', 'problem': 'Problem 1', 'sentences': ['Text 1', 'Answer']}
    assert traces[-1]['id'] == 'm2-9'
    assert human[0] == {'id': 'm1-1', 'sentences': [
        {'index': 0, 'text': 'Text 1', 'label': 'Read'}, {'index': 1, 'text': 'Answer', 'label': 'Answer'}]}
    assert [r['id'] for r in traces] == [r['id'] for r in human]


def test_missing_ground_truth_is_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fixture(tmp_path, 'm1')
    Path('ThinkARM/data/ground_truth/m1/9.json').unlink()
    with pytest.raises(FileNotFoundError, match='9.json'):
        module.main(['m1', '--out', 'out'])
    assert not Path('out/traces.jsonl').exists()
