import json

import pytest

from labeling.evaluate import evaluate
from labeling.labelers import LABELS


def record(id, labels, texts=None, indices=None):
    return {'id': id, 'sentences': [
        {'index': i if indices is None else indices[i], 'text': text, 'label': label}
        for i, (text, label) in enumerate(zip(texts or [f's{i}' for i in range(len(labels))], labels))]}


def score(tmp_path, predicted, human):
    paths = []
    for name, rows in [('labels', predicted), ('human', human)]:
        path = tmp_path / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        paths.append(path)
    return evaluate(*paths)


def test_perfect_and_hand_computed_scores(tmp_path, capsys):
    human = [record('a', ['Read', 'Answer', 'Read', 'Answer'])]
    perfect = score(tmp_path, human, human)
    assert perfect['accuracy'] == perfect['kappa'] == 1.0
    result = score(tmp_path, [record('a', ['Read', 'Read', 'Read', 'Answer'])], human)
    assert result['accuracy'] == 0.75
    assert result['kappa'] == 0.5
    assert result['sentences'] == 4
    assert result['traces'] == 1
    matrix = result['confusion']
    assert len(matrix) == len(matrix[0]) == len(LABELS)
    assert matrix[LABELS.index('Answer')][LABELS.index('Read')] == 1
    assert matrix[LABELS.index('Read')][LABELS.index('Answer')] == 0
    output = capsys.readouterr().out
    assert 'Accuracy' in output and 'Kappa' in output
    assert 'human' in output and 'predicted' in output


def test_text_mismatch_and_missing_trace_are_reported(tmp_path, capsys):
    good = record('good', ['Read', 'Answer'])
    result = score(tmp_path, [good, record('bad', ['Read'], ['different'])],
                   [good, record('bad', ['Read']), record('missing', ['Read'])])
    assert result['accuracy'] == 1.0
    assert result['sentences'] == 2 and result['traces'] == 1
    assert result['skipped'] == ['bad', 'missing']
    output = capsys.readouterr().out
    assert 'bad' in output and 'index 0' in output
    assert 'missing' in output and 'Missing' in output


def test_index_alignment_and_extra_predictions(tmp_path):
    result = score(tmp_path, [record('x', ['Answer', 'Read', 'Explore'], ['s1', 's0', 'extra'], [1, 0, 2])],
                   [record('x', ['Read', 'Answer'])])
    assert result['accuracy'] == result['kappa'] == 1.0
    assert result['sentences'] == 2


def test_missing_index_excludes_whole_trace(tmp_path, capsys):
    result = score(tmp_path, [record('x', ['Read'])], [record('x', ['Read', 'Answer'])])
    assert result['skipped'] == ['x']
    assert result['traces'] == result['sentences'] == 0
    assert result['accuracy'] is None and result['kappa'] is None
    assert 'index 1' in capsys.readouterr().out
