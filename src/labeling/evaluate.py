"""Score predicted sentence labels against human labels."""

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from .labelers import LABELS


def _read(path):
    rows = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['id'] in rows:
            raise ValueError(f'Duplicate id {row["id"]!r} in {path}')
        rows[row['id']] = row
    return rows


def _pair_labels(trace_id, human_sentences, predicted_sentences):
    """Return (human, predicted) label pairs, or None if the trace cannot be scored.

    A trace is unscorable when a predicted sentence is missing or its text
    differs from the human-labeled one.
    """
    by_index = {s['index']: s for s in predicted_sentences}
    pairs = []
    for sentence in human_sentences:
        index = sentence['index']
        if index not in by_index:
            print(f'Missing sentence: {trace_id}, index {index}')
            return None
        other = by_index[index]
        if sentence['text'] != other['text']:
            print(f'Text mismatch: {trace_id}, index {index}')
            return None
        if sentence['label'] not in LABELS or other['label'] not in LABELS:
            raise ValueError(f'Unknown label for {trace_id}, index {index}')
        pairs.append((sentence['label'], other['label']))
    return pairs


def _print_report(result):
    print(f'Accuracy: {result["accuracy"]}\nKappa: {result["kappa"]}')
    print(f'Scored sentences: {result["sentences"]}; traces: {result["traces"]}')
    print(f'Skipped ids: {", ".join(result["skipped"]) or "none"}')
    print('Confusion matrix (human rows, predicted columns):')
    print('          ' + ' '.join(f'{label:>9}' for label in LABELS))
    for label, row in zip(LABELS, result['confusion'], strict=True):
        print(f'{label:>9} ' + ' '.join(f'{value:>9}' for value in row))


def evaluate(labels_path, human_path):
    """Compare predictions with human labels, print a report and return the metrics."""
    predicted, human = _read(labels_path), _read(human_path)
    expected_labels, predicted_labels, skipped = [], [], []
    traces = 0
    for trace_id, record in human.items():
        if trace_id not in predicted:
            print(f'Missing id: {trace_id}')
            skipped.append(trace_id)
            continue
        pairs = _pair_labels(trace_id, record['sentences'], predicted[trace_id]['sentences'])
        if pairs is None:
            skipped.append(trace_id)
        elif pairs:
            traces += 1
            expected_labels.extend(pair[0] for pair in pairs)
            predicted_labels.extend(pair[1] for pair in pairs)
    result = dict(
        accuracy=None,
        kappa=None,
        sentences=len(expected_labels),
        traces=traces,
        skipped=skipped,
        confusion=[[0] * len(LABELS) for _ in LABELS],
    )
    if expected_labels:
        result.update(
            accuracy=float(accuracy_score(expected_labels, predicted_labels)),
            kappa=float(cohen_kappa_score(expected_labels, predicted_labels, labels=LABELS)),
            confusion=confusion_matrix(expected_labels, predicted_labels, labels=LABELS).tolist(),
        )
    _print_report(result)
    return result
