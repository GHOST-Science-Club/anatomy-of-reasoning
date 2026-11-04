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


def evaluate(labels_path, human_path):
    predicted, human = _read(labels_path), _read(human_path)
    expected_labels, predicted_labels, skipped = [], [], []
    traces = 0
    for id, record in human.items():
        if id not in predicted:
            print(f'Missing id: {id}')
            skipped.append(id)
            continue
        by_index = {s['index']: s for s in predicted[id]['sentences']}
        pairs = []
        for sentence in record['sentences']:
            index = sentence['index']
            if index not in by_index:
                print(f'Missing sentence: {id}, index {index}')
                break
            other = by_index[index]
            if sentence['text'] != other['text']:
                print(f'Text mismatch: {id}, index {index}')
                break
            if sentence['label'] not in LABELS or other['label'] not in LABELS:
                raise ValueError(f'Unknown label for {id}, index {index}')
            pairs.append((sentence['label'], other['label']))
        else:
            if pairs:
                traces += 1
                expected_labels.extend(pair[0] for pair in pairs)
                predicted_labels.extend(pair[1] for pair in pairs)
            continue
        skipped.append(id)
    result = dict(accuracy=None, kappa=None, sentences=len(expected_labels), traces=traces,
                  skipped=skipped, confusion=[[0] * len(LABELS) for _ in LABELS])
    if expected_labels:
        result.update(accuracy=float(accuracy_score(expected_labels, predicted_labels)),
                      kappa=float(cohen_kappa_score(expected_labels, predicted_labels, labels=LABELS)),
                      confusion=confusion_matrix(expected_labels, predicted_labels, labels=LABELS).tolist())
    print(f'Accuracy: {result["accuracy"]}\nKappa: {result["kappa"]}')
    print(f'Scored sentences: {result["sentences"]}; traces: {traces}')
    print(f'Skipped ids: {", ".join(skipped) or "none"}')
    print('Confusion matrix (human rows, predicted columns):')
    print('          ' + ' '.join(f'{label:>9}' for label in LABELS))
    for label, row in zip(LABELS, result['confusion']):
        print(f'{label:>9} ' + ' '.join(f'{value:>9}' for value in row))
    return result
