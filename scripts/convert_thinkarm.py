import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Convert ThinkARM human-labeled prefixes to JSONL')
    parser.add_argument('models', nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    traces, human = [], []
    for model in args.models:
        raw = json.loads(Path(f'ThinkARM/data/raw/{model}.json').read_text(encoding='utf-8'))
        for number in range(1, 10):
            ground = Path(f'ThinkARM/data/ground_truth/{model}/{number}.json')
            sentences = json.loads(ground.read_text(encoding='utf-8'))
            id = f'{model}-{number}'
            traces.append({'id': id, 'problem': raw[number - 1]['Instruction'],
                           'sentences': [s['sentence'] for s in sentences]})
            human.append({'id': id, 'sentences': [
                {'index': index, 'text': s['sentence'], 'label': s['human_label']}
                for index, s in enumerate(sentences)]})
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in [('traces', traces), ('human', human)]:
        (args.out / f'{name}.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


if __name__ == '__main__':
    main()
