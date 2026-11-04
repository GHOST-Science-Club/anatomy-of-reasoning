import argparse
import json
from pathlib import Path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description='Convert ThinkARM reasoning traces and available human labels to JSONL')
    parser.add_argument('models', nargs='+', help='source model names under ThinkARM/data/raw')
    parser.add_argument('--out', type=Path, required=True, help='destination for traces.jsonl and human.jsonl')
    parser.add_argument('--limit', type=positive_int, help='maximum traces per model, after filtering')
    parser.add_argument('--human-only', action='store_true', help='export only human-annotated sentence prefixes for evaluation')
    args = parser.parse_args(argv)
    traces, human = [], []
    for model in args.models:
        raw = json.loads(Path(f'ThinkARM/data/raw/{model}.json').read_text(encoding='utf-8'))
        count = 0
        for number, row in enumerate(raw, 1):
            if args.limit is not None and count >= args.limit:
                break
            ground = Path(f'ThinkARM/data/ground_truth/{model}/{number}.json')
            if args.human_only and not ground.is_file():
                continue
            sentences = json.loads(ground.read_text(encoding='utf-8')) if ground.is_file() else None
            id = f'{model}-{number}'
            trace = {'id': id, 'problem': row['Instruction']}
            if args.human_only:
                trace['sentences'] = [s['sentence'] for s in sentences]
            else:
                trace['trace'] = row['Response']
            traces.append(trace)
            count += 1
            if sentences is not None:
                human.append({'id': id, 'sentences': [
                    {'index': index, 'text': s['sentence'], 'label': s['human_label']}
                    for index, s in enumerate(sentences)]})
        if args.human_only and not count:
            parser.error(f'No human labels found for {model}')
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in [('traces', traces), ('human', human)]:
        (args.out / f'{name}.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    print(f'Wrote {len(traces)} traces and {len(human)} human-labeled prefixes to {args.out}')


if __name__ == '__main__':
    main()
