"""Command-line entry point: `python -m labeling {label,evaluate}`."""

import argparse
import asyncio

from .run import redact_error, run


def build_parser():
    parser = argparse.ArgumentParser(prog='labeling')
    commands = parser.add_subparsers(dest='command', required=True)
    label = commands.add_parser('label')
    label.add_argument('config')
    label.add_argument('--run', required=True)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--labels', required=True)
    evaluate.add_argument('--human', required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == 'label':
            return asyncio.run(run(args.config, args.run))
        # Imported lazily so labeling does not require scikit-learn.
        from .evaluate import evaluate as score

        score(args.labels, args.human)
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(1, f'error: {redact_error(exc)}\n')


if __name__ == '__main__':
    raise SystemExit(main())
