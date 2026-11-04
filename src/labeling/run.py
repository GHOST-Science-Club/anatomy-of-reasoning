"""Run a labeling job: load config and input, label traces concurrently, write resumable output.

Each run lives in `runs/<name>/` with the config copy, git commit metadata and
`labels.jsonl`. Re-running the same name skips already labeled traces.
"""

import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess
import time

from dotenv import load_dotenv
from tqdm import tqdm
import yaml

from .labelers import LABELS, REGISTRY
from .splitting import process_response_to_sentences

DEFAULTS = dict(batch_size=20, concurrency=16, traces_in_flight=2, response_format='json_schema')
RESPONSE_FORMATS = ('json_schema', 'json_object')
API_KEY_VARIABLES = ('OPENAI_API_KEY', 'LABELER_API_KEY')
RUN_NAME_PATTERN = re.compile(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*')
REPO_ROOT = Path(__file__).resolve().parents[2]
# Settings that may change between resumed runs of the same name.
SCHEDULING_KEYS = frozenset({'concurrency', 'traces_in_flight'})


def redact_error(error):
    """Return the error text with API keys from the environment masked."""
    text = str(error)
    for key in API_KEY_VARIABLES:
        value = os.getenv(key)
        if value:
            text = text.replace(value, '[redacted]')
    return text


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Config must be a YAML mapping')
    for key in ('input', 'labeler', 'model'):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f'Config requires {key}')
    for key in ('batch_size', 'concurrency', 'traces_in_flight'):
        value = config.get(key, DEFAULTS[key])
        if type(value) is not int or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config.get('response_format', DEFAULTS['response_format']) not in RESPONSE_FORMATS:
        raise ValueError('response_format must be json_schema or json_object')
    if config['labeler'] not in REGISTRY:
        raise ValueError(
            f'Unknown labeler {config["labeler"]!r}; registered: {", ".join(REGISTRY)}'
        )
    return config


def _validate_row(row):
    if (
        not isinstance(row, dict)
        or not isinstance(row.get('id'), str)
        or not isinstance(row.get('problem'), str)
    ):
        raise ValueError('id and problem must be strings')
    if ('trace' in row) == ('sentences' in row):
        raise ValueError('provide exactly one of trace or sentences')
    if 'trace' in row and not isinstance(row['trace'], str):
        raise ValueError('trace must be a string')
    if 'sentences' in row and (
        not isinstance(row['sentences'], list)
        or not all(isinstance(s, str) for s in row['sentences'])
    ):
        raise ValueError('sentences must be a list of strings')


def read_input(path):
    """Read and validate the JSONL input, rejecting duplicate ids."""
    records, ids = [], set()
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        try:
            row = json.loads(line)
            _validate_row(row)
            if row['id'] in ids:
                raise ValueError(f'duplicate id {row["id"]!r}')
            ids.add(row['id'])
            records.append(row)
        except ValueError as exc:
            raise ValueError(f'{path}, line {number}: {exc}') from exc
    return records


def _without_scheduling(config):
    return {k: v for k, v in config.items() if k not in SCHEDULING_KEYS}


def _create_run_folder(folder, config_path):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
    folder.mkdir(parents=True)
    (folder / 'config.yaml').write_bytes(Path(config_path).read_bytes())
    (folder / 'meta.json').write_text(json.dumps({'commit': commit}) + '\n', encoding='utf-8')


def _recover_completed(output):
    """Return ids already in `output`, repairing a truncated final line.

    An unparseable last line (an interrupted write) is cut off; an unparseable
    earlier line is an error.
    """
    completed = set()
    with output.open('rb+') as stream:
        lines = stream.readlines()
        offset = 0
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if i != len(lines) - 1:
                    raise ValueError(f'Invalid JSON in {output} at line {i + 1}') from None
                stream.truncate(offset)
                break
            completed.add(row['id'])
            offset += len(line)
        else:
            if lines and not lines[-1].endswith(b'\n'):
                stream.seek(0, 2)
                stream.write(b'\n')
    return completed


def prepare_run(config_path, config, name):
    """Create or resume `runs/<name>`; return the output path and completed ids."""
    if not RUN_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            'Invalid run name; use a single folder name with letters, numbers, dots, '
            'underscores or hyphens'
        )
    folder = Path('runs') / name
    if folder.exists():
        stored = yaml.safe_load((folder / 'config.yaml').read_text(encoding='utf-8'))
        if _without_scheduling(config) != _without_scheduling(stored):
            raise ValueError('Configuration differs from the saved config; start a new run')
    else:
        _create_run_folder(folder, config_path)
    output = folder / 'labels.jsonl'
    completed = _recover_completed(output) if output.exists() else set()
    return output, completed


async def _label_row(labeler, row):
    sentences = (
        row['sentences'] if 'sentences' in row else process_response_to_sentences(row['trace'])
    )
    predictions = await labeler.label(row['problem'], sentences) if sentences else []
    if len(predictions) != len(sentences) or any(p.label not in LABELS for p in predictions):
        raise ValueError('Labeler returned invalid predictions or sentence count')
    return {
        'id': row['id'],
        'sentences': [
            {'index': i, 'text': text, **asdict(prediction)}
            for i, (text, prediction) in enumerate(zip(sentences, predictions, strict=True))
        ],
    }


async def run(config_path, name):
    """Label all pending traces of a config; return 1 if any failed, else 0."""
    load_dotenv()
    config = load_config(config_path)
    records = read_input(config['input'])
    output, completed = prepare_run(config_path, config, name)
    pending = [row for row in records if row['id'] not in completed]
    succeeded, failed = 0, []
    if not pending:
        print(f'Succeeded: 0; already completed: {len(completed)}; failed ids: none')
        return 0
    labeler = REGISTRY[config['labeler']](config)
    iterator = iter(pending)
    sentence_count = 0
    start = time.perf_counter()
    with (
        output.open('a', encoding='utf-8') as stream,
        tqdm(total=len(pending), unit='trace') as progress,
    ):

        async def worker():
            nonlocal succeeded, sentence_count
            for row in iterator:
                try:
                    result = await _label_row(labeler, row)
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                    stream.flush()
                    succeeded += 1
                    sentence_count += len(result['sentences'])
                except Exception as exc:
                    failed.append(row['id'])
                    tqdm.write(f'Failed {row["id"]}: {redact_error(exc)}')
                finally:
                    progress.update(1)

        workers = min(config.get('traces_in_flight', DEFAULTS['traces_in_flight']), len(pending))
        await asyncio.gather(*(worker() for _ in range(workers)))
    elapsed = time.perf_counter() - start
    print(f'Succeeded: {succeeded}; failed ids: {", ".join(failed) or "none"}')
    if succeeded:
        print(
            f'Elapsed: {elapsed:.1f}s; {elapsed / succeeded:.2f}s per trace; '
            f'{elapsed / max(sentence_count, 1):.3f}s per sentence ({sentence_count} sentences)'
        )
    return int(bool(failed))
