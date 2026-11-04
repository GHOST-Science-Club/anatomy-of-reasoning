import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess

from dotenv import load_dotenv
from tqdm import tqdm
import yaml

from .labelers import LABELS, REGISTRY
from .splitting import process_response_to_sentences


DEFAULTS = dict(batch_size=20, concurrency=16, traces_in_flight=2, response_format='json_schema')


def redact_error(error):
    text = str(error)
    for key in ('OPENAI_API_KEY', 'LABELER_API_KEY'):
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
    if config.get('response_format', DEFAULTS['response_format']) not in ('json_schema', 'json_object'):
        raise ValueError('response_format must be json_schema or json_object')
    if config['labeler'] not in REGISTRY:
        raise ValueError(f'Unknown labeler {config["labeler"]!r}; registered: {", ".join(REGISTRY)}')
    return config


def read_input(path):
    records, ids = [], set()
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not isinstance(row.get('problem'), str):
                raise ValueError('id and problem must be strings')
            if ('trace' in row) == ('sentences' in row):
                raise ValueError('provide exactly one of trace or sentences')
            if 'trace' in row and not isinstance(row['trace'], str):
                raise ValueError('trace must be a string')
            if 'sentences' in row and (not isinstance(row['sentences'], list) or not all(isinstance(s, str) for s in row['sentences'])):
                raise ValueError('sentences must be a list of strings')
            if row['id'] in ids:
                raise ValueError(f'duplicate id {row["id"]!r}')
            ids.add(row['id'])
            records.append(row)
        except ValueError as exc:
            raise ValueError(f'{path}, line {number}: {exc}') from exc
    return records


def prepare_run(config_path, config, name):
    if not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*', name):
        raise ValueError('Invalid run name; use a single folder name with letters, numbers, dots, underscores or hyphens')
    folder = Path('runs') / name
    if folder.exists():
        stored = yaml.safe_load((folder / 'config.yaml').read_text(encoding='utf-8'))
        scheduling = {'concurrency', 'traces_in_flight'}
        if {k: v for k, v in config.items() if k not in scheduling} != {k: v for k, v in stored.items() if k not in scheduling}:
            raise ValueError('Configuration differs from the saved config; start a new run')
    else:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2], text=True).strip()
        folder.mkdir(parents=True)
        (folder / 'config.yaml').write_bytes(Path(config_path).read_bytes())
        (folder / 'meta.json').write_text(json.dumps({'commit': commit}) + '\n', encoding='utf-8')
    output = folder / 'labels.jsonl'
    completed = set()
    if output.exists():
        with output.open('rb+') as stream:
            lines = stream.readlines()
            offset = 0
            for i, line in enumerate(lines):
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    if i != len(lines) - 1:
                        raise ValueError(f'Invalid JSON in {output} at line {i + 1}')
                    stream.truncate(offset)
                    break
                completed.add(row['id'])
                offset += len(line)
            else:
                if lines and not lines[-1].endswith(b'\n'):
                    stream.seek(0, 2)
                    stream.write(b'\n')
    return output, completed


async def run(config_path, name):
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
    with output.open('a', encoding='utf-8') as stream, tqdm(total=len(pending), unit='trace') as progress:
        async def worker():
            nonlocal succeeded
            for row in iterator:
                try:
                    sentences = row['sentences'] if 'sentences' in row else process_response_to_sentences(row['trace'])
                    predictions = await labeler.label(row['problem'], sentences) if sentences else []
                    if len(predictions) != len(sentences) or any(p.label not in LABELS for p in predictions):
                        raise ValueError('Labeler returned invalid predictions or sentence count')
                    result = {'id': row['id'], 'sentences': [
                        {'index': i, 'text': text, **asdict(prediction)}
                        for i, (text, prediction) in enumerate(zip(sentences, predictions))]}
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                    stream.flush()
                    succeeded += 1
                except Exception as exc:
                    failed.append(row['id'])
                    tqdm.write(f'Failed {row["id"]}: {redact_error(exc)}')
                finally:
                    progress.update(1)
        await asyncio.gather(*(worker() for _ in range(min(config.get('traces_in_flight', 2), len(pending)))))
    print(f'Succeeded: {succeeded}; failed ids: {", ".join(failed) or "none"}')
    return int(bool(failed))
