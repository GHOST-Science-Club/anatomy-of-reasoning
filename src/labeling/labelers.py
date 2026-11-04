import asyncio
from asyncio import sleep
from collections.abc import Callable
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import re
from typing import Protocol

from openai import AsyncOpenAI, APIConnectionError, APIStatusError

LABELS = ['Read', 'Analyze', 'Plan', 'Implement', 'Explore', 'Verify', 'Monitor', 'Answer']
DEFAULT_GUIDEBOOK = 'ThinkARM/guidebook/sentence_guide.md'
PROMPTS = Path(__file__).parent / 'prompts'


@dataclass
class Prediction:
    label: str
    reason: str | None = None
    probs: dict[str, float] | None = None


class Labeler(Protocol):
    async def label(self, problem: str, sentences: list[str]) -> list[Prediction]: ...


def render_sentences(sentences: list[str]) -> str:
    return '\n'.join(f'[{i}] {text}' for i, text in enumerate(sentences, 1))


def build_prompt(problem, context, batch_sentences, *, guidebook=None):
    if guidebook is None:
        guidebook = Path(DEFAULT_GUIDEBOOK).read_text(encoding='utf-8')
    values = dict(guidebook=guidebook, problem=problem,
                  context=f'The previous sentences are:\n\n {context}' if context else 'There is no previous sentences.',
                  sentences=render_sentences(batch_sentences), n=str(len(batch_sentences)))
    template = PROMPTS.joinpath('batch.md').read_text(encoding='utf-8')
    # A single substitution pass keeps tokens inside inserted text literal.
    batch = re.sub(r'\{(guidebook|problem|context|sentences|n)\}', lambda m: values[m[1]], template)
    return PROMPTS.joinpath('instructions.md').read_text(encoding='utf-8').strip() + '\n\n' + batch


def backoff(attempt):
    return min(30, 2 ** attempt + random.random())


def parse_predictions(content, count):
    try:
        rows = json.loads(content)['sentences']
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError('wrong sentence count')
        indexed = {}
        canonical = {label.lower(): label for label in LABELS}
        for row in rows:
            index = row['index']
            if type(index) is not int or not 1 <= index <= count or index in indexed:
                raise ValueError('invalid sentence indices')
            if not isinstance(row['reason'], str):
                raise ValueError('invalid reason')
            indexed[index] = Prediction(canonical[row['category'].lower()], row['reason'])
        return [indexed[i] for i in range(1, count + 1)]
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError(f'Invalid labeling response: {exc}') from exc


class OpenAICompatLabeler:
    def __init__(self, config):
        self.guidebook = Path(config.get('guidebook', DEFAULT_GUIDEBOOK)).read_text(encoding='utf-8')
        self.model = config['model']
        self.batch_size = config.get('batch_size', 20)
        self.semaphore = asyncio.Semaphore(config.get('concurrency', 16))
        base_url = config.get('base_url')
        key = os.getenv('LABELER_API_KEY') or os.getenv('OPENAI_API_KEY')
        self.client = AsyncOpenAI(base_url=base_url, api_key=key or ('unused' if base_url else None), max_retries=0)
        mode = config.get('response_format', 'json_schema')
        if mode == 'json_object':
            self.response_format = {'type': mode}
        elif mode == 'json_schema':
            item = {'type': 'object', 'properties': {
                'index': {'type': 'integer'}, 'reason': {'type': 'string'},
                'category': {'type': 'string', 'enum': LABELS}},
                'required': ['index', 'reason', 'category'], 'additionalProperties': False}
            self.response_format = {'type': mode, 'json_schema': {
                'name': 'sentence_labels', 'strict': True, 'schema': {
                    'type': 'object', 'properties': {'sentences': {'type': 'array', 'items': item}},
                    'required': ['sentences'], 'additionalProperties': False}}}
        else:
            raise ValueError('response_format must be json_schema or json_object')
        self.options = {key: config[key] for key in ('temperature', 'max_tokens') if config.get(key) is not None}

    async def _batch(self, problem, context, sentences):
        prompt = build_prompt(problem, context, sentences, guidebook=self.guidebook)
        for attempt in range(5):
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model, messages=[{'role': 'user', 'content': prompt}],
                        response_format=self.response_format, **self.options)
                try:
                    content = response.choices[0].message.content
                except (AttributeError, IndexError, TypeError) as exc:
                    raise ValueError('Missing response content') from exc
                return parse_predictions(content, len(sentences))
            except ValueError:
                if attempt == 4:
                    raise
            except (APIStatusError, APIConnectionError) as exc:
                if isinstance(exc, APIStatusError) and exc.status_code != 429 and not 500 <= exc.status_code < 600:
                    raise
                if attempt == 4:
                    raise
                await sleep(backoff(attempt))

    async def label(self, problem: str, sentences: list[str]) -> list[Prediction]:
        tasks = [asyncio.create_task(self._batch(problem, '\n'.join(sentences[:s]), sentences[s:s + self.batch_size]))
                 for s in range(0, len(sentences), self.batch_size)]
        try:
            batches = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return [prediction for batch in batches for prediction in batch]


REGISTRY: dict[str, Callable[[dict], Labeler]] = {'openai': OpenAICompatLabeler}
