"""Sentence labelers: prompt building, response parsing and an OpenAI-compatible client."""

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

from httpx import AsyncClient, HTTPStatusError, TransportError
from openai import APIConnectionError, APIStatusError, AsyncOpenAI

LABELS = ['Read', 'Analyze', 'Plan', 'Implement', 'Explore', 'Verify', 'Monitor', 'Answer']
DEFAULT_GUIDEBOOK = 'ThinkARM/guidebook/sentence_guide.md'
PROMPTS = Path(__file__).parent / 'prompts'
MAX_ATTEMPTS = 5
PLACEHOLDER_PATTERN = re.compile(r'\{(guidebook|problem|context|sentences|n)\}')


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
    """Fill the prompt templates for one batch of sentences.

    `context` is the text of all sentences preceding the batch.
    """
    if guidebook is None:
        guidebook = Path(DEFAULT_GUIDEBOOK).read_text(encoding='utf-8')
    values = dict(
        guidebook=guidebook,
        problem=problem,
        context=(
            f'The previous sentences are:\n\n {context}'
            if context
            else 'There is no previous sentences.'
        ),
        sentences=render_sentences(batch_sentences),
        n=str(len(batch_sentences)),
    )
    template = PROMPTS.joinpath('batch.md').read_text(encoding='utf-8')
    # A single substitution pass keeps tokens inside inserted text literal.
    batch = PLACEHOLDER_PATTERN.sub(lambda match: values[match[1]], template)
    instructions = PROMPTS.joinpath('instructions.md').read_text(encoding='utf-8')
    return instructions.strip() + '\n\n' + batch


def backoff(attempt):
    return min(30, 2**attempt + random.random())


def parse_predictions(content, count):
    """Parse a model response into `count` predictions ordered by sentence index.

    Raises ValueError if the response is malformed or incomplete.
    """
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


def _build_response_format(mode):
    if mode == 'json_object':
        return {'type': mode}
    if mode != 'json_schema':
        raise ValueError('response_format must be json_schema or json_object')
    item = {
        'type': 'object',
        'properties': {
            'index': {'type': 'integer'},
            'reason': {'type': 'string'},
            'category': {'type': 'string', 'enum': LABELS},
        },
        'required': ['index', 'reason', 'category'],
        'additionalProperties': False,
    }
    schema = {
        'type': 'object',
        'properties': {'sentences': {'type': 'array', 'items': item}},
        'required': ['sentences'],
        'additionalProperties': False,
    }
    return {
        'type': mode,
        'json_schema': {'name': 'sentence_labels', 'strict': True, 'schema': schema},
    }


def _is_retryable(exc):
    """Connection errors, rate limits and server errors are worth retrying."""
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or 500 <= exc.status_code < 600
    return True


class OpenAICompatLabeler:
    def __init__(self, config):
        self.guidebook = Path(config.get('guidebook', DEFAULT_GUIDEBOOK)).read_text(
            encoding='utf-8'
        )
        self.model = config['model']
        self.batch_size = config.get('batch_size', 20)
        self.semaphore = asyncio.Semaphore(config.get('concurrency', 16))
        base_url = config.get('base_url')
        key = os.getenv('LABELER_API_KEY') or os.getenv('OPENAI_API_KEY')
        self.client = AsyncOpenAI(
            base_url=base_url, api_key=key or ('unused' if base_url else None), max_retries=0
        )
        self.response_format = _build_response_format(config.get('response_format', 'json_schema'))
        self.options = {
            name: config[name]
            for name in ('temperature', 'max_tokens')
            if config.get(name) is not None
        }
        template_kwargs = config.get('chat_template_kwargs')
        if template_kwargs is not None:
            if not isinstance(template_kwargs, dict):
                raise ValueError('chat_template_kwargs must be a mapping or null')
            self.options['extra_body'] = {'chat_template_kwargs': template_kwargs}

    async def _batch(self, problem, context, sentences):
        """Label one batch, retrying invalid responses and transient API errors."""
        prompt = build_prompt(problem, context, sentences, guidebook=self.guidebook)
        for attempt in range(MAX_ATTEMPTS):
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{'role': 'user', 'content': prompt}],
                        response_format=self.response_format,
                        **self.options,
                    )
                try:
                    content = response.choices[0].message.content
                except (AttributeError, IndexError, TypeError) as exc:
                    raise ValueError('Missing response content') from exc
                return parse_predictions(content, len(sentences))
            except ValueError:
                if attempt == MAX_ATTEMPTS - 1:
                    raise
            except (APIStatusError, APIConnectionError) as exc:
                if not _is_retryable(exc) or attempt == MAX_ATTEMPTS - 1:
                    raise
                await sleep(backoff(attempt))

    async def label(self, problem: str, sentences: list[str]) -> list[Prediction]:
        """Label all sentences; each batch sees every earlier sentence as context."""
        tasks = [
            asyncio.create_task(
                self._batch(
                    problem,
                    '\n'.join(sentences[:start]),
                    sentences[start : start + self.batch_size],
                )
            )
            for start in range(0, len(sentences), self.batch_size)
        ]
        try:
            batches = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return [prediction for batch in batches for prediction in batch]


BASAL_TARGET = '\n\nTARGET SENTENCE:\n'


def load_basal_question(path):
    """Read the instructions (first paragraph) and `Label: description` criteria lines."""
    instructions, _, rest = Path(path).read_text(encoding='utf-8').strip().partition('\n\n')
    criteria = dict(line.split(': ', 1) for line in rest.splitlines() if line.strip())
    if list(criteria) != LABELS:
        raise ValueError(f'{path} must describe exactly these labels in order: {LABELS}')
    return {'type': 'choice', 'instructions': instructions, 'criteria': criteria}


def build_basal_state(problem, earlier, target):
    context = '\n'.join(earlier) if earlier else '(none)'
    return f'PROBLEM:\n{problem}\n\nEARLIER SENTENCES:\n{context}{BASAL_TARGET}{target}'


def parse_basal_answer(data):
    """Turn a /v1/systemone response into a Prediction; raise ValueError on an invalid choice."""
    try:
        answer = data['answers']['label']
        if answer['choice'] not in LABELS:
            raise ValueError(f'unknown choice {answer["choice"]!r}')
        return Prediction(answer['choice'], None, dict(answer['probabilities']))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Invalid basal response: {exc}') from exc


class BasalLabeler:
    """Labels each sentence with one /v1/systemone choice request to a basal-serve server."""

    def __init__(self, config):
        self.question = load_basal_question(config.get('guidebook', PROMPTS / 'basal.md'))
        self.url = config.get('base_url', 'http://localhost:8000').rstrip('/') + '/v1/systemone'
        self.semaphore = asyncio.Semaphore(config.get('concurrency', 16))
        self.client = AsyncClient(timeout=300)

    async def _sentence(self, problem, earlier, target):
        body = {
            'state': build_basal_state(problem, earlier, target),
            'questions': {'label': self.question},
        }
        for attempt in range(MAX_ATTEMPTS):
            try:
                async with self.semaphore:
                    response = await self.client.post(self.url, json=body)
                response.raise_for_status()
                break
            except (HTTPStatusError, TransportError) as exc:
                status = getattr(getattr(exc, 'response', None), 'status_code', None)
                retryable = status is None or status == 429 or status >= 500
                if not retryable or attempt == MAX_ATTEMPTS - 1:
                    raise
                await sleep(backoff(attempt))
        return parse_basal_answer(response.json())

    async def label(self, problem: str, sentences: list[str]) -> list[Prediction]:
        tasks = [
            asyncio.create_task(self._sentence(problem, sentences[:i], sentence))
            for i, sentence in enumerate(sentences)
        ]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


REGISTRY: dict[str, Callable[[dict], Labeler]] = {
    'openai': OpenAICompatLabeler,
    'basal': BasalLabeler,
}
