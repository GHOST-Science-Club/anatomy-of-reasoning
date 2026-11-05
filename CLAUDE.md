# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Automatic analysis of LLM reasoning. ThinkARM (git submodule in `ThinkARM/`, never edit) is the starting point; the repo is expected to grow beyond it. Keep code very simple with minimal boilerplate; add comments and docstrings only where really needed. Tests and ruff are recommended for agentic workflows, not strictly required. See `README.md` for the full evaluation walkthrough.

## Commands

Python >= 3.13, managed with `uv`. Run everything from the repo root (paths such as `ThinkARM/...`, `runs/` and `configs/` are relative to it).

```bash
git submodule update --init --recursive   # if ThinkARM is empty
uv sync
uv run pytest                              # offline; no API calls
uv run pytest tests/test_run.py::test_name # single test
uv run ruff check . && uv run ruff format .
uv run python scripts/convert_thinkarm.py deepseekR1 --human-only --out data/thinkarm
uv run python -m labeling label configs/llm.yaml --run <name>
uv run python -m labeling evaluate --labels runs/<name>/labels.jsonl --human data/thinkarm/human.jsonl
```

## Architecture

Pipeline: `scripts/convert_thinkarm.py` (ThinkARM raw/ground-truth JSON -> `traces.jsonl` + `human.jsonl`) -> `labeling label` (`run.py`) -> `runs/<name>/labels.jsonl` -> `labeling evaluate` (`evaluate.py`, accuracy/kappa/confusion vs `human.jsonl`).

- `run.py` orchestrates a run: validates the YAML config and JSONL input, creates/resumes `runs/<name>/` (stores a config copy and git commit; on resume only `concurrency` and `traces_in_flight` may differ; a truncated last line of `labels.jsonl` is repaired), and runs `traces_in_flight` workers that append one result line per trace. Any failed trace gives a nonzero exit.
- Input records carry either `trace` (raw text, split into sentences by `splitting.py`) or pre-split `sentences`, never both.
- `labelers.py` holds `LABELS`, `Prediction`, the `Labeler` protocol, prompt building (`prompts/*.md` filled in a single regex pass so inserted text stays literal) and `OpenAICompatLabeler`. It labels sentences in batches, each batch seeing all earlier sentences as context, with retries on invalid responses and 429/5xx/connection errors. New backends are registered in `labelers.REGISTRY` and selected by the config's `labeler` key.
- `splitting.py` is a heuristic sentence splitter ported from ThinkARM (math spans, abbreviations, decimals, `</think>` handling, colon/`=` merging). Its exact output defines sentence indices, so changes silently break comparison with human labels.
- API keys come from `LABELER_API_KEY` or `OPENAI_API_KEY` (env or `.env`); `redact_error` masks them in all error output.

## Testing notes

Tests monkeypatch module-level names (`AsyncOpenAI`, `sleep` in `labelers`; `load_dotenv`, `REGISTRY`, `process_response_to_sentences` in `run`), so keep those names importable at module level. `tests/test_convert.py` loads the script by file path.
