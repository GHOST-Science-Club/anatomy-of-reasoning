# Anatomy of Reasoning

## Setup

1. Install [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/).

2. Clone the repo (the `--recurse-submodules` flag also downloads ThinkARM):

   ```bash
   git clone --recurse-submodules <repo-url>
   cd anatomy-of-reasoning
   ```

   Already cloned without it? Run `git submodule update --init --recursive`.

3. Install the Python packages:

   ```bash
   uv sync
   ```

## Sentence labeling

Input is JSONL with unique string `id`, `problem`, and either `trace` or a
`sentences` list. Create a YAML file in `configs/`, for example:

```yaml
input: data/traces.jsonl
labeler: openai
model: gpt-5
base_url: null
batch_size: 20
concurrency: 16
traces_in_flight: 2
response_format: json_schema
temperature: null
max_tokens: null
guidebook: ThinkARM/guidebook/sentence_guide.md
```

Set `OPENAI_API_KEY` in your environment or `.env`. `LABELER_API_KEY` takes
precedence; a local `base_url` without a key uses a dummy key. Use
`response_format: json_object` if your server lacks strict schema support.
Paths are relative to the current working directory; run commands from the repo root.

```bash
uv run python -m labeling label configs/experiment.yaml --run experiment-1
uv run python -m labeling evaluate --labels runs/experiment-1/labels.jsonl --human data/human.jsonl
```

Results are flushed per completed trace to `runs/experiment-1/labels.jsonl`.
Repeat the label command to resume missing or failed traces. Failures produce a
nonzero exit status. Only `concurrency` and `traces_in_flight` may change when
resuming; other configuration changes require a new run name.

To convert ThinkARM human-labeled prefixes (all nine problems per model):

```bash
uv run python scripts/convert_thinkarm.py deepseekR1 --out data/thinkarm
```

A new backend implements `Labeler.label(problem, sentences)` and registers a
factory in `labelers.REGISTRY`. Factories receive the entire YAML configuration;
return one `Prediction` per sentence, or raise to fail the trace.

Run the offline tests with `uv run pytest`.
