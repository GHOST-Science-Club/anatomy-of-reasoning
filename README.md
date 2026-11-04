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

## Test a model

Run commands from the repository root. Paths in the config are relative to this
directory.

1. Copy the credentials template and edit the new file to set your API key:

   ```bash
   cp secrets.example .env
   ```

   `OPENAI_API_KEY` is used for OpenAI. `LABELER_API_KEY`, when set, takes
   precedence. You can also set these in your shell; existing environment values
   take precedence over `.env`. Both `.env` and `.env.*` are gitignored.

2. Copy the example config, then edit `model` to the model you want to test:

   ```bash
   cp configs/example.yaml configs/experiment.yaml
   ```

   For an OpenAI-compatible server such as LiteLLM, set `base_url` to its API
   root, for example `http://localhost:4000/v1`, and set `model` to the model name
   exposed by that server. The client appends `/chat/completions` to this root.
   For example:

   ```yaml
   model: your-model-name
   base_url: https://your-server.example/v1
   response_format: json_object
   ```

   Use `response_format: json_object` if the server does
   not support strict JSON schemas. For a server that requires no key, leave
   `OPENAI_API_KEY` empty and `LABELER_API_KEY` unset; the client uses a dummy key.
   The server must accept OpenAI-compatible chat completion requests. An API with
   a different request format or endpoint, such as `/v1/systemone`, needs its own
   labeler implementation; changing `base_url` alone is insufficient.

3. Create a tiny input to check the connection and labeling flow:

   ```bash
   mkdir -p data
   cat > data/traces.jsonl <<'EOF_TRACES'
   {"id":"smoke-1","problem":"What is 2 + 2?","trace":"I need to add two and two. Two plus two equals four. The answer is 4."}
   EOF_TRACES
   ```

   If you already have traces, set `input` in your config to their JSONL path
   instead. Each line needs a unique string `id`, a string `problem`, and either
   a string `trace` or a list of strings named `sentences`.

4. Run the model. This sends requests to the configured provider:

   ```bash
   uv run python -m labeling label configs/experiment.yaml --run smoke-1
   ```

   Inspect `runs/smoke-1/labels.jsonl`: each sentence has its text, index, label,
   reason, and optional probabilities. The run also saves the YAML configuration
   and repository commit. This smoke test checks that labeling works; use the
   human-label evaluation below to measure accuracy.

Repeat the same command to resume missing or failed traces. Completed traces are
skipped, and failed traces produce a nonzero exit status. Only `concurrency` and
`traces_in_flight` may change when resuming. For a different model or other config
changes, use a new run name. Results are flushed after each completed trace.

## Evaluate against human labels

Convert ThinkARM's human-labeled prefixes for a source model (nine problems):

```bash
uv run python scripts/convert_thinkarm.py deepseekR1 --out data/thinkarm
```

Here `deepseekR1` selects the reasoning traces; `model` in your YAML selects the
model that labels those traces. The converter also accepts several source model
names in one command.

Change `input` in `configs/experiment.yaml` to `data/thinkarm/traces.jsonl`, then
label and score with a new run name:

```bash
uv run python -m labeling label configs/experiment.yaml --run thinkarm-1
uv run python -m labeling evaluate --labels runs/thinkarm-1/labels.jsonl --human data/thinkarm/human.jsonl
```

Evaluation prints accuracy, Cohen's kappa, scored sentence and trace counts,
skipped ids, and a confusion matrix. Missing traces and traces with mismatched
sentence text are reported and excluded from scoring. You can supply your own
human JSONL with the same ids and sentence `index`, `text`, and `label` fields.

## Development

Run the tests without making model requests:

```bash
uv run pytest
```

A new backend implements `Labeler.label(problem, sentences)` and registers a
factory in `labelers.REGISTRY`. Factories receive the entire YAML configuration;
return one `Prediction` per sentence, or raise to fail the trace.
