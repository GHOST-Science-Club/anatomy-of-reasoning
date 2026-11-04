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

## Evaluate a model against human labels

Run commands from the repository root. The included `configs/label.yaml`
already points to the evaluation data created below. Use it directly; no config
copy or input-path edit is needed for this workflow.

1. Set your API key in your shell or an existing `.env`. To create a credentials
   file from the template if needed:

   ```bash
   cp -n secrets.example .env
   ```

   Edit `.env` to fill in `OPENAI_API_KEY`. For another provider, you can set
   `LABELER_API_KEY` instead; it takes precedence. Existing shell values take
   precedence over `.env`. Both `.env` and `.env.*` are gitignored.

2. Open `configs/label.yaml`. Set `model` to the model you want to evaluate
   and `base_url` to your provider's API root. For OpenAI, use
   `https://api.openai.com/v1`.
   If using an OpenAI-compatible server such as LiteLLM, fill in its `base_url`
   (for example, `http://localhost:4000/v1`) and the model name it serves.

   The client appends `/chat/completions` to `base_url`. Keep
   `response_format: json_schema` if supported; otherwise set it to
   `json_object`. For a server that requires no key, leave `OPENAI_API_KEY`
   empty and `LABELER_API_KEY` unset; the client uses a dummy key. APIs with a
   different request format or endpoint, such as `/v1/systemone`, need their own
   labeler implementation.

3. Convert ThinkARM's human-annotated sentence prefixes:

   ```bash
   uv run python scripts/convert_thinkarm.py deepseekR1 --human-only --out data/thinkarm
   ```

   This creates `data/thinkarm/traces.jsonl` and `data/thinkarm/human.jsonl`.
   `deepseekR1` selects the source of the reasoning traces; `model` in the YAML
   selects the model that labels them. There are nine annotated prefixes for
   this source, so you only pay to label sentences that can be scored.

4. Label the prefixes using the included config. This sends requests to your
   configured provider:

   ```bash
   uv run python -m labeling label configs/label.yaml --run thinkarm-1
   ```

5. Compare the predictions against the human labels:

   ```bash
   uv run python -m labeling evaluate --labels runs/thinkarm-1/labels.jsonl --human data/thinkarm/human.jsonl
   ```

Evaluation prints accuracy, Cohen's kappa, scored sentence and trace counts,
skipped ids, and a confusion matrix. Missing traces and traces with mismatched
sentence text are reported and excluded from scoring.

Predictions are saved to `runs/thinkarm-1/labels.jsonl`, along with the config and
repository commit. Repeat the labeling command to resume missing or failed
traces; completed traces are skipped. Failures produce a nonzero exit status.
Only `concurrency` and `traces_in_flight` may change when resuming. For a different
model or other config changes, use a new run name. Paths in the config are
relative to the repository root.

## Control reasoning on vLLM

The included `configs/label.yaml` disables thinking for vLLM models whose chat
template supports `enable_thinking`:

```yaml
chat_template_kwargs:
  enable_thinking: false
```

Some model templates use `thinking: false` instead; use the switch supported by
your served model. Set the switch to `true` to enable thinking explicitly, or
leave `chat_template_kwargs: null` to use the server default. Keep it `null` for
OpenAI or servers that do not support chat template arguments.

Use a new run name after changing this setting. The per-sentence `reason` in the
label output is still requested; it is separate from the model's thinking mode.

## Label full ThinkARM traces

Without `--human-only`, the converter exports every full reasoning trace,
including those without human annotations:

```bash
uv run python scripts/convert_thinkarm.py deepseekR1 --out data/thinkarm-all
```

For a smaller experiment, use `--limit` to take the first N traces per source
model:

```bash
uv run python scripts/convert_thinkarm.py deepseekR1 --limit 20 --out data/thinkarm-20
```

Set `input` in `configs/label.yaml` to `data/thinkarm-20/traces.jsonl`, then
label them with a new run name:

```bash
uv run python -m labeling label configs/label.yaml --run thinkarm-20
```

Both modes write `traces.jsonl` and `human.jsonl`. The latter contains only the
available human labels for the exported traces and is empty if none exist.
You can evaluate a full-trace run against that file; only the annotated prefixes
are scored. `--limit` also works with `--human-only`, taking the first N annotated
problems per model. Multiple source models can be given before the options;
problem ids stay the same across all modes.

## Development

Run the tests without making model requests:

```bash
uv run pytest
```

A new backend implements `Labeler.label(problem, sentences)` and registers a
factory in `labelers.REGISTRY`. Factories receive the entire YAML configuration;
return one `Prediction` per sentence, or raise to fail the trace.
