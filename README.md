# Anatomy of Reasoning

## Setup

Install [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone --recurse-submodules <repo-url>
cd anatomy-of-reasoning
uv sync
```

Already cloned without ThinkARM? Run `git submodule update --init --recursive`.

## Evaluate a model

Run commands from the repository root. `configs/label.yaml` already points to the
evaluation data below; use it directly.

1. Set your API key in your shell or `.env`. To create a credentials file:

   ```bash
   cp -n secrets.example .env
   ```

   Fill in `OPENAI_API_KEY` or `LABELER_API_KEY` (which takes precedence).
   Existing shell values override `.env`; credentials files are gitignored.

2. Edit `model` and `base_url` in `configs/label.yaml`. Use your server's model
   name and API root, such as `http://localhost:4000/v1`. For OpenAI, use
   `https://api.openai.com/v1` and set `chat_template_kwargs: null`.

   The server must accept OpenAI-compatible `/chat/completions` requests. If it
   lacks strict JSON schema support, set `response_format: json_object`. For a
   server without authentication, leave both API keys empty or unset.

3. Export ThinkARM's human-annotated sentence prefixes:

   ```bash
   uv run python scripts/convert_thinkarm.py deepseekR1 --human-only --out data/thinkarm
   ```

   This creates `traces.jsonl` and `human.jsonl`. `deepseekR1` selects the source
   traces (nine annotated prefixes); `model` in the config selects the labeler.

4. Label and evaluate:

   ```bash
   uv run python -m labeling label configs/label.yaml --run thinkarm-1
   uv run python -m labeling evaluate --labels runs/thinkarm-1/labels.jsonl --human data/thinkarm/human.jsonl
   ```

Evaluation prints accuracy, Cohen's kappa, counts, skipped ids, and a confusion
matrix. Missing traces and sentence-text mismatches are reported and excluded.

Results, config, and repository commit are saved under `runs/thinkarm-1/`.
Repeat the label command to resume; completed traces are skipped and failures
produce a nonzero exit status. Only `concurrency` and `traces_in_flight` may
change when resuming. Use a new run name for other config changes.

## Reasoning on vLLM

The included config disables thinking for model templates supporting this switch:

```yaml
chat_template_kwargs:
  enable_thinking: false
```

Some templates use `thinking: false` instead. Set the switch to `true` to enable
thinking, or `chat_template_kwargs: null` to use server defaults. Use `null` for
OpenAI and servers without template-argument support. The output's per-sentence
`reason` is separate from thinking mode.

## Label full ThinkARM traces

Export all traces, or select the first N per source model:

```bash
# All full traces, including those without human labels
uv run python scripts/convert_thinkarm.py deepseekR1 --out data/thinkarm-all

# First 20 full traces
uv run python scripts/convert_thinkarm.py deepseekR1 --limit 20 --out data/thinkarm-20
```

Set `input` in `configs/label.yaml` to the exported `traces.jsonl`, then use a new
run name:

```bash
uv run python -m labeling label configs/label.yaml --run thinkarm-20
```

Both modes also write available matching annotations to `human.jsonl`; evaluation
scores only those annotated prefixes. `--limit` works with `--human-only` too.
Multiple source models can be supplied before the options; ids remain stable.

## Development

Run offline tests with `uv run pytest`.

To add a backend, implement `Labeler.label(problem, sentences)` and register its
factory in `labelers.REGISTRY`. Return one `Prediction` per sentence or raise.
