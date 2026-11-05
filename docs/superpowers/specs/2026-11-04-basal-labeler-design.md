# Basal labeler: design

## Intent

Test whether a typed-decision classifier, `Remek/basal-2.0-4.5B`, can label reasoning
sentences faster than the LLM labelers, with agreement with the human labels as close to the LLMs'
as possible. The model returns a probability per option in one forward pass and generates no
text. It runs on the user's own NVIDIA GPU.

## Goals

- Add a `basal` labeler selected by the `labeler: basal` config key. It plugs into the existing
  run, resume, evaluate flow without changes to those.
- **Speed is the main goal.** Compare wall-clock time (the existing per-trace and per-sentence
  timing output) and accuracy/kappa against an LLM run on the same traces.
- Default install is unchanged: no new entries in `dependencies`, no torch or CUDA. basal is an
  optional extra in the same environment: `uv sync --extra basal`.
- The README tells the user how to set up and run the server, and says it can be skipped.

## Non-goals

- No fine-tuning or training. This is a zero-shot test.
- No in-process model loading (see decisions).
- ThinkARM is not edited.
- **The agent does not benchmark or run evaluations.** No full labeling runs, timing
  comparisons, or accuracy/kappa measurements. The user runs those. The agent may install basal on
  the provided GPU and send a handful of requests only to confirm the labeler works (see Tests).

## Anti-goals

- Making the default `uv sync` heavier or requiring a GPU for tests (`uv run pytest` stays
  offline and uses a fake server).
- Silent label drift: an unknown or missing answer must fail the trace, not default to a label.

## Decisions

**Server, same environment.** The user runs `basal-serve` on their GPU machine; the labeler only
sends HTTP requests. basal is the `basal` extra in `pyproject.toml`: `basal[fp8]` from the GitHub
tarball and `torch==2.11.0` from the cu128 index, both via `[tool.uv.sources]` (the torch index
applies only to this extra). basal requires Python >=3.10, so the repo's >=3.13 works; its
`transformers==5.17.0` pin needed the repo's unused `transformers` lower bound relaxed to 5.17.

```bash
uv sync --extra basal
uv run basal-serve --model Remek/basal-2.0-4.5B --mode fast --template-prefix off --port 8000
```

The README notes the CUDA 12.8-compatible driver requirement and `configs/basal.yaml` as the
matching run config.

**Labeler.** `BasalLabeler` in `labelers.py` (or a small sibling module), registered in
`REGISTRY['basal']`. It uses `httpx`, already installed through `openai`, so no new dependency.
Config: `base_url` (default `http://localhost:8000`), `concurrency`, `guidebook`.

**One request per sentence.** `POST /v1/systemone` with a `choice` question:
- `state`: problem, all earlier sentences as context, then the target sentence, clearly marked.
- `instructions`: which category the marked sentence belongs to.
- `criteria`: the 8 `LABELS`, each with a one-line description taken from the guidebook.

The label is the returned `choice`. The full `probabilities` go into `Prediction.probs`
(the field already exists). `reason` stays `None`. Requests run concurrently under the existing
`concurrency` semaphore, with the same retry/backoff on connection errors and 429/5xx.

**Failure handling.** A response whose `choice` is not one of `LABELS` raises `ValueError`, as
in `parse_predictions`, and counts as a failed trace.

## Left to the builder

- Exact wording of the question and the 8 criteria descriptions. Condense them from
  `ThinkARM/guidebook/sentence_guide.md`; keep them in a prompt file under `prompts/`.
- Response envelope: the model card shows per-question fields but not the full wrapper, so
  inspect a real response (or the engine's Python client) before writing the parser.
- Whether the server accepts several states per request. If it does, batch sentences to go
  faster; if not, per-sentence requests are the design.
- Whether the response parser lives in `labelers.py` or a sibling module.

## Tests

Offline unit tests need no GPU and no model: fake the HTTP layer the way `labelers.AsyncOpenAI`
is monkeypatched today (module-level name, so it stays importable). Keep them short:

- Request shape: the body has `state` (with the target sentence marked and earlier sentences as
  context) and one `choice` question whose `criteria` keys are exactly `LABELS`.
- A valid response becomes a `Prediction` with the right label and `probs` stored.
- A `choice` that is not in `LABELS` raises `ValueError`.
- A 503 or connection error is retried, then succeeds; a 400 is not retried.
- Output order matches input order when requests finish out of order.
- `labeler: basal` resolves through `REGISTRY`.

The fake responses must copy the real envelope, so the agent first captures one real response
(see below) and builds the fixtures from it.

One live check, run by the agent and not part of `uv run pytest`: install basal on the provided
GPU following the README commands, start the server, label two or three short sentences through
the new labeler, and confirm labels and probabilities come back and the README commands work. It
confirms correctness only; no timing, no accuracy numbers, no full runs. The agent then stops the
server and reports what it saw.

## Context length and truncation

Every sentence sees all earlier ones, so the state grows with the trace. The model has a 32k-token
context. The builder first measures the longest state in `data/thinkarm/traces.jsonl` (a quick
length count, not a model run). If everything fits with headroom, no truncation is added.

If some do not fit, use a **recent-window rule** (my proposal):

- New config key `max_context_chars`, default 60000 (about 20k tokens at a conservative 3 chars
  per token, leaving room for the problem, criteria and instructions). Characters rather than
  tokens so no tokenizer is needed; the builder may lower the default after seeing real token
  counts from the live check.
- The state always keeps the problem and the target sentence. Earlier sentences are added from
  the most recent backwards until the budget is used up; older ones are dropped.
- When anything is dropped, the context starts with the line `[N earlier sentences omitted]` so the
  model knows the context is partial.
- Sentence indices and output order are unaffected, since only the context text changes.
- A target sentence or problem that alone exceeds the budget fails the trace with a clear error
  rather than being cut silently.

Trade-off: dropping the oldest sentences loses the beginning of the reasoning, where `Read` and
early `Analyze` sentences usually sit. This is acceptable for the experiment, and affects only the
longest traces. Possible later variants, not part of this work: also keep the first few
sentences, or summarize the omitted part.

Tests: a unit test with a small `max_context_chars` checks that the newest sentences are kept,
the marker shows the right count, and an oversized target fails.
