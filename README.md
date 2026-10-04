# LLM Eval & Observability

[![CI](https://github.com/saianthireddy/llm-eval-observability/actions/workflows/ci.yml/badge.svg)](https://github.com/saianthireddy/llm-eval-observability/actions/workflows/ci.yml)

Evaluation, tracing and a CI regression gate for retrieval-augmented LLM
systems, run against a real one:
[rag-support-automation](https://github.com/saianthireddy/rag-support-automation).

It answers four questions about a RAG service, separately, so a bad number
points at a stage instead of at "the model":

| Layer | Metrics | Needs a model? |
|---|---|---|
| Retrieval | hit@1, recall@k, MRR against labeled source documents | no |
| Refusal | off-topic questions refused, answerable questions wrongly refused | no |
| Generation | faithfulness (judge), correctness vs reference (token F1), citations that point at retrieved sources | yes |
| Operations | per-stage latency p50 / p95 / max from traces | no |

And a fifth one that most eval setups skip: **how trustworthy is the judge
itself?** (see [Calibrating the judge](#calibrating-the-judge)).

The core is standard-library Python with no dependencies. CI runs the full
offline eval on every push against a pinned commit of the target, uploads the
report and traces, and fails the build if a metric falls below `baseline.json`.

## Results against rag-support-automation

Offline run (the target's hashing embedder, `TOP_K=4`, `MIN_SCORE=0.05`), on
this repo's golden set: 22 answerable questions over the target's five sample
documents and 8 off-topic ones. The questions are newly written for this
repo, not the target's own eval queries.

| hit@1 | recall@4 | MRR | Off-topic refused | Answerable refused |
|---|---|---|---|---|
| 0.91 | 0.95 | 0.92 | 0.62 | 0.00 |

A test re-runs the eval and fails if this row no longer matches.

**What it found.**

- **The relevance floor generalises worse than its own benchmark says.** On
  the target's own eval set, the floor refuses 0.83 of off-topic questions.
  On these held-out questions it refuses 0.62 (5 of 8). The floor was tuned on
  the set that reports it, so its 0.83 was optimistic. The three that got
  through ("Who is the current prime minister of Japan?", a translation
  request, a stock tip) match on generic words such as "current" and "good".
  That points at the bag-of-words embedder, not at the threshold: raising the
  floor would start refusing real questions, which is still at 0.00 and is
  the number that must not move.
- **Two retrieval misses, both distractor cases.** "Customer lost their data"
  ranks the security policy (which discusses breaches and lost data) above the
  support SOP that holds the procedure. "Refund for something charged two
  weeks ago" ranks the security policy first. Neither question shares much
  vocabulary with its answer.
- **Generation metrics are n/a offline, on purpose.** Without a model, the
  target returns no answer. Scoring an empty or context-echo answer as
  "faithful" would put a perfect, meaningless number in the report, so the
  harness marks those cases offline and the gate reports `faithfulness` as
  *skipped*, not passed. With `OPENAI_API_KEY` set, the same command scores
  real answers.

## Calibrating the judge

Faithfulness is only as good as whatever grades it. `data/judge_calibration.jsonl`
holds 30 hand-labeled (context, answer) pairs drawn from the target's
documents:

- 14 faithful (7 close to the source wording, 7 paraphrased)
- 16 unfaithful, with four kinds of error: a changed **number**, a swapped
  **entity**, a flipped **negation**, and an added **fabricated** claim

```
python -m llmeval calibrate --judge heuristic   # or --judge llm
```

| Judge | Accuracy | Precision | Recall (unfaithful caught) |
|---|---|---|---|
| heuristic | 0.83 | 0.87 | 0.81 |

| Error type | number | negation | fabricated | entity |
|---|---|---|---|---|
| Caught | 4/4 | 4/4 | 4/4 | **1/4** |

The offline heuristic checks three things per sentence: content-word
coverage, exact numbers and codes, and matching negation. It catches the
errors that change a token it checks, and misses swaps that keep the shape of
the sentence: "approved by the finance team" passes against "approved by the
support lead". It also raised two false alarms on loose paraphrases. So
heuristic faithfulness is a cheap regression signal for every commit, and the
LLM judge (`--judge llm`) is what to trust for absolute numbers. Running the
same calibration on it is one command.

Caveats: 30 items is small, and the same person wrote the heuristic and the
labels, so these numbers are more likely flattering than harsh.

## Tracing

Every case is a trace: `case` → `retrieve` → `generate` → `judge`, with
attributes such as `top_k`, `min_score`, `hits`, `top_score`, `model` and
`refused`. Spans are written to `runs/<name>/traces.jsonl` using
OpenTelemetry's field names (`trace_id`, `span_id`, `parent_span_id`, ns
timestamps, `status`), so they map onto an OTLP exporter without changing call
sites. The report summarises latency per span:

```
| Span     | n  | p50  | p95  | max  |
| retrieve | 30 | 2.84 | 3.68 | 4.48 |   (ms, offline, CI-class machine)
```

A failure inside the target is recorded on its span with `status: ERROR`, and
the run carries on. One broken case does not hide the other 29.

## Regression gate

`baseline.json` gives each metric a direction and a tolerance:

```json
"hit_at_1":           {"value": 0.91, "tolerance": 0.05, "direction": "higher"},
"answerable_refused": {"value": 0.0,  "tolerance": 0.0,  "direction": "lower"}
```

`llmeval run --gate baseline.json` exits 1 if any metric crosses its bound,
which fails the CI job. Metrics the run could not compute are reported as
`skipped`. Wrongly refused answerable questions have zero tolerance, because
each one is a real user dropped. Latency is reported but not gated, since
shared runners are too noisy for it.

## Run it

```bash
git clone https://github.com/saianthireddy/llm-eval-observability
git clone https://github.com/saianthireddy/rag-support-automation
cd llm-eval-observability
pip install -e ".[dev]"

python -m llmeval run --target-repo ../rag-support-automation --gate baseline.json
python -m llmeval calibrate
pytest -q
```

Outputs go to `runs/latest/`: `report.md`, `results.json` (per-case detail)
and `traces.jsonl`.

To score generation, `pip install -e ".[llm]"` and set `OPENAI_API_KEY`. The
target then calls its own model, and `--judge auto` switches to the LLM judge
(`JUDGE_MODEL`, default `gpt-4o-mini`).

## Layout

```
src/llmeval/
  targets.py    the system under test: rag-support-automation in-process,
                built from its own components; model injectable for tests
  runner.py     per-case scoring (retrieval / refusal / generation) + aggregates
  judges.py     HeuristicJudge (offline) and LLMJudge (any complete(prompt) callable)
  calibrate.py  judge vs human labels: precision, recall, per error type
  tracing.py    nested spans, JSONL export, latency percentiles
  gate.py       baseline comparison with direction and tolerance
  report.py     markdown report
data/
  golden.jsonl             30 cases: question, expected sources, reference answer
  judge_calibration.jsonl  30 labeled answers for judge calibration
baseline.json              gate thresholds
```

## Limitations

- Small sets: 22 answerable questions, 8 off-topic, 30 calibration items.
  The numbers are a regression signal, not a benchmark. Every aggregate in
  the report carries its case count for that reason.
- Retrieval is scored at the document level, so a hit means the right file,
  not necessarily the right passage.
- Token-F1 correctness rewards wording overlap with the reference. It is a
  trend line; for absolute correctness, use the LLM judge.
- CI only exercises the offline path. The LLM judge and generation scoring
  run in tests through injected fake models, not real API calls.

## License

MIT
