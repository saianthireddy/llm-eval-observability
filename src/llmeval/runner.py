"""Run a golden set through a target and score it.

Per case, three layers are scored independently, so a failure points at a
stage rather than at "the RAG system":

  retrieval   hit@1, recall@k, reciprocal rank           (answerable cases)
  refusal     refused when it should / answered when it should
  generation  faithfulness (judge), correctness vs reference (token F1),
              citations that point at retrieved sources  (answered, non-offline)

Aggregates are means over the cases each metric applies to, with the count
kept beside every number so "1.00" over two cases is visible as such.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from .dataset import Case
from .judges import Judge
from .targets import Target
from .text import citations, strip_citations, token_f1
from .tracing import Tracer


@dataclass
class CaseResult:
    id: str
    question: str
    answerable: bool
    refused: bool
    retrieved: list[str]
    expected: list[str]
    answer: str = ""
    offline: bool = False
    hit_at_1: float | None = None
    recall_at_k: float | None = None
    reciprocal_rank: float | None = None
    refusal_correct: bool = False
    faithfulness: float | None = None
    unsupported: list[str] = field(default_factory=list)
    judge_error: str = ""
    correctness_f1: float | None = None
    citation_valid: float | None = None
    error: str = ""


@dataclass
class RunResult:
    target: str
    judge: str
    cases: list[CaseResult]
    metrics: dict[str, dict]
    latency: dict[str, dict]

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "judge": self.judge,
            "metrics": self.metrics,
            "latency": self.latency,
            "cases": [asdict(c) for c in self.cases],
        }


def _retrieval(case: Case, retrieved: list[str]) -> tuple[float, float, float]:
    expected = set(case.expected_sources)
    hit1 = float(bool(retrieved) and retrieved[0] in expected)
    recall = float(any(s in expected for s in retrieved))
    rr = next((1 / i for i, s in enumerate(retrieved, 1) if s in expected), 0.0)
    return hit1, recall, rr


def score_case(case: Case, response, judge: Judge) -> CaseResult:
    retrieved = [c.source for c in response.contexts]
    result = CaseResult(
        id=case.id,
        question=case.question,
        answerable=case.answerable,
        refused=response.refused,
        retrieved=retrieved,
        expected=list(case.expected_sources),
        answer=response.answer,
        offline=response.offline,
        refusal_correct=response.refused != case.answerable,
    )
    if case.answerable:
        result.hit_at_1, result.recall_at_k, result.reciprocal_rank = _retrieval(case, retrieved)

    if response.refused or response.offline:
        return result

    verdict = judge.faithfulness(response.answer, [c.text for c in response.contexts])
    if verdict.error:
        result.judge_error = verdict.error
    else:
        result.faithfulness = verdict.score
        result.unsupported = verdict.unsupported
    if case.reference:
        result.correctness_f1 = token_f1(strip_citations(response.answer), case.reference)
    cited = citations(response.answer)
    result.citation_valid = sum(c in set(retrieved) for c in cited) / len(cited) if cited else 0.0
    return result


def _mean(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"value": None, "n": 0}
    return {"value": round(sum(values) / len(values), 4), "n": len(values)}


def aggregate(results: Sequence[CaseResult]) -> dict[str, dict]:
    answerable = [r for r in results if r.answerable]
    off_topic = [r for r in results if not r.answerable]
    generated = [r for r in results if not r.refused and not r.offline]
    return {
        "hit_at_1": _mean([r.hit_at_1 for r in answerable]),
        "recall_at_k": _mean([r.recall_at_k for r in answerable]),
        "mrr": _mean([r.reciprocal_rank for r in answerable]),
        "off_topic_refused": _mean([float(r.refused) for r in off_topic]),
        "answerable_refused": _mean([float(r.refused) for r in answerable]),
        "faithfulness": _mean([r.faithfulness for r in generated]),
        "correctness_f1": _mean([r.correctness_f1 for r in generated]),
        "citation_valid": _mean([r.citation_valid for r in generated]),
        "judge_errors": {"value": sum(bool(r.judge_error) for r in results), "n": len(generated)},
        "errors": {"value": sum(bool(r.error) for r in results), "n": len(results)},
    }


def run(
    target: Target, cases: Sequence[Case], judge: Judge, tracer: Tracer | None = None
) -> tuple[RunResult, Tracer]:
    tracer = tracer or Tracer()
    results = []
    for case in cases:
        with tracer.span("case", case_id=case.id, answerable=case.answerable) as span:
            try:
                response = target.ask(case.question, tracer)
            except Exception as exc:  # one broken case must not hide the rest
                span.set(error=f"{type(exc).__name__}: {exc}")
                span.status = "ERROR"
                results.append(
                    CaseResult(
                        case.id,
                        case.question,
                        case.answerable,
                        refused=False,
                        retrieved=[],
                        expected=list(case.expected_sources),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            with tracer.span("judge", judge=judge.name):
                result = score_case(case, response, judge)
            span.set(refused=result.refused, refusal_correct=result.refusal_correct)
            results.append(result)
    return (
        RunResult(
            target=target.name,
            judge=judge.name,
            cases=results,
            metrics=aggregate(results),
            latency=tracer.latency_summary(),
        ),
        tracer,
    )
