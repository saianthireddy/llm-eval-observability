"""Markdown report for a run: metrics with counts, latency, gate, failures."""

from __future__ import annotations

from .gate import Check
from .runner import RunResult

_LABELS = {
    "hit_at_1": "Retrieval hit@1",
    "recall_at_k": "Retrieval recall@k",
    "mrr": "Retrieval MRR",
    "off_topic_refused": "Off-topic refused",
    "answerable_refused": "Answerable refused",
    "faithfulness": "Faithfulness",
    "correctness_f1": "Correctness (token F1)",
    "citation_valid": "Valid citations",
    "judge_errors": "Judge errors",
    "errors": "Target errors",
}


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def render(run: RunResult, checks: list[Check] | None = None) -> str:
    lines = [f"# Eval report: {run.target}", "", f"Judge: `{run.judge}`", ""]
    offline = sum(c.offline for c in run.cases)
    if offline:
        lines += [
            f"> {offline} answered case(s) ran with no model (offline stub), so "
            "generation metrics are n/a for them. Set `OPENAI_API_KEY` to score generation.",
            "",
        ]
    lines += ["| Metric | Value | Cases |", "|---|---|---|"]
    for key, label in _LABELS.items():
        m = run.metrics.get(key, {})
        lines.append(f"| {label} | {_fmt(m.get('value'))} | {m.get('n', 0)} |")

    lines += ["", "## Latency (ms)", "", "| Span | n | p50 | p95 | max |", "|---|---|---|---|---|"]
    for name, s in run.latency.items():
        lines.append(
            f"| {name} | {s['count']} | {s['p50_ms']:.2f} | {s['p95_ms']:.2f} | {s['max_ms']:.2f} |"
        )

    if checks is not None:
        lines += [
            "",
            "## Regression gate",
            "",
            "| Metric | Actual | Bound | Status |",
            "|---|---|---|---|",
        ]
        for c in checks:
            op = ">=" if c.direction == "higher" else "<="
            lines.append(f"| {c.metric} | {_fmt(c.actual)} | {op} {c.bound} | {c.status} |")

    failures = [
        c
        for c in run.cases
        if c.error or c.judge_error or not c.refusal_correct or c.hit_at_1 == 0.0 or c.unsupported
    ]
    if failures:
        lines += ["", "## Cases to look at", ""]
        for c in failures:
            why = []
            if c.error:
                why.append(f"error: {c.error}")
            if not c.refusal_correct:
                why.append(
                    "refused an answerable question"
                    if c.answerable
                    else "answered an off-topic question"
                )
            if c.hit_at_1 == 0.0:
                why.append(f"top source {c.retrieved[:1] or 'none'}, expected {c.expected}")
            if c.unsupported:
                why.append(f"{len(c.unsupported)} unsupported claim(s)")
            if c.judge_error:
                why.append("judge error")
            lines.append(f"- `{c.id}` {c.question!r}: " + "; ".join(why))
    return "\n".join(lines) + "\n"


def render_calibration(report: dict) -> str:
    lines = [
        f"# Judge calibration: {report['judge']}",
        "",
        f"{report['items']} labeled items, {report['judge_errors']} judge errors.",
        "",
        "| Accuracy | Precision (flags) | Recall (unfaithful caught) |",
        "|---|---|---|",
        f"| {_fmt(report['accuracy'])} | {_fmt(report['precision'])} | {_fmt(report['recall'])} |",
        "",
        "| Error type | Caught |",
        "|---|---|",
    ]
    for kind, v in report["recall_by_error_type"].items():
        lines.append(f"| {kind} | {v['caught']}/{v['of']} |")
    if report["misses"]:
        lines += [
            "",
            "Misses: " + ", ".join(f"`{m['id']}` ({m['kind']})" for m in report["misses"]),
        ]
    return "\n".join(lines) + "\n"
