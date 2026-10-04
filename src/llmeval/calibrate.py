"""Evaluate the evaluator: score a judge against hand-labeled answers.

A faithfulness number is only as good as the judge behind it. Each calibration
item is a context, an answer and a human label (faithful or not, and if not,
what kind of error). The judge's job is to flag the unfaithful ones.

Reported as detection of *unfaithful* answers, because that is what the judge
is for:
  precision  of the answers it flagged, how many were really unfaithful
  recall     of the unfaithful answers, how many it caught
  per error type recall, which shows *what kind* of mistake slips through
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from .dataset import CalibrationItem
from .judges import Judge


def calibrate(judge: Judge, items: Sequence[CalibrationItem]) -> dict:
    tp = fp = fn = tn = errors = 0
    by_type: dict[str, list[bool]] = defaultdict(list)
    misses = []
    for item in items:
        verdict = judge.faithfulness(item.answer, [item.context])
        if verdict.error:
            errors += 1
            continue
        flagged = not verdict.faithful
        if not item.faithful:
            by_type[item.error_type or "unspecified"].append(flagged)
        if flagged and not item.faithful:
            tp += 1
        elif flagged:
            fp += 1
            misses.append({"id": item.id, "kind": "false alarm"})
        elif not item.faithful:
            fn += 1
            misses.append({"id": item.id, "kind": f"missed {item.error_type}"})
        else:
            tn += 1
    scored = tp + fp + fn + tn
    return {
        "judge": judge.name,
        "items": len(items),
        "judge_errors": errors,
        "accuracy": round((tp + tn) / scored, 4) if scored else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "recall_by_error_type": {
            k: {"caught": sum(v), "of": len(v)} for k, v in sorted(by_type.items())
        },
        "misses": misses,
    }
