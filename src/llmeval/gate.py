"""Regression gate: fail CI when a metric falls below its baseline.

``baseline.json`` maps metric names to a floor/ceiling:

    {"hit_at_1": {"value": 0.8, "tolerance": 0.05, "direction": "higher"},
     "answerable_refused": {"value": 0.0, "tolerance": 0.0, "direction": "lower"}}

A "higher" metric fails below ``value - tolerance``; a "lower" one fails above
``value + tolerance``. A metric the run could not compute (generation metrics
on an offline run) is reported as skipped, not passed: the gate only vouches
for what it measured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Check:
    metric: str
    actual: float | None
    bound: float
    direction: str
    status: str  # pass | fail | skipped


def load_baseline(path: str | Path) -> dict:
    baseline = json.loads(Path(path).read_text())
    for name, spec in baseline.items():
        if spec.get("direction") not in ("higher", "lower"):
            raise ValueError(f"baseline {name!r}: direction must be 'higher' or 'lower'")
    return baseline


def check(metrics: dict[str, dict], baseline: dict) -> list[Check]:
    checks = []
    for name, spec in baseline.items():
        tol = spec.get("tolerance", 0.0)
        higher = spec["direction"] == "higher"
        bound = spec["value"] - tol if higher else spec["value"] + tol
        actual = metrics.get(name, {}).get("value")
        if actual is None:
            status = "skipped"
        elif (actual >= bound - 1e-9) if higher else (actual <= bound + 1e-9):
            status = "pass"
        else:
            status = "fail"
        checks.append(Check(name, actual, round(bound, 4), spec["direction"], status))
    return checks


def passed(checks: list[Check]) -> bool:
    return not any(c.status == "fail" for c in checks)
