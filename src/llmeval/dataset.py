"""Golden-set loading. One JSON object per line:

    {"id": "billing-01", "question": "...", "answerable": true,
     "expected_sources": ["billing_faq.md"], "reference": "..."}

Off-topic rows have ``"answerable": false`` and no sources; the target should
refuse or escalate on those.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    answerable: bool
    expected_sources: tuple[str, ...] = ()
    reference: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


def load_cases(path: str | Path) -> list[Case]:
    cases, seen = [], set()
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        case = Case(
            id=row["id"],
            question=row["question"],
            answerable=bool(row["answerable"]),
            expected_sources=tuple(row.get("expected_sources", ())),
            reference=row.get("reference", ""),
            tags=tuple(row.get("tags", ())),
        )
        if case.id in seen:
            raise ValueError(f"{path}:{lineno}: duplicate case id {case.id!r}")
        if case.answerable and not case.expected_sources:
            raise ValueError(f"{path}:{lineno}: answerable case {case.id!r} has no sources")
        seen.add(case.id)
        cases.append(case)
    return cases


@dataclass(frozen=True)
class CalibrationItem:
    id: str
    context: str
    answer: str
    faithful: bool
    error_type: str = ""  # for unfaithful items: number, entity, negation, fabricated...


def load_calibration(path: str | Path) -> list[CalibrationItem]:
    items = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            items.append(
                CalibrationItem(
                    id=row["id"],
                    context=row["context"],
                    answer=row["answer"],
                    faithful=bool(row["faithful"]),
                    error_type=row.get("error_type", ""),
                )
            )
    return items
