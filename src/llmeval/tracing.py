"""Minimal tracing: nested spans with timings and attributes, exported as JSONL.

The field names follow OpenTelemetry's span model (trace_id, span_id,
parent_span_id, name, start/end in ns, attributes, status), so the file can be
mapped onto an OTLP exporter later without changing call sites. Nothing here
needs the OpenTelemetry SDK: an eval run should not need a collector running.

    tracer = Tracer()
    with tracer.span("case", question=q) as case:
        with tracer.span("retrieve") as s:
            s.set(hits=3)
"""

from __future__ import annotations

import json
import secrets
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    start_ns: int
    end_ns: int = 0
    attributes: dict = field(default_factory=dict)
    status: str = "OK"

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1e6

    def set(self, **attributes) -> None:
        self.attributes.update(attributes)


class Tracer:
    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._stack: list[Span] = []

    @contextmanager
    def span(self, name: str, **attributes) -> Iterator[Span]:
        parent = self._stack[-1] if self._stack else None
        span = Span(
            name=name,
            trace_id=parent.trace_id if parent else secrets.token_hex(16),
            span_id=secrets.token_hex(8),
            parent_span_id=parent.span_id if parent else None,
            start_ns=time.perf_counter_ns(),
            attributes=dict(attributes),
        )
        self._stack.append(span)
        try:
            yield span
        except BaseException as exc:
            span.status = "ERROR"
            span.set(error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            span.end_ns = time.perf_counter_ns()
            self._stack.pop()
            self.spans.append(span)

    def export_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for span in self.spans:
                fh.write(json.dumps({**asdict(span), "duration_ms": span.duration_ms}) + "\n")

    def latency_summary(self) -> dict[str, dict[str, float]]:
        """p50 / p95 / max milliseconds per span name."""
        by_name: dict[str, list[float]] = defaultdict(list)
        for span in self.spans:
            by_name[span.name].append(span.duration_ms)
        return {name: _percentiles(values) for name, values in sorted(by_name.items())}


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def pct(p: float) -> float:
        # nearest-rank: honest for small samples, never interpolates past the data
        rank = max(1, -(-len(ordered) * p // 100))
        return ordered[int(rank) - 1]

    return {
        "count": len(ordered),
        "p50_ms": round(pct(50), 3),
        "p95_ms": round(pct(95), 3),
        "max_ms": round(ordered[-1], 3),
    }
