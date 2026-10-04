import json

import pytest

from llmeval.dataset import Case, load_cases
from llmeval.gate import check, passed
from llmeval.judges import HeuristicJudge
from llmeval.report import render
from llmeval.runner import run
from llmeval.targets import Context, Response, looks_like_refusal
from llmeval.text import citations, numbers, sentences, token_f1
from llmeval.tracing import Tracer

DOC = "Invoices are issued on the first business day of each month."


class FakeTarget:
    name = "fake"

    def __init__(self, answers):
        self.answers = answers

    def ask(self, question, tracer):
        with tracer.span("retrieve"):
            spec = self.answers[question]
        if spec is None:
            return Response(answer="escalate", refused=True)
        if isinstance(spec, Exception):
            raise spec
        source, answer = spec
        return Response(answer=answer, contexts=[Context(source, DOC, 0.5)])


CASES = [
    Case("a", "When are invoices issued?", True, ("billing.md",), DOC),
    Case("b", "Wrong doc?", True, ("billing.md",)),
    Case("c", "Capital of France?", False),
    Case("d", "Refused but answerable?", True, ("billing.md",)),
]


def _run(answers, cases=CASES):
    return run(FakeTarget(answers), cases, HeuristicJudge())


def test_scores_each_layer_separately():
    result, tracer = _run(
        {
            "When are invoices issued?": ("billing.md", DOC + " [billing.md]"),
            "Wrong doc?": (
                "manual.md",
                "Invoices are emailed by the finance director. [nowhere.md]",
            ),
            "Capital of France?": None,
            "Refused but answerable?": None,
        }
    )
    m = result.metrics
    # retrieval over the 3 answerable cases; the refused one retrieved nothing
    assert m["hit_at_1"] == {"value": round(1 / 3, 4), "n": 3}
    assert m["off_topic_refused"] == {"value": 1.0, "n": 1}
    assert m["answerable_refused"] == {"value": round(1 / 3, 4), "n": 3}
    # generation over the 2 answered cases
    assert m["faithfulness"] == {"value": 0.5, "n": 2}
    assert m["citation_valid"] == {"value": 0.5, "n": 2}
    by_id = {c.id: c for c in result.cases}
    assert by_id["a"].correctness_f1 == 1.0 and by_id["b"].correctness_f1 is None
    assert by_id["b"].unsupported == ["Invoices are emailed by the finance director."]
    names = {s.name for s in tracer.spans}
    assert {"case", "retrieve", "judge"} <= names


def test_offline_answers_are_not_scored_as_faithful():
    class Offline(FakeTarget):
        def ask(self, question, tracer):
            return Response(answer="", contexts=[Context("billing.md", DOC, 0.5)], offline=True)

    result, _ = run(Offline({}), CASES[:1], HeuristicJudge())
    assert result.metrics["hit_at_1"]["value"] == 1.0
    assert result.metrics["faithfulness"] == {"value": None, "n": 0}
    assert "n/a" in render(result)


def test_a_crashing_case_is_recorded_not_fatal():
    result, tracer = _run({"When are invoices issued?": RuntimeError("boom")}, CASES[:1])
    assert result.metrics["errors"]["value"] == 1
    assert any(s.status == "ERROR" for s in tracer.spans)


def test_tracer_nests_spans_and_exports(tmp_path):
    tracer = Tracer()
    with tracer.span("outer", k=1) as outer, tracer.span("inner") as inner:
        inner.set(hits=2)
    assert inner.parent_span_id == outer.span_id and inner.trace_id == outer.trace_id
    tracer.export_jsonl(tmp_path / "t.jsonl")
    rows = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert [r["name"] for r in rows] == ["inner", "outer"]
    assert rows[0]["attributes"] == {"hits": 2}
    assert tracer.latency_summary()["outer"]["count"] == 1


def test_gate():
    metrics = {
        "hit_at_1": {"value": 0.80},
        "answerable_refused": {"value": 0.05},
        "faithfulness": {"value": None},
    }
    baseline = {
        "hit_at_1": {"value": 0.85, "tolerance": 0.05, "direction": "higher"},
        "answerable_refused": {"value": 0.0, "tolerance": 0.0, "direction": "lower"},
        "faithfulness": {"value": 0.9, "tolerance": 0.05, "direction": "higher"},
    }
    status = {c.metric: c.status for c in check(metrics, baseline)}
    assert status == {"hit_at_1": "pass", "answerable_refused": "fail", "faithfulness": "skipped"}
    assert not passed(check(metrics, baseline))


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hold for ten seconds.", {"10"}),
        ("Error code N-22 and E-14", {"n-22", "e-14"}),
        ("Five business days", {"5"}),
    ],
)
def test_numbers(text, expected):
    assert numbers(text) == expected


def test_text_helpers():
    assert sentences("First claim. Second claim! [a.md]\n- third") == [
        "First claim.",
        "Second claim!",
        "third",
    ]
    assert citations("x [a.md] y [b.md]") == ["a.md", "b.md"]
    assert token_f1("issued monthly", "issued monthly") == 1.0
    assert token_f1("nothing shared", "entirely different") == 0.0


def test_refusal_wording():
    assert looks_like_refusal("The context does not contain this; please escalate.")
    assert not looks_like_refusal("Hold the power button for ten seconds.")


def test_golden_set_is_valid():
    from llmeval.cli import ROOT

    cases = load_cases(ROOT / "data" / "golden.jsonl")
    assert sum(c.answerable for c in cases) >= 20 and sum(not c.answerable for c in cases) >= 8
    assert all(c.reference for c in cases if c.answerable)


def test_duplicate_ids_rejected(tmp_path):
    row = json.dumps({"id": "x", "question": "q", "answerable": False})
    (tmp_path / "g.jsonl").write_text(row + "\n" + row + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(tmp_path / "g.jsonl")
