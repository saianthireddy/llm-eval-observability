"""End to end against a real rag-support-automation checkout."""

import json

from llmeval.cli import ROOT, main
from llmeval.dataset import load_cases
from llmeval.judges import HeuristicJudge
from llmeval.runner import run
from llmeval.targets import RagSupportTarget


def test_offline_run_scores_retrieval_and_skips_generation(target_repo):
    result, tracer = run(
        RagSupportTarget(target_repo), load_cases(ROOT / "data" / "golden.jsonl"), HeuristicJudge()
    )
    m = result.metrics
    assert m["hit_at_1"]["n"] == 22 and m["off_topic_refused"]["n"] == 8
    assert m["faithfulness"]["value"] is None  # no model, nothing to judge
    assert m["errors"]["value"] == 0
    generate = [s for s in tracer.spans if s.name == "generate"]
    assert generate and all(s.attributes["model"] == "offline-stub" for s in generate)


def test_generation_path_with_an_injected_model(target_repo):
    """An 'extractive' model that copies the top chunk's first sentence: it is
    faithful by construction, so the judge must score it 1.0 and the
    citation must check out."""

    def extractive(system, user):
        context = user.split("Context:\n", 1)[1]
        source = context.split("]", 1)[0].lstrip("[")
        first = context.split("\n", 2)[1]
        return f"{first} [{source}]"

    target = RagSupportTarget(target_repo, llm=extractive)
    case = load_cases(ROOT / "data" / "golden.jsonl")[0]
    result, tracer = run(target, [case], HeuristicJudge())
    c = result.cases[0]
    assert not c.offline and not c.refused
    assert c.faithfulness == 1.0 and c.citation_valid == 1.0
    assert any(s.name == "generate" and s.attributes["model"] == "custom" for s in tracer.spans)


def test_cli_run_with_gate_writes_artifacts(target_repo, tmp_path):
    out = tmp_path / "run"
    code = main(
        [
            "run",
            "--target-repo",
            str(target_repo),
            "--gate",
            str(ROOT / "baseline.json"),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    results = json.loads((out / "results.json").read_text())
    assert results["target"] == "rag-support-automation"
    assert (out / "traces.jsonl").read_text().count("\n") > 30
    assert "Regression gate" in (out / "report.md").read_text()


def test_readme_run_numbers_are_current(target_repo):
    result, _ = run(
        RagSupportTarget(target_repo), load_cases(ROOT / "data" / "golden.jsonl"), HeuristicJudge()
    )
    m = result.metrics
    readme = (ROOT / "README.md").read_text()
    row = (
        f"| {m['hit_at_1']['value']:.2f} | {m['recall_at_k']['value']:.2f} "
        f"| {m['mrr']['value']:.2f} | {m['off_topic_refused']['value']:.2f} "
        f"| {m['answerable_refused']['value']:.2f} |"
    )
    assert row in readme, f"README results row out of date; expected: {row}"
