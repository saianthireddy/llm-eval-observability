import json

import pytest

from llmeval.calibrate import calibrate
from llmeval.cli import ROOT
from llmeval.dataset import load_calibration
from llmeval.judges import HeuristicJudge, LLMJudge

CONTEXT = (
    "A declined card triggers three automatic retries over ten days. "
    "Service is not suspended until the grace period expires."
)


@pytest.mark.parametrize(
    "answer,faithful",
    [
        ("A declined card triggers three automatic retries over ten days.", True),
        ("Three automatic retries happen over ten days after a card is declined.", True),
        ("A declined card triggers five automatic retries over ten days.", False),  # number
        ("Service is suspended until the grace period expires.", False),  # negation dropped
        ("Retries happen over ten days. Customers also get an apology email.", False),  # invented
    ],
)
def test_heuristic_judge(answer, faithful):
    verdict = HeuristicJudge().faithfulness(answer, [CONTEXT])
    assert verdict.faithful is faithful, verdict


def test_heuristic_scores_the_fraction_of_supported_claims():
    answer = "A declined card triggers three automatic retries. The CEO approves every retry."
    verdict = HeuristicJudge().faithfulness(answer, [CONTEXT])
    assert verdict.score == 0.5
    assert verdict.unsupported == ["The CEO approves every retry."]


def test_citations_do_not_count_as_claims():
    verdict = HeuristicJudge().faithfulness(
        "A declined card triggers three automatic retries. [billing_faq.md]", [CONTEXT]
    )
    assert verdict.faithful


def test_llm_judge_parses_json_and_scores():
    seen = {}

    def fake(prompt):
        seen["prompt"] = prompt
        return 'Sure: {"claims": 4, "unsupported": ["refunds take a year"]}'

    verdict = LLMJudge(fake).faithfulness("answer text", ["ctx one", "ctx two"])
    assert verdict.score == 0.75 and verdict.unsupported == ["refunds take a year"]
    assert "ctx one" in seen["prompt"] and "answer text" in seen["prompt"]


def test_llm_judge_reports_garbage_instead_of_passing_it():
    verdict = LLMJudge(lambda _: "I think it's fine!").faithfulness("a", ["b"])
    assert verdict.error and not verdict.faithful


def test_calibration_counts_and_error_types():
    items = load_calibration(ROOT / "data" / "judge_calibration.jsonl")
    report = calibrate(HeuristicJudge(), items)
    c = report["confusion"]
    assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == len(items)
    # A perfect judge, simulated from the labels, must score perfectly.
    labels = {i.answer: i.faithful for i in items}

    def oracle(prompt):
        answer = prompt.split("Answer:\n", 1)[1].split("\n\nReply with JSON", 1)[0]
        return json.dumps({"claims": 1, "unsupported": [] if labels[answer] else [answer]})

    perfect = calibrate(LLMJudge(oracle), items)
    assert perfect["accuracy"] == perfect["precision"] == perfect["recall"] == 1.0


def test_readme_calibration_numbers_are_current():
    """The README quotes the heuristic's calibration; fail if it drifts."""
    items = load_calibration(ROOT / "data" / "judge_calibration.jsonl")
    r = calibrate(HeuristicJudge(), items)
    readme = (ROOT / "README.md").read_text()
    row = f"| heuristic | {r['accuracy']:.2f} | {r['precision']:.2f} | {r['recall']:.2f} |"
    assert row in readme, f"README calibration row out of date; expected: {row}"
