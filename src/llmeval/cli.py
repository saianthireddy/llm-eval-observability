"""Command line.

    python -m llmeval run --target-repo ../rag-support-automation [--gate baseline.json]
    python -m llmeval calibrate [--judge heuristic|llm]

``run`` writes results.json, traces.jsonl and report.md to ``--out`` and exits
1 if the gate fails. ``calibrate`` scores a judge on data/judge_calibration.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import gate as gate_mod
from .calibrate import calibrate
from .dataset import load_calibration, load_cases
from .judges import HeuristicJudge, LLMJudge, default_judge, openai_complete
from .report import render, render_calibration
from .runner import run
from .targets import RagSupportTarget

ROOT = Path(__file__).resolve().parents[2]


def _judge(name: str):
    if name == "heuristic":
        return HeuristicJudge()
    if name == "llm":
        return LLMJudge(openai_complete())
    return default_judge()


def cmd_run(args) -> int:
    target = RagSupportTarget(args.target_repo)
    cases = load_cases(args.golden)
    result, tracer = run(target, cases, _judge(args.judge))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    tracer.export_jsonl(out / "traces.jsonl")

    checks = None
    if args.gate:
        checks = gate_mod.check(result.metrics, gate_mod.load_baseline(args.gate))
    report = render(result, checks)
    (out / "report.md").write_text(report)
    print(report)
    if checks is not None and not gate_mod.passed(checks):
        print("Regression gate FAILED", file=sys.stderr)
        return 1
    return 0


def cmd_calibrate(args) -> int:
    report = calibrate(_judge(args.judge), load_calibration(args.data))
    print(render_calibration(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llmeval")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="evaluate a target on the golden set")
    p.add_argument("--target-repo", required=True, help="path to a rag-support-automation checkout")
    p.add_argument("--golden", default=str(ROOT / "data" / "golden.jsonl"))
    p.add_argument("--judge", choices=["auto", "heuristic", "llm"], default="auto")
    p.add_argument("--gate", help="baseline.json; exit 1 on regression")
    p.add_argument("--out", default="runs/latest")
    p.set_defaults(func=cmd_run)

    c = sub.add_parser("calibrate", help="score a judge on labeled answers")
    c.add_argument("--data", default=str(ROOT / "data" / "judge_calibration.jsonl"))
    c.add_argument("--judge", choices=["auto", "heuristic", "llm"], default="heuristic")
    c.add_argument("--json", help="also write the report as JSON")
    c.set_defaults(func=cmd_calibrate)

    args = parser.parse_args(argv)
    return args.func(args)
