"""Faithfulness judges: is every claim in an answer supported by its context?

Two implementations behind one interface:

``HeuristicJudge``
    Offline and deterministic. A sentence counts as supported when one context
    sentence covers most of its content words AND every number/code in it,
    with matching negation. Cheap enough to run on every commit, blind to
    paraphrase and to errors that keep the same words. How blind is measured,
    not assumed: ``llmeval calibrate`` scores it against hand-labeled cases.

``LLMJudge``
    Asks a model to list unsupported claims and return JSON. Takes any
    ``complete(prompt) -> str`` callable, so tests use a fake and production
    uses ``openai_complete``. Unparseable output is reported as an error,
    never silently scored as faithful.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .text import content_stems, has_negation, numbers, sentences


@dataclass
class Verdict:
    score: float  # fraction of answer claims supported by the context, 0..1
    unsupported: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def faithful(self) -> bool:
        return not self.error and not self.unsupported


class Judge(Protocol):
    name: str

    def faithfulness(self, answer: str, contexts: Sequence[str]) -> Verdict: ...


class HeuristicJudge:
    name = "heuristic"

    def __init__(self, coverage: float = 0.6):
        self.coverage = coverage

    def _supported(self, claim: str, context_sentences: list[str]) -> bool:
        claim_stems = content_stems(claim)
        if not claim_stems:
            return True  # nothing checkable ("See below.")
        claim_numbers = numbers(claim)
        claim_negated = has_negation(claim)
        for ctx in context_sentences:
            ctx_stems = content_stems(ctx)
            if len(claim_stems & ctx_stems) / len(claim_stems) < self.coverage:
                continue
            if not claim_numbers <= numbers(ctx):
                continue
            if claim_negated != has_negation(ctx):
                continue
            return True
        return False

    def faithfulness(self, answer: str, contexts: Sequence[str]) -> Verdict:
        claims = sentences(answer)
        if not claims:
            return Verdict(score=1.0)
        context_sentences = [s for c in contexts for s in sentences(c)]
        unsupported = [c for c in claims if not self._supported(c, context_sentences)]
        return Verdict(score=1 - len(unsupported) / len(claims), unsupported=unsupported)


JUDGE_PROMPT = """\
You are grading whether an answer is faithful to the context it was given.
Split the answer into its factual claims. A claim is SUPPORTED only if the
context states it or directly implies it; numbers, codes, names and negations
must match exactly. Background knowledge does not count as support.

Context:
{context}

Answer:
{answer}

Reply with JSON only, no prose:
{{"claims": <total number of claims>, "unsupported": [<each unsupported claim, quoted>]}}"""


class LLMJudge:
    name = "llm"

    def __init__(self, complete: Callable[[str], str]):
        self._complete = complete

    def faithfulness(self, answer: str, contexts: Sequence[str]) -> Verdict:
        prompt = JUDGE_PROMPT.format(context="\n\n".join(contexts), answer=answer)
        raw = self._complete(prompt)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            data = json.loads(match.group(0) if match else raw)
            total = int(data["claims"])
            unsupported = [str(c) for c in data.get("unsupported", [])]
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            return Verdict(score=0.0, error=f"unparseable judge output: {exc}: {raw[:200]!r}")
        if total <= 0:
            return Verdict(score=1.0 if not unsupported else 0.0, unsupported=unsupported)
        supported = max(total - len(unsupported), 0)
        return Verdict(score=supported / total, unsupported=unsupported)


def openai_complete(model: str = "gpt-4o-mini") -> Callable[[str], str]:
    from openai import OpenAI  # optional dependency

    client = OpenAI()

    def complete(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    return complete


def default_judge() -> Judge:
    """The LLM judge when ``OPENAI_API_KEY`` is set, the heuristic otherwise."""
    if os.getenv("OPENAI_API_KEY"):
        return LLMJudge(openai_complete(os.getenv("JUDGE_MODEL", "gpt-4o-mini")))
    return HeuristicJudge()
