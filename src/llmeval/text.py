"""Small, dependency-free text helpers shared by the metrics and judges."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the and or but if then than so of to in on at by for with from into onto
    over under is are was were be been being am do does did done has have had having
    it its this that these those there here what which who whom whose when where why
    how i you he she we they me him her us them my your his our their can could would
    should will shall may might must not no nor yes as about after before again all any
    each few more most other some such only own same too very just also up down out off
    once per via""".split()
)

NEGATIONS = frozenset({"not", "no", "never", "none", "cannot", "without", "nor"})

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")
_CITATION = re.compile(r"\[([^\[\]\s]+\.(?:md|txt|pdf|html))\]")
_NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "twenty": "20",
    "thirty": "30",
    "sixty": "60",
    "ninety": "90",
    "hundred": "100",
}


def tokens(text: str) -> list[str]:
    """Lowercased word tokens; hyphenated codes like ``e-14`` stay whole.

    ``n't`` contractions become ``not`` so negation survives tokenisation."""
    text = re.sub(r"n't\b", " not", text.lower())
    return _TOKEN.findall(text)


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS]


def stem(token: str) -> str:
    """A deliberately crude suffix stripper: enough to match "retries" with
    "retry" and "applied" with "apply", not a real stemmer."""
    for suffix in ("ies", "ied"):
        if token.endswith(suffix) and len(token) > 4:
            return token[: -len(suffix)] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def content_stems(text: str) -> set[str]:
    return {stem(t) for t in content_tokens(text)}


def numbers(text: str) -> set[str]:
    """Numeric facts in a text: digits, number words, and codes like ``N-22``.

    These are what a wrong answer most often gets wrong, and overlap scores
    barely notice when one number is swapped for another."""
    found = set()
    for tok in tokens(text):
        if any(ch.isdigit() for ch in tok):
            found.add(tok)
        elif tok in _NUMBER_WORDS:
            found.add(_NUMBER_WORDS[tok])
    return found


def has_negation(text: str) -> bool:
    return any(t in NEGATIONS for t in tokens(text))


def sentences(text: str) -> list[str]:
    text = _CITATION.sub("", text)
    parts = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line or line.startswith("#"):
            continue
        parts.extend(p.strip() for p in _SENTENCE.split(line) if p.strip())
    return parts


def strip_citations(text: str) -> str:
    return _CITATION.sub("", text)


def citations(text: str) -> list[str]:
    return _CITATION.findall(text)


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 over content tokens."""
    pred, ref = content_tokens(prediction), content_tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    remaining = list(ref)
    common = 0
    for tok in pred:
        if tok in remaining:
            remaining.remove(tok)
            common += 1
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)
