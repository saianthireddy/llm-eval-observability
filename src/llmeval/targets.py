"""Systems under test. A target answers a question and reports what it used:

    target.ask(question, tracer) -> Response(answer, contexts, refused, offline)

``contexts`` are the retrieved chunks in rank order, so retrieval can be scored
separately from generation. ``offline`` marks answers that came from no model
at all; generation metrics are skipped for those rather than scored as perfect.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .tracing import Tracer


@dataclass
class Context:
    source: str
    text: str
    score: float


@dataclass
class Response:
    answer: str
    contexts: list[Context] = field(default_factory=list)
    refused: bool = False
    offline: bool = False


_REFUSAL = re.compile(
    r"(does not|doesn't|do not|don't) (contain|include|cover|mention)|"
    r"(could not|couldn't|cannot|can't|unable to) (find|answer)|"
    r"no (relevant )?(information|documentation)",
    re.IGNORECASE,
)


def looks_like_refusal(answer: str) -> bool:
    """A model told to answer only from context declines in prose ("the context
    does not cover this, please escalate") rather than with the fixed message."""
    return bool(_REFUSAL.search(answer))


class Target(Protocol):
    name: str

    def ask(self, question: str, tracer: Tracer) -> Response: ...


class RagSupportTarget:
    """``saianthireddy/rag-support-automation``, run in-process.

    Built from the target's own components (embedder, splitter, store,
    retriever, chain, settings) rather than re-implemented, so the eval measures
    that code. Retrieval and generation are wrapped in spans.

    Without ``OPENAI_API_KEY`` the chain's model is replaced by a stub that
    returns no answer and marks the response offline; retrieval, refusal and
    latency are still real.
    """

    name = "rag-support-automation"

    def __init__(self, repo: str | Path, llm: Callable[[str, str], str] | None = None):
        """``llm(system, user) -> str`` overrides the model (tests, other
        providers). Default: the target's own OpenAI call when
        ``OPENAI_API_KEY`` is set, the offline stub otherwise."""
        repo = Path(repo).resolve()
        if not (repo / "src" / "rag_support").is_dir():
            raise FileNotFoundError(f"{repo} is not a rag-support-automation checkout")
        if str(repo / "src") not in sys.path:
            sys.path.insert(0, str(repo / "src"))

        from rag_support.config import get_settings
        from rag_support.embeddings.embedder import HashingEmbedder
        from rag_support.generation.chain import ESCALATION, RagChain
        from rag_support.ingestion.loader import load_documents
        from rag_support.ingestion.splitter import split_document
        from rag_support.retrieval.retriever import Retriever
        from rag_support.vectorstore.memory_store import InMemoryStore

        settings = get_settings()
        self.settings = settings
        self.escalation = ESCALATION
        embedder = HashingEmbedder()
        store = InMemoryStore()
        chunks = []
        for doc in load_documents(str(repo / "data" / "sample_docs")):
            chunks.extend(split_document(doc, settings.chunk_size, settings.chunk_overlap))
        store.add(
            embedder.embed([c.text for c in chunks]),
            [{"text": c.text, "source": c.source} for c in chunks],
        )
        self.retriever = Retriever(
            embedder, store, top_k=settings.top_k, min_score=settings.min_score
        )

        self.use_llm = llm is not None or bool(os.getenv("OPENAI_API_KEY"))
        self._model_name = "custom" if llm else settings.chat_model
        self._last_results: list = []
        self._tracer: Tracer | None = None

        outer = self

        class _TracedRetriever:
            def retrieve(self, query: str):
                with outer._tracer.span(
                    "retrieve", top_k=settings.top_k, min_score=settings.min_score
                ) as span:
                    results = outer.retriever.retrieve(query)
                    span.set(
                        hits=len(results), top_score=round(results[0].score, 4) if results else None
                    )
                outer._last_results = results
                return results

        self._chain = RagChain(_TracedRetriever(), llm=self._llm, chat_model=settings.chat_model)
        self._call_model = llm or self._chain._openai_llm

    def _llm(self, system: str, user: str) -> str:
        model = self._model_name if self.use_llm else "offline-stub"
        with self._tracer.span("generate", model=model) as span:
            if not self.use_llm:
                return ""
            answer = self._call_model(system, user)
            span.set(answer_chars=len(answer))
            return answer

    def ask(self, question: str, tracer: Tracer) -> Response:
        self._tracer, self._last_results = tracer, []
        result = self._chain.ask(question)
        contexts = [Context(r.source, r.text, r.score) for r in self._last_results]
        refused = (
            result.answer == self.escalation
            or not contexts
            or (self.use_llm and looks_like_refusal(result.answer))
        )
        return Response(
            answer=result.answer,
            contexts=contexts,
            refused=refused,
            offline=not self.use_llm and not refused,
        )
