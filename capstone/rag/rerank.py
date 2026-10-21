from __future__ import annotations

from typing import Protocol

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from capstone.rag.base import Hit


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]: ...


class NoopReranker:
    name = "none"

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        return hits[:k]


class CrossEncoderReranker:
    """e.g. BAAI/bge-reranker-base or cross-encoder/ms-marco-MiniLM-L-6-v2 (`deep` extra)."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self.name = model
        self._m = CrossEncoder(model)

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self._m.predict([(query, h.chunk.content) for h in hits])
        ranked = sorted(zip(hits, scores, strict=True), key=lambda x: -float(x[1]))
        return [Hit(h.chunk, float(s), {**h.ranks, "rerank": i + 1})
                for i, (h, s) in enumerate(ranked[:k])]


class _Scores(BaseModel):
    scores: list[int] = Field(description="relevance 0-3 for each passage, in input order")


class LLMReranker:
    """Listwise LLM reranking: slower and costlier, but no extra model to host."""

    name = "llm"

    def __init__(self, model: BaseChatModel) -> None:
        self._llm = model.with_structured_output(_Scores)

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        passages = "\n\n".join(f"[{i}] {h.chunk.content[:600]}" for i, h in enumerate(hits))
        out = self._llm.invoke(
            "Score each passage 0-3 for how well it answers the question "
            "(3 = directly answers, 0 = irrelevant).\n"
            f"Question: {query}\n\nPassages:\n{passages}")
        assert isinstance(out, _Scores)
        scores = (out.scores + [0] * len(hits))[: len(hits)]
        ranked = sorted(zip(hits, scores, strict=True), key=lambda x: -x[1])
        return [Hit(h.chunk, float(s), {**h.ranks, "rerank": i + 1})
                for i, (h, s) in enumerate(ranked[:k])]
