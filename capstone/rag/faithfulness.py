"""Generation-side RAG metrics.

faithfulness:      fraction of claims in the answer supported by the retrieved context
                   (RAGAS-style: extract claims, then verify each claim).
citation_validity: every cited chunk id was actually retrieved, and the cited chunks support the
                   answer.
answer_relevance:  does the answer address the question (judge).

`lexical_faithfulness` is an offline proxy for tests and CI. Report the LLM version.
"""

from __future__ import annotations

import re
from typing import Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from capstone.rag.bm25 import tokenize


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(tokenize(s)) >= 2]


def lexical_faithfulness(answer: str, context: str, threshold: float = 0.6) -> float:
    ctx = set(tokenize(context))
    claims = _sentences(re.sub(r"\[[^\]]+\]", "", answer))
    if not claims:
        return 1.0
    supported = sum(1 for c in claims
                    if (toks := tokenize(c)) and sum(t in ctx for t in toks) / len(toks)
                    >= threshold)
    return supported / len(claims)


def citations(answer: str) -> list[str]:
    return re.findall(r"\[([^\]\s]+:[^\]\s]+)\]", answer)


def citation_validity(answer: str, retrieved_ids: list[str]) -> float:
    cited = citations(answer)
    if not cited:
        return 0.0
    return sum(c in retrieved_ids for c in cited) / len(cited)


class _Claims(BaseModel):
    claims: list[str]


class _Verdict(BaseModel):
    claim: str
    verdict: Literal["supported", "contradicted", "not_found"]


class _Verdicts(BaseModel):
    verdicts: list[_Verdict]


class LLMFaithfulness:
    def __init__(self, model: BaseChatModel) -> None:
        self.extract = model.with_structured_output(_Claims)
        self.verify = model.with_structured_output(_Verdicts)

    def score(self, question: str, answer: str, context: str) -> tuple[float, list[_Verdict]]:
        claims = self.extract.invoke(
            "Split the answer into atomic factual claims (one fact each). Ignore citations and "
            f"pleasantries.\nQuestion: {question}\nAnswer: {answer}")
        assert isinstance(claims, _Claims)
        if not claims.claims:
            return 1.0, []
        verdicts = self.verify.invoke(
            "For each claim decide if the CONTEXT supports it, contradicts it, or doesn't "
            "mention it. Use only the context, not world knowledge.\n"
            f"CONTEXT:\n{context}\n\nCLAIMS:\n" + "\n".join(f"- {c}" for c in claims.claims))
        assert isinstance(verdicts, _Verdicts)
        v = verdicts.verdicts
        return (sum(x.verdict == "supported" for x in v) / len(v) if v else 0.0), v
