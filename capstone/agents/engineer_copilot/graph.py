"""Engineer Copilot: grounded Q&A over equipment manuals and FAQs for field engineers.

    retrieve -> rerank -> expand (small-to-big) -> generate -> check_faithfulness
                                                      ^               |
                                                      +-- retry once --+--> abstain if still bad

Audience is engineers, so engineer-only procedures are allowed (unlike customer-facing agents).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from capstone.guardrails.injection import spotlight
from capstone.rag.base import Hit, Retriever
from capstone.rag.faithfulness import LLMFaithfulness, citation_validity, lexical_faithfulness
from capstone.rag.rerank import NoopReranker, Reranker

MODEL_RE = re.compile(r"\b(Thermacore X\d+|Heatwell Combi \d+|Coolstream AC-\d+|Verdant HP-\d+|"
                      r"Aquaflow WH-\d+|Sentinel FP-\d+|Voltix DB-\d+)\b", re.I)
ABSTAIN = "I couldn't find this in the manuals for your tenant. Please check with the office."


@dataclass(frozen=True)
class CopilotContext:
    tenant_id: str


class CopilotState(TypedDict, total=False):
    question: str
    hits: list[Hit]
    context: str
    answer: str
    faithfulness: float
    citation_validity: float
    attempts: int
    abstained: bool


@dataclass
class CopilotDeps:
    retriever: Retriever
    reranker: Reranker = field(default_factory=NoopReranker)
    model: BaseChatModel | None = None  # None -> extractive answers (offline)
    judge: LLMFaithfulness | None = None
    k: int = 4
    min_faithfulness: float = 0.7


GEN_PROMPT = """You are a senior field engineer assistant. Answer the engineer's question using
ONLY the manual excerpts below. Cite every factual sentence with the chunk id in square brackets,
e.g. [doc:sec:3]. If the excerpts do not contain the answer, reply exactly: {abstain}
{strict}
EXCERPTS:
{context}

QUESTION: {question}"""


def build_graph(deps: CopilotDeps) -> StateGraph[CopilotState, CopilotContext, CopilotState,
                                                  CopilotState]:
    def retrieve(state: CopilotState, runtime: Runtime[CopilotContext]) -> dict[str, Any]:
        q = state["question"]
        hits = deps.retriever.search(runtime.context.tenant_id, q, k=20)
        m = MODEL_RE.search(q)
        if m:  # metadata boost: prefer the manual for the model the engineer named
            name = m.group(1).lower()
            hits = sorted(hits, key=lambda h: name not in (h.chunk.metadata.get("title") or "")
                          .lower())
        return {"hits": deps.reranker.rerank(q, hits, deps.k), "attempts": 0}

    def expand(state: CopilotState) -> dict[str, Any]:
        seen, blocks = set(), []
        for h in state["hits"]:
            key = h.chunk.parent_id or h.chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            text = h.chunk.metadata.get("parent_content") or h.chunk.content
            blocks.append(f"[{h.chunk.chunk_id}]\n{spotlight(text, 'manual')}")
        return {"context": "\n\n".join(blocks)}

    def generate(state: CopilotState) -> dict[str, Any]:
        attempts = state.get("attempts", 0) + 1
        if not state["hits"]:
            return {"answer": ABSTAIN, "attempts": attempts, "abstained": True}
        if deps.model is None:
            top = state["hits"][0].chunk
            body = (top.metadata.get("parent_content") or top.content).split("\n", 1)[-1]
            return {"answer": f"{body.strip()} [{top.chunk_id}]", "attempts": attempts}
        strict = ("\nYour previous answer contained unsupported claims. Quote the excerpts "
                  "closely and drop anything not stated in them.\n") if attempts > 1 else ""
        msg = deps.model.invoke(GEN_PROMPT.format(abstain=ABSTAIN, strict=strict,
                                                  context=state["context"],
                                                  question=state["question"]))
        answer = str(msg.content).strip()
        return {"answer": answer, "attempts": attempts, "abstained": answer == ABSTAIN}

    def check(state: CopilotState) -> dict[str, Any]:
        if state.get("abstained"):
            return {"faithfulness": 1.0, "citation_validity": 1.0}
        if deps.judge is not None:
            f, _ = deps.judge.score(state["question"], state["answer"], state["context"])
        else:
            f = lexical_faithfulness(state["answer"], state["context"])
        ids = [h.chunk.chunk_id for h in state["hits"]]
        return {"faithfulness": f, "citation_validity": citation_validity(state["answer"], ids)}

    def after_check(state: CopilotState) -> str:
        if state.get("abstained") or state["faithfulness"] >= deps.min_faithfulness:
            return END
        return "generate" if state["attempts"] < 2 else "abstain"

    def abstain(state: CopilotState) -> dict[str, Any]:
        return {"answer": ABSTAIN, "abstained": True}

    g = StateGraph(CopilotState, context_schema=CopilotContext)
    g.add_node("retrieve", retrieve)
    g.add_node("expand", expand)
    g.add_node("generate", generate)
    g.add_node("check", check)
    g.add_node("abstain", abstain)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "expand")
    g.add_edge("expand", "generate")
    g.add_edge("generate", "check")
    g.add_conditional_edges("check", after_check, [END, "generate", "abstain"])
    g.add_edge("abstain", END)
    return g
