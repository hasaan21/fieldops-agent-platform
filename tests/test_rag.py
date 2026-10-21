import pytest

from capstone.agents.engineer_copilot.graph import ABSTAIN, CopilotContext, CopilotDeps, build_graph
from capstone.rag.base import Chunk
from capstone.rag.bm25 import BM25
from capstone.rag.chunking import chunk_document, fixed_size, markdown_sections, recursive
from capstone.rag.faithfulness import citation_validity, lexical_faithfulness
from capstone.rag.fusion import rrf
from capstone.rag.ingest import synthetic_index
from capstone.rag.memory_store import InMemoryHybridIndex
from capstone.rag.metrics import ndcg_at_k, reciprocal_rank

DOC = {"doc_id": "d1", "tenant_id": "t1", "kb_id": "kb", "title": "X200 manual",
       "content": "# X200\n\n## Safety\nIsolate power first.\n\n## Fault codes\n### Code F22\n"
                  "Meaning: Low water pressure. Re-pressurise to 1.2 bar.\n"}


def test_fixed_size_overlap():
    parts = fixed_size("abcdefghij" * 10, size=30, overlap=10)
    assert all(len(p) <= 30 for p in parts)
    assert parts[0][-10:] == parts[1][:10]


def test_recursive_respects_size():
    text = ("Sentence one is here. " * 40).strip()
    assert all(len(p) <= 120 for p in recursive(text, 120))


def test_markdown_sections_keep_heading_path():
    secs = markdown_sections(DOC["content"])
    assert ("X200 > Fault codes > Code F22", "Meaning: Low water pressure. Re-pressurise to "
            "1.2 bar.") in secs


def test_parent_chunks_point_to_full_section():
    chunks = chunk_document(DOC, "parent", size=90)
    assert all(c.parent_id for c in chunks)
    f22 = [c for c in chunks if c.section == "Code F22"]
    assert f22 and "1.2 bar" in f22[0].metadata["parent_content"]


def test_rrf_rewards_agreement():
    fused = dict(rrf([["a", "b", "c"], ["b", "a", "d"]]))
    assert fused["a"] == pytest.approx(fused["b"])
    assert fused["a"] > fused["c"] > 0


def test_bm25_prefers_rare_terms():
    bm = BM25(["boiler pressure low", "boiler service", "boiler boiler boiler"])
    scores = bm.scores("pressure")
    assert scores[0] > 0 and scores[1] == 0


def test_retrieval_metrics():
    assert reciprocal_rank([False, True]) == 0.5
    assert ndcg_at_k([True], 5) == 1.0
    assert ndcg_at_k([False, True], 5) == pytest.approx(1 / 1.585, rel=1e-3)


def test_hybrid_index_filters_and_isolates():
    idx = InMemoryHybridIndex()
    idx.add([Chunk("a", "t1", "kb", "d1", "boiler pressure F22"),
             Chunk("b", "t2", "kb", "d2", "boiler pressure F22")])
    assert [h.chunk.chunk_id for h in idx.search("t1", "F22 pressure")] == ["a"]
    assert idx.search("t1", "F22", filters={"doc_id": "nope"}) == []


def test_faithfulness_proxy_and_citations():
    ctx = "Code F22 means low water pressure. Re-pressurise to 1.2 bar."
    assert lexical_faithfulness("F22 means low water pressure.", ctx) == 1.0
    assert lexical_faithfulness("Replace the heat exchanger immediately.", ctx) == 0.0
    assert citation_validity("Low pressure [d1:sec:3].", ["d1:sec:3"]) == 1.0
    assert citation_validity("Low pressure [d9:sec:1].", ["d1:sec:3"]) == 0.0


def test_copilot_answers_with_citation_from_own_tenant():
    app = build_graph(CopilotDeps(retriever=synthetic_index("parent"))).compile()
    out = app.invoke({"question": "What does error code F28 mean on the Thermacore X200?"},
                     context=CopilotContext(tenant_id="t1"))
    assert "Ignition failure" in out["answer"]
    assert out["citation_validity"] == 1.0
    assert all(h.chunk.tenant_id == "t1" for h in out["hits"])


def test_copilot_abstains_without_knowledge():
    app = build_graph(CopilotDeps(retriever=InMemoryHybridIndex())).compile()
    out = app.invoke({"question": "What does code F28 mean?"},
                     context=CopilotContext(tenant_id="t1"))
    assert out["answer"] == ABSTAIN
