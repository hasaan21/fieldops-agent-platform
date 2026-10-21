import json
from uuid import uuid4

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from capstone.app.main import app
from capstone.data.loader import load_table
from capstone.evals import feedback_loop
from capstone.observability import otel
from capstone.observability.ledger import (
    Ledger,
    LedgerCallback,
    LLMCall,
    RunRecord,
    cost_usd,
    price_for,
)

client = TestClient(app)
T1 = {"x-api-key": "dev-key-t1"}
T2 = {"x-api-key": "dev-key-t2"}


def test_pricing_uses_longest_model_prefix():
    assert price_for("openai:gpt-4.1-mini-2025-04-14") == price_for("gpt-4.1-mini")
    assert price_for("gpt-4.1") != price_for("gpt-4.1-mini")
    assert cost_usd("gpt-4.1-mini", 1_000_000, 0) == 0.40
    assert price_for("some-local-model") == (0.0, 0.0)


def test_callback_reads_standard_usage_metadata():
    cb = LedgerCallback()
    rid = uuid4()
    cb.on_chat_model_start({}, [[]], run_id=rid, metadata={"ls_model_name": "gpt-4.1-mini"})
    msg = AIMessage("ok", usage_metadata={"input_tokens": 1200, "output_tokens": 80,
                                          "total_tokens": 1280})
    cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]), run_id=rid)
    (call,) = cb.calls
    assert (call.model, call.input_tokens, call.output_tokens) == ("gpt-4.1-mini", 1200, 80)
    assert abs(call.cost_usd - (1200 * 0.40 + 80 * 1.60) / 1e6) < 1e-12


def test_budgets_are_per_channel():
    voice = RunRecord("r1", "t1", "job_intake", "voice", latency_ms=2100)
    sms = RunRecord("r2", "t1", "job_intake", "sms", latency_ms=2100)
    pricey = RunRecord("r3", "t1", "job_intake", "sms", latency_ms=100,
                       llm_calls=[LLMCall("gpt-4.1", 20_000, 2_000, 900, 0.056)])
    assert voice.budget_violations() and not sms.budget_violations()
    assert any("cost" in v for v in pricey.budget_violations())


def test_ledger_summary_is_tenant_scoped():
    ledger = Ledger()
    for i, (tenant, lat) in enumerate([("t1", 100), ("t1", 300), ("t2", 50)]):
        ledger.record(RunRecord(f"r{i}", tenant, "job_intake", "sms", lat, outcome="booked"))
    s1 = ledger.summary("t1")
    assert s1["runs"] == 2 and s1["latency_ms"]["p50"] == 300
    assert ledger.summary("t3") == {"tenant_id": "t3", "runs": 0}


def test_agent_runs_emit_spans_and_ledger_rows():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    original = otel.tracer
    otel.tracer = provider.get_tracer("test")
    try:
        cust = next(c for c in load_table("customers") if c["tenant_id"] == "t1")
        r = client.post("/v1/agents/job-intake/messages", headers=T1, json={
            "channel": "voice", "sender": cust["phone"],
            "text": "How much is your call-out fee?"}).json()
    finally:
        otel.tracer = original
    assert r["usage"]["llm_calls"] == 0 and r["usage"]["cost_usd"] == 0  # offline mode
    (span,) = [s for s in exporter.get_finished_spans() if s.name == "agent.job_intake"]
    assert span.attributes["fieldops.tenant_id"] == "t1"
    assert span.attributes["fieldops.channel"] == "voice"
    usage_t1 = client.get("/v1/usage", headers=T1).json()
    assert usage_t1["runs"] >= 1 and "voice" in usage_t1["latency_p95_by_channel"]
    assert "voice" not in client.get("/v1/usage", headers=T2).json().get(
        "latency_p95_by_channel", {})


def test_readiness_probe():
    body = client.get("/readyz").json()
    assert body["status"] == "ready" and body["graph"] is True


def test_feedback_loop_collect_dedupe_review_promote(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback_loop, "CANDIDATES", tmp_path / "candidates.jsonl")
    monkeypatch.setattr(feedback_loop, "REGRESSIONS", tmp_path / "regressions.jsonl")

    def eval_row(i, msg, score):
        return {"id": f"x{i}", "inputs": {"tenant_id": "t1", "channel": "sms", "sender": "",
                                          "message": msg},
                "reference": {"intent": "book_job", "expected_outcome": "booked"},
                "metadata": {"failure_mode": "none"}, "outputs": {"outcome": "clarify"},
                "scores": [{"key": "outcome_correct", "score": score}]}

    rows = [c for r in [eval_row(1, "Boiler down, card 4111 1111 1111 1111", 0.0),
                        eval_row(2, "Boiler down, card 4111 1111 1111 1111", 0.0),  # dup
                        eval_row(3, "All fine here", 1.0)]
            if (c := feedback_loop.candidate_from_eval_row(r))]
    assert len(rows) == 2  # passing run is not a candidate
    assert "4111" not in rows[0]["inputs"]["message"]  # re-redacted before entering git
    assert feedback_loop.collect(rows, against_splits=False) == {
        "found": 2, "added": 1, "duplicates": 1}

    class FakeRun:
        id = uuid4()
        extra = {"metadata": {"tenant_id": "t2", "channel": "email"}}
        inputs = {"messages": [{"content": "Lights keep tripping in the office"}]}
        outputs = {"outcome": "clarify", "classification": {"intent": "book_job"}}

    online = feedback_loop.candidate_from_langsmith_run(FakeRun(), ["user_score"])
    assert online["metadata"]["labelled"] is False
    feedback_loop.collect([online], against_splits=False)
    assert feedback_loop.promote() == {"promoted": 1, "regressions_total": 1,
                                       "still_unlabelled": 1}
    assert feedback_loop.review(auto_accept=True) == 1
    assert feedback_loop.promote()["regressions_total"] == 2
    lines = (tmp_path / "regressions.jsonl").read_text().splitlines()
    assert all(json.loads(line)["metadata"]["split"] == "regressions" for line in lines)
