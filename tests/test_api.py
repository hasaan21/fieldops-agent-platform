from fastapi.testclient import TestClient

from capstone.app.main import app
from capstone.data.loader import load_table

client = TestClient(app)
T1 = {"x-api-key": "dev-key-t1"}
T2 = {"x-api-key": "dev-key-t2"}


def test_requires_api_key():
    assert client.post("/v1/agents/job-intake/messages",
                       json={"sender": "x", "text": "hi"}).status_code == 422
    assert client.post("/v1/agents/job-intake/messages", headers={"x-api-key": "nope"},
                       json={"sender": "x", "text": "hi"}).status_code == 401


def test_review_flow_and_cross_tenant_resume_is_forbidden():
    cust = next(c for c in load_table("customers") if c["tenant_id"] == "t1")
    r = client.post("/v1/agents/job-intake/messages", headers=T1, json={
        "channel": "sms", "sender": cust["phone"],
        "text": "The Coolstream AC-12 is rattling, can you send someone?"}).json()
    assert r["status"] == "pending_review", r
    thread = r["thread_id"]
    assert any(i["thread_id"] == thread for i in client.get("/v1/reviews", headers=T1).json())
    assert client.post(f"/v1/reviews/{thread}", headers=T2,
                       json={"approved": True}).status_code == 403
    done = client.post(f"/v1/reviews/{thread}", headers=T1, json={"approved": True}).json()
    assert done["outcome"] == "booked"
