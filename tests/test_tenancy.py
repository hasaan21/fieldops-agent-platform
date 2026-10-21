import pytest
from langgraph.store.memory import InMemoryStore

from capstone.agents.memory import load_memories, save_memory
from capstone.core.tenancy import TenantIsolationError
from capstone.data.loader import load_table
from capstone.rag.ingest import synthetic_index


def test_crm_never_returns_other_tenants_customers(crm):
    t2_customer = next(c for c in load_table("customers") if c["tenant_id"] == "t2")
    assert crm.find_customer("t1", phone=t2_customer["phone"]) is None
    assert crm.find_customer("t2", phone=t2_customer["phone"])["customer_id"] == \
        t2_customer["customer_id"]


def test_cannot_create_job_for_other_tenants_customer(crm):
    t2_customer = next(c for c in load_table("customers") if c["tenant_id"] == "t2")
    slot = crm.find_available_slots("t1", "gas", "normal")[0]
    with pytest.raises(TenantIsolationError):
        crm.create_job("t1", customer_id=t2_customer["customer_id"], site_id=None,
                       category="gas", urgency="normal", description="x", slot=slot)


def test_unknown_tenant_rejected(crm):
    with pytest.raises(TenantIsolationError):
        crm.find_available_slots("t999", "gas", "normal")


def test_job_creation_is_idempotent(crm):
    slot = crm.find_available_slots("t1", "gas", "normal")[0]
    kwargs = dict(customer_id=None, site_id=None, category="gas", urgency="normal",
                  description="x", slot=slot, idempotency_key="msg-123")
    assert crm.create_job("t1", **kwargs)["job_id"] == crm.create_job("t1", **kwargs)["job_id"]


def test_memory_namespaces_are_isolated():
    store = InMemoryStore()
    save_memory(store, "t1", "c1", "k", "t1 secret")
    assert load_memories(store, "t2", "c1") == []
    assert load_memories(store, "t1", "c1") == ["t1 secret"]


def test_retrieval_only_returns_own_tenant_chunks():
    idx = synthetic_index("markdown")
    for tenant in ["t1", "t2", "t3"]:
        hits = idx.search(tenant, "call-out fee warranty emergency number", k=10)
        assert hits and all(h.chunk.tenant_id == tenant for h in hits)
