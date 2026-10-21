"""Long-term agent memory on the LangGraph Store.

Short-term memory is the thread: messages persisted by the checkpointer, keyed by thread_id.
Long-term memory is the Store: facts that outlive a thread, namespaced per tenant and customer
so one tenant's memories can't be searched from another tenant's runs.

    namespace = ("tenants", tenant_id, "customers", customer_id, "memories")
"""

from __future__ import annotations

from datetime import UTC, datetime

from langgraph.store.base import BaseStore


def namespace(tenant_id: str, customer_id: str) -> tuple[str, ...]:
    return ("tenants", tenant_id, "customers", customer_id, "memories")


def save_memory(store: BaseStore, tenant_id: str, customer_id: str, key: str, text: str,
                kind: str = "event") -> None:
    store.put(namespace(tenant_id, customer_id), key,
              {"text": text, "kind": kind, "at": datetime.now(UTC).isoformat()})


def load_memories(store: BaseStore, tenant_id: str, customer_id: str, query: str | None = None,
                  limit: int = 5) -> list[str]:
    items = store.search(namespace(tenant_id, customer_id), query=query, limit=limit)
    return [str(i.value["text"]) for i in items]
