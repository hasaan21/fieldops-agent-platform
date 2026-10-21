from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

SearchMode = Literal["vector", "bm25", "hybrid"]


@dataclass
class Chunk:
    chunk_id: str
    tenant_id: str
    kb_id: str
    doc_id: str
    content: str
    section: str | None = None
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # per-retriever rank, for debugging


class Retriever(Protocol):
    def search(self, tenant_id: str, query: str, k: int = 5, *, mode: SearchMode = "hybrid",
               filters: dict[str, Any] | None = None) -> list[Hit]: ...
