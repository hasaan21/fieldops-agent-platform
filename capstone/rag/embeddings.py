from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


class HashingEmbedder:
    """Deterministic, offline, free. Character n-gram hashing: lexical, not semantic.

    Use it for tests and CI. Run the real ablation with OpenAIEmbedder (or a local
    sentence-transformers model) and report the difference. That gap is your evidence that
    dense embeddings earn their cost.
    """

    name = "hashing-char-ngrams"

    def __init__(self, dim: int = 1536) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self._vec = HashingVectorizer(n_features=dim, analyzer="char_wb", ngram_range=(3, 5),
                                      alternate_sign=False, norm=None)

    def embed(self, texts: list[str]) -> np.ndarray:
        m = self._vec.transform([t.lower() for t in texts]).toarray().astype(np.float32)
        return _normalize(np.log1p(m))


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536) -> None:
        from langchain_openai import OpenAIEmbeddings

        self.name, self.dim = model, dim
        self._client = OpenAIEmbeddings(model=model, dimensions=dim)

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(self._client.embed_documents(texts), dtype=np.float32))


class SentenceTransformerEmbedder:
    """Local open model, e.g. BAAI/bge-small-en-v1.5 (needs the `deep` extra)."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model
        self._m = SentenceTransformer(model)
        self.dim = int(self._m.get_sentence_embedding_dimension() or 384)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._m.encode(texts, normalize_embeddings=True), dtype=np.float32)


def default_embedder() -> Embedder:
    from capstone.core.config import get_settings

    s = get_settings()
    if s.openai_api_key and not s.fieldops_offline:
        return OpenAIEmbedder(s.fieldops_embedding_model)
    return HashingEmbedder()
