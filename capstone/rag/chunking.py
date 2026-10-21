"""Chunking strategies. Week 3's ablation compares them on the same gold set.

- fixed:     N characters with overlap. Simple baseline; splits mid-sentence.
- recursive: split on paragraph, then line, then sentence, then word until under size.
- markdown:  one chunk per heading section, with the heading path kept as `section`.
- parent:    small child chunks for precise matching, each pointing to its markdown section
             (the "parent") so generation gets the whole section (small-to-big retrieval).
- semantic:  split where embedding similarity between adjacent sentences drops.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np

from capstone.rag.base import Chunk

Strategy = Literal["fixed", "recursive", "markdown", "parent", "semantic"]


def fixed_size(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    if size <= overlap:
        raise ValueError("size must exceed overlap")
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return [c for c in out if c.strip()]


def recursive(text: str, size: int = 400, separators: Sequence[str] = ("\n\n", "\n", ". ", " ")
              ) -> list[str]:
    if len(text) <= size:
        return [text] if text.strip() else []
    if not separators:
        return fixed_size(text, size, 0)
    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    out: list[str] = []
    buf = ""
    for part in parts:
        candidate = f"{buf}{sep}{part}" if buf else part
        if len(candidate) <= size:
            buf = candidate
            continue
        if buf:
            out.append(buf)
        if len(part) > size:
            out.extend(recursive(part, size, rest))
            buf = ""
        else:
            buf = part
    if buf.strip():
        out.append(buf)
    return out


def markdown_sections(text: str) -> list[tuple[str, str]]:
    """Returns (heading_path, body) per section. '# A' > '## B' > '### C' -> 'A > B > C'."""
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append((" > ".join(path), content))

    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            body = []
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
        else:
            body.append(line)
    flush()
    return sections


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def semantic(text: str, embed: Callable[[list[str]], np.ndarray], threshold: float = 0.35,
             max_chars: int = 800) -> list[str]:
    sents = _sentences(text)
    if len(sents) <= 1:
        return sents
    vecs = embed(sents)
    sims = (vecs[:-1] * vecs[1:]).sum(axis=1)
    out, cur = [], [sents[0]]
    for sent, sim in zip(sents[1:], sims, strict=True):
        if sim < threshold or sum(map(len, cur)) + len(sent) > max_chars:
            out.append(" ".join(cur))
            cur = []
        cur.append(sent)
    out.append(" ".join(cur))
    return out


def _leaf(path: str) -> str:
    return path.split(" > ")[-1] if path else ""


def chunk_document(doc: dict[str, Any], strategy: Strategy = "markdown", size: int = 400,
                   embed: Callable[[list[str]], np.ndarray] | None = None) -> list[Chunk]:
    base = {"tenant_id": doc["tenant_id"], "kb_id": doc["kb_id"], "doc_id": doc["doc_id"]}
    meta = {"title": doc.get("title"), "kind": doc.get("kind")}
    text = doc["content"]
    chunks: list[Chunk] = []
    if strategy in {"fixed", "recursive", "semantic"}:
        if strategy == "fixed":
            pieces = fixed_size(text, size)
        elif strategy == "recursive":
            pieces = recursive(text, size)
        else:
            if embed is None:
                raise ValueError("semantic chunking needs an embed function")
            pieces = semantic(text, embed)
        for i, p in enumerate(pieces):
            chunks.append(Chunk(chunk_id=f"{doc['doc_id']}:{strategy}:{i}", content=p,
                                section=None, metadata=meta, **base))
        return chunks
    for i, (path, body) in enumerate(markdown_sections(text)):
        section_id = f"{doc['doc_id']}:sec:{i}"
        header = f"{doc.get('title', '')} - {path}\n"
        if strategy == "markdown":
            chunks.append(Chunk(chunk_id=section_id, content=header + body, section=_leaf(path),
                                metadata=meta, **base))
        else:  # parent: children carry parent's full text in metadata for small-to-big expansion
            for j, piece in enumerate(recursive(body, max(120, size // 3))):
                chunks.append(Chunk(chunk_id=f"{section_id}:{j}", content=header + piece,
                                    section=_leaf(path), parent_id=section_id,
                                    metadata={**meta, "parent_content": header + body}, **base))
    return chunks
