"""Split documents into overlapping chunks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    char_start: int
    char_end: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _words(text: str) -> list[str]:
    return text.split()


def _sentences(text: str) -> list[str]:
    try:
        import nltk

        try:
            from nltk.tokenize import sent_tokenize

            return sent_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass
            from nltk.tokenize import sent_tokenize

            return sent_tokenize(text)
    except Exception:
        out = []
        buf = []
        for part in text.replace("?", ".").replace("!", ".").split("."):
            p = part.strip()
            if p:
                out.append(p + ".")
        return out if out else [text]


def _chunk_by_sentence(docs: list[Any], chunk_size: int, overlap: int) -> list[Chunk]:
    """Sliding window over words with sentence boundaries approximated by word windows."""
    out: list[Chunk] = []
    step = max(1, chunk_size - overlap)
    for doc in docs:
        full = doc.text
        sents = _sentences(full)
        words_all: list[str] = []
        for s in sents:
            words_all.extend(_words(s))
        if not words_all:
            continue
        i0 = 0
        chunk_idx = 0
        while i0 < len(words_all):
            i1 = min(i0 + chunk_size, len(words_all))
            piece = " ".join(words_all[i0:i1])
            if len(piece) < 20:
                i0 += step
                continue
            cid = f"{doc.doc_id}-{chunk_idx:04d}"
            out.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc.doc_id,
                    source=doc.source,
                    text=piece,
                    char_start=i0,
                    char_end=i1,
                    metadata=dict(getattr(doc, "metadata", {}) or {}),
                )
            )
            chunk_idx += 1
            i0 += step
    return out


def _chunk_by_word(docs: list[Any], chunk_size: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    step = max(1, chunk_size - overlap)
    for doc in docs:
        words = _words(doc.text)
        idx = 0
        chunk_idx = 0
        while idx < len(words):
            piece_words = words[idx : idx + chunk_size]
            piece = " ".join(piece_words)
            if len(piece) < 20:
                idx += step
                continue
            cid = f"{doc.doc_id}-{chunk_idx:04d}"
            out.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc.doc_id,
                    source=doc.source,
                    text=piece,
                    char_start=idx,
                    char_end=idx + len(piece_words),
                    metadata=dict(getattr(doc, "metadata", {}) or {}),
                )
            )
            chunk_idx += 1
            idx += step
    return out


def _chunk_by_character(docs: list[Any], chunk_size: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    step = max(1, chunk_size - overlap)
    for doc in docs:
        full = doc.text
        idx = 0
        chunk_idx = 0
        while idx < len(full):
            piece = full[idx : idx + chunk_size]
            if len(piece) < 20:
                break
            cid = f"{doc.doc_id}-{chunk_idx:04d}"
            out.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc.doc_id,
                    source=doc.source,
                    text=piece,
                    char_start=idx,
                    char_end=idx + len(piece),
                    metadata=dict(getattr(doc, "metadata", {}) or {}),
                )
            )
            chunk_idx += 1
            idx += step
    return out


def chunk_documents(
    docs: list[Any],
    chunk_size: int = 300,
    overlap: int = 50,
    split_by: str = "sentence",
) -> list[Chunk]:
    """Split documents into Chunk objects."""
    if split_by == "sentence":
        chunks = _chunk_by_sentence(docs, chunk_size, overlap)
    elif split_by == "word":
        chunks = _chunk_by_word(docs, chunk_size, overlap)
    elif split_by == "character":
        chunks = _chunk_by_character(docs, chunk_size, overlap)
    else:
        raise ValueError(f"Unknown split_by: {split_by}")

    lengths = [len(c.text) for c in chunks]
    avg_len = sum(lengths) / max(len(lengths), 1)
    logger.info(
        "Chunked into %d chunks (avg length %.0f chars, split_by=%s)",
        len(chunks),
        avg_len,
        split_by,
    )
    return chunks
