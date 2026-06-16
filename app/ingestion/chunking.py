"""
Document ingestion and chunking pipeline.

Loads raw documents, validates schema, and splits text into overlapping
chunks suitable for embedding and retrieval. Supports both fixed-size
and semantic (sentence-boundary aware) chunking strategies, configurable
via the pipeline config.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single retrievable unit of text with provenance metadata."""

    chunk_id: str
    doc_id: str
    title: str
    source: str
    year: int
    text: str
    chunk_index: int


REQUIRED_FIELDS = {"id", "title", "source", "year", "text"}


def load_raw_documents(path: str | Path) -> list[dict]:
    """Load and validate raw documents from a JSON file.

    Each document must contain id, title, source, year, and text fields.
    Documents failing validation are logged and skipped rather than
    raising, so a single malformed record does not halt ingestion.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)

    valid_docs = []
    for i, doc in enumerate(raw_docs):
        missing = REQUIRED_FIELDS - doc.keys()
        if missing:
            logger.warning("Skipping document at index %d: missing fields %s", i, missing)
            continue
        if not doc["text"].strip():
            logger.warning("Skipping document %s: empty text field", doc.get("id"))
            continue
        valid_docs.append(doc)

    logger.info("Loaded %d/%d valid documents from %s", len(valid_docs), len(raw_docs), path)
    return valid_docs


def _split_into_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (avoids heavy NLP dependencies)."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunk(
    text: str,
    target_chunk_size: int = 400,
    overlap_sentences: int = 1,
) -> list[str]:
    """Split text into chunks at sentence boundaries.

    Groups consecutive sentences until the target character size is
    reached, then starts a new chunk. A configurable number of trailing
    sentences from the previous chunk are repeated at the start of the
    next chunk to preserve cross-chunk context (overlap).
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        current.append(sentence)
        current_len += len(sentence)

        if current_len >= target_chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current_len = sum(len(s) for s in current)

    if current:
        chunks.append(" ".join(current))

    return chunks


def fixed_size_chunk(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Split text into fixed-size character windows with overlap.

    Provided as the baseline strategy for the chunking comparison
    described in the project README. Does not respect sentence
    boundaries, which can fragment context mid-sentence.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())
        if start + chunk_size >= len(text):
            break
    return chunks


def build_chunks(
    documents: list[dict],
    strategy: str = "semantic",
    **chunk_kwargs,
) -> list[Chunk]:
    """Chunk a collection of documents using the specified strategy.

    Args:
        documents: list of validated raw document dicts.
        strategy: one of "semantic" or "fixed".
        chunk_kwargs: forwarded to the underlying chunking function.

    Returns:
        Flat list of Chunk objects with provenance metadata preserved
        from the source document.
    """
    chunker = {
        "semantic": semantic_chunk,
        "fixed": fixed_size_chunk,
    }.get(strategy)

    if chunker is None:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")

    all_chunks: list[Chunk] = []
    for doc in documents:
        pieces = chunker(doc["text"], **chunk_kwargs)
        for idx, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{doc['id']}_chunk{idx}",
                    doc_id=doc["id"],
                    title=doc["title"],
                    source=doc["source"],
                    year=doc["year"],
                    text=piece,
                    chunk_index=idx,
                )
            )

    logger.info("Built %d chunks from %d documents using '%s' strategy",
                len(all_chunks), len(documents), strategy)
    return all_chunks


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    """Persist chunks to a JSONL file for downstream embedding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk)) + "\n")
    logger.info("Saved %d chunks to %s", len(chunks), path)


def run_ingestion_pipeline(
    raw_path: str | Path = "data/raw/pharma_abstracts.json",
    output_path: str | Path = "data/processed/chunks.jsonl",
    strategy: str = "semantic",
) -> list[Chunk]:
    """End-to-end ingestion: load, validate, chunk, persist."""
    docs = load_raw_documents(raw_path)
    chunks = build_chunks(docs, strategy=strategy)
    save_chunks(chunks, output_path)
    return chunks


if __name__ == "__main__":
    run_ingestion_pipeline()
