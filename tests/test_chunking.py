import json

import pytest

from app.ingestion.chunking import (
    build_chunks,
    fixed_size_chunk,
    load_raw_documents,
    semantic_chunk,
)


def test_load_raw_documents_filters_invalid(tmp_path):
    docs = [
        {"id": "1", "title": "A", "source": "S", "year": 2024, "text": "hello world"},
        {"id": "2", "title": "B", "source": "S", "year": 2024, "text": ""},
        {"id": "3", "title": "C", "year": 2024, "text": "missing source"},
    ]
    path = tmp_path / "docs.json"
    path.write_text(json.dumps(docs))

    valid = load_raw_documents(path)

    assert len(valid) == 1
    assert valid[0]["id"] == "1"


def test_semantic_chunk_respects_sentence_boundaries():
    text = "First sentence here. Second sentence here. Third sentence here. Fourth one too."
    chunks = semantic_chunk(text, target_chunk_size=30, overlap_sentences=1)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.strip().endswith(".")


def test_semantic_chunk_overlap():
    text = "A. B. C. D. E."
    chunks = semantic_chunk(text, target_chunk_size=1, overlap_sentences=1)

    assert len(chunks) >= 2


def test_fixed_size_chunk_respects_size_and_overlap():
    text = "x" * 100
    chunks = fixed_size_chunk(text, chunk_size=20, overlap=5)

    assert all(len(c) <= 20 for c in chunks)
    assert len(chunks) >= 1


def test_fixed_size_chunk_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        fixed_size_chunk("text", chunk_size=10, overlap=10)


def test_build_chunks_preserves_provenance():
    docs = [
        {
            "id": "doc1",
            "title": "Title 1",
            "source": "Source 1",
            "year": 2024,
            "text": "Sentence one. Sentence two. Sentence three.",
        }
    ]
    chunks = build_chunks(docs, strategy="semantic", target_chunk_size=10)

    assert all(c.doc_id == "doc1" for c in chunks)
    assert all(c.title == "Title 1" for c in chunks)
    assert chunks[0].chunk_id == "doc1_chunk0"


def test_build_chunks_unknown_strategy_raises():
    with pytest.raises(ValueError):
        build_chunks([{"id": "1", "title": "T", "source": "S", "year": 2024, "text": "hi"}], strategy="nonexistent")
