import numpy as np

from app.ingestion.chunking import Chunk
from app.retrieval.embeddings import TfidfEmbedding, get_default_embedding_model
from app.retrieval.vectorstore import BM25Index, HybridRetriever, VectorStore


SAMPLE_CHUNKS = [
    Chunk(chunk_id="c1", doc_id="d1", title="Diabetes", source="S", year=2023,
          text="GLP-1 receptor agonists reduce HbA1c levels in type 2 diabetes patients.",
          chunk_index=0),
    Chunk(chunk_id="c2", doc_id="d2", title="Pruning", source="S", year=2026,
          text="Structured pruning of ResNet-56 achieves a 5.1x MAC reduction on CIFAR-10.",
          chunk_index=0),
    Chunk(chunk_id="c3", doc_id="d3", title="MLOps", source="S", year=2024,
          text="Drift detection using PSI and KS-test triggers automated retraining pipelines.",
          chunk_index=0),
]


def _build_retriever():
    embedding_model = get_default_embedding_model(corpus_for_fit=[c.text for c in SAMPLE_CHUNKS])
    vs = VectorStore(embedding_model)
    vs.build(SAMPLE_CHUNKS)
    bm25 = BM25Index()
    bm25.build(SAMPLE_CHUNKS)
    return HybridRetriever(vs, bm25)


def test_tfidf_embedding_produces_normalized_vectors():
    model = TfidfEmbedding(dim=128)
    model.fit([c.text for c in SAMPLE_CHUNKS])

    vectors = model.embed([c.text for c in SAMPLE_CHUNKS])

    assert vectors.shape == (3, 128)
    norms = np.linalg.norm(vectors, axis=1)
    for norm in norms:
        assert norm == 0.0 or abs(norm - 1.0) < 1e-5


def test_vector_store_search_returns_relevant_chunk():
    embedding_model = get_default_embedding_model(corpus_for_fit=[c.text for c in SAMPLE_CHUNKS])
    vs = VectorStore(embedding_model)
    vs.build(SAMPLE_CHUNKS)

    results = vs.search("MAC reduction in ResNet pruning", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c2"


def test_bm25_index_search_returns_relevant_chunk():
    bm25 = BM25Index()
    bm25.build(SAMPLE_CHUNKS)

    results = bm25.search("drift detection retraining pipelines", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c3"


def test_hybrid_retriever_merges_dense_and_sparse():
    retriever = _build_retriever()

    results = retriever.retrieve("diabetes HbA1c treatment", top_k=2)

    assert len(results) <= 2
    assert results[0].chunk.chunk_id == "c1"
    assert all(r.retrieval_method == "hybrid" for r in results)


def test_hybrid_retriever_handles_unmatched_query_gracefully():
    retriever = _build_retriever()

    results = retriever.retrieve("zzz nonexistent term xyz", top_k=2)

    assert isinstance(results, list)
