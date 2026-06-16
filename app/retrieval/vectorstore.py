"""
Vector store and hybrid retrieval.

Wraps a FAISS index for dense vector search and a BM25 index for
sparse lexical search, combining both via a configurable hybrid
re-ranking strategy. This mirrors the "hybrid retrieval combining
dense vector search with BM25 sparse re-ranking" approach described
in the project README, which improved answer faithfulness over pure
vector search.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.ingestion.chunking import Chunk
from app.retrieval.embeddings import EmbeddingModel, _tokenize

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval, with its relevance score."""

    chunk: Chunk
    score: float
    retrieval_method: str  # "dense", "sparse", or "hybrid"


class VectorStore:
    """FAISS-backed dense vector index over document chunks.

    Uses an inner-product index (IndexFlatIP) over L2-normalized
    embeddings, which is equivalent to cosine similarity search.
    """

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Embed all chunks and construct the FAISS index."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self.embedding_model.embed(texts)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        logger.info("Built FAISS index with %d vectors (dim=%d)", len(chunks), embeddings.shape[1])

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the top_k chunks most similar to the query."""
        if self.index is None:
            raise RuntimeError("VectorStore.build() must be called before search()")

        query_vec = self.embedding_model.embed_query(query).reshape(1, -1)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(RetrievedChunk(chunk=self.chunks[idx], score=float(score), retrieval_method="dense"))
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([c.__dict__ for c in self.chunks], f)
        logger.info("Saved vector store to %s", path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.json", "r", encoding="utf-8") as f:
            self.chunks = [Chunk(**c) for c in json.load(f)]
        logger.info("Loaded vector store from %s (%d chunks)", path, len(self.chunks))


class BM25Index:
    """BM25 sparse lexical index over document chunks."""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("Built BM25 index with %d documents", len(chunks))

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self.bm25 is None:
            raise RuntimeError("BM25Index.build() must be called before search()")

        scores = self.bm25.get_scores(_tokenize(query))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append(RetrievedChunk(chunk=self.chunks[idx], score=float(scores[idx]), retrieval_method="sparse"))
        return results


class HybridRetriever:
    """Combines dense (FAISS) and sparse (BM25) retrieval with re-ranking.

    Both retrievers run independently over the same chunk set, and
    results are merged via reciprocal rank fusion (RRF), which is
    robust to the different score scales of dense cosine similarity
    and BM25 lexical scores.
    """

    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index, rrf_k: int = 60):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5, candidate_pool: int = 20) -> list[RetrievedChunk]:
        """Retrieve and re-rank chunks for a query.

        Args:
            query: natural language query string.
            top_k: number of final results to return.
            candidate_pool: number of candidates retrieved from each
                index before fusion.
        """
        dense_results = self.vector_store.search(query, top_k=candidate_pool)
        sparse_results = self.bm25_index.search(query, top_k=candidate_pool)

        # reciprocal rank fusion
        fused_scores: dict[str, float] = {}
        chunk_lookup: dict[str, RetrievedChunk] = {}

        for rank, result in enumerate(dense_results):
            cid = result.chunk.chunk_id
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            chunk_lookup[cid] = result

        for rank, result in enumerate(sparse_results):
            cid = result.chunk.chunk_id
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            if cid not in chunk_lookup:
                chunk_lookup[cid] = result

        ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:top_k]

        return [
            RetrievedChunk(chunk=chunk_lookup[cid].chunk, score=fused_scores[cid], retrieval_method="hybrid")
            for cid in ranked_ids
        ]
