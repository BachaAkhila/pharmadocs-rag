"""
Core RAG chain: retrieval + generation with confidence scoring.

`RAGChain` is the single-step RAG pipeline (retrieve -> generate).
It also computes a simple confidence score from retrieval results,
which `LangGraphAgent` (in `agent_graph.py`) uses to decide whether
to retry with a reformulated query.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from app.agents.llm_client import LLMClient, LLMResponse, get_default_llm_client
from app.retrieval.vectorstore import HybridRetriever, RetrievedChunk

_STOPWORDS_CONF = {
    "what", "how", "why", "when", "where", "is", "are", "the", "a", "an",
    "did", "does", "do", "of", "in", "on", "for", "to", "and", "or", "tell",
    "me", "about",
}


@dataclass
class RAGResult:
    """Full result of a single RAG query, including provenance and timing."""

    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    confidence: float
    model_name: str
    latency_ms: float
    metadata: dict = field(default_factory=dict)


class RAGChain:
    """Single-step retrieval-augmented generation pipeline."""

    def __init__(self, retriever: HybridRetriever, llm_client: LLMClient | None = None, top_k: int = 5):
        self.retriever = retriever
        self.llm_client = llm_client or get_default_llm_client()
        self.top_k = top_k

    def run(self, query: str) -> RAGResult:
        """Execute retrieval and generation for a single query."""
        start = time.perf_counter()

        retrieved = self.retriever.retrieve(query, top_k=self.top_k)
        confidence = self._compute_confidence(retrieved, query)

        llm_response: LLMResponse = self.llm_client.generate(query, retrieved)

        latency_ms = (time.perf_counter() - start) * 1000

        return RAGResult(
            query=query,
            answer=llm_response.answer,
            retrieved_chunks=retrieved,
            confidence=confidence,
            model_name=llm_response.model_name,
            latency_ms=latency_ms,
            metadata={"top_k": self.top_k},
        )

    @staticmethod
    def _compute_confidence(retrieved: list[RetrievedChunk], query: str = "") -> float:
        """Confidence heuristic combining retrieval score and lexical overlap.

        Returns a value in [0, 1]. Used by the agent layer to decide
        whether retrieval results are strong enough to answer directly,
        or whether the query should be reformulated and retried.

        Combines two signals:
          - normalized top RRF retrieval score
          - fraction of query content-terms that appear in the top
            retrieved chunk (lexical grounding check)
        """
        if not retrieved:
            return 0.0

        top_score = retrieved[0].score
        normalized_score = min(top_score * 30, 1.0)

        if query:
            query_terms = set(re.findall(r"[a-z0-9]+", query.lower())) - _STOPWORDS_CONF
            if query_terms:
                chunk_terms = set(re.findall(r"[a-z0-9]+", retrieved[0].chunk.text.lower()))
                overlap_ratio = len(query_terms & chunk_terms) / len(query_terms)
            else:
                overlap_ratio = 1.0
        else:
            overlap_ratio = 1.0

        count_factor = min(len(retrieved) / 3, 1.0)

        return round(normalized_score * count_factor * overlap_ratio, 4)
