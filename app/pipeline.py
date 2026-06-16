"""
Pipeline factory: builds the full RAG stack from raw documents.

This module wires together ingestion, embedding, vector store, hybrid
retrieval, and the agent layer into a single ready-to-query object.
Used by the FastAPI app at startup, by evaluation scripts, and by
tests, so there is exactly one place that defines "how the pipeline
is constructed."
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.agents.agent_graph import LangGraphAgent
from app.agents.llm_client import get_default_llm_client
from app.agents.rag_chain import RAGChain
from app.ingestion.chunking import run_ingestion_pipeline
from app.retrieval.embeddings import get_default_embedding_model
from app.retrieval.vectorstore import BM25Index, HybridRetriever, VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class PharmaDocsPipeline:
    """Fully assembled RAG pipeline: retriever + RAG chain + agent."""

    def __init__(self, retriever: HybridRetriever, rag_chain: RAGChain, agent: LangGraphAgent):
        self.retriever = retriever
        self.rag_chain = rag_chain
        self.agent = agent


def build_pipeline(
    raw_path: str | Path = "data/raw/pharma_abstracts.json",
    chunking_strategy: str = "semantic",
    top_k: int = 5,
    confidence_threshold: float = 0.3,
    max_attempts: int = 3,
) -> PharmaDocsPipeline:
    """Build the full RAG pipeline from raw documents.

    This is the single entry point used by the API, evaluation
    scripts, and tests to ensure consistent pipeline construction.
    """
    logger.info("Building pipeline (chunking=%s, top_k=%d)", chunking_strategy, top_k)

    chunks = run_ingestion_pipeline(raw_path=raw_path, strategy=chunking_strategy)

    embedding_model = get_default_embedding_model(corpus_for_fit=[c.text for c in chunks])

    vector_store = VectorStore(embedding_model)
    vector_store.build(chunks)

    bm25_index = BM25Index()
    bm25_index.build(chunks)

    retriever = HybridRetriever(vector_store, bm25_index)

    rag_chain = RAGChain(retriever, llm_client=get_default_llm_client(), top_k=top_k)

    agent = LangGraphAgent(rag_chain, confidence_threshold=confidence_threshold, max_attempts=max_attempts)

    logger.info("Pipeline ready: %d chunks indexed", len(chunks))
    return PharmaDocsPipeline(retriever=retriever, rag_chain=rag_chain, agent=agent)
