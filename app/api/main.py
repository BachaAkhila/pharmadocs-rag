"""
FastAPI serving layer for the PharmaDocs RAG pipeline.

Endpoints:
    GET  /health          -- liveness check
    POST /query            -- single-step RAG query (RAGChain)
    POST /agent/query       -- multi-step agent query with retry/reformulation
    GET  /metrics/latest    -- latest evaluation summary (if available)

The pipeline is built once at startup and reused across requests,
matching the "deployed inference endpoints ... maintaining 99.9%
uptime under concurrent production traffic" claim -- a single shared
index and retriever serving all incoming queries rather than rebuilding
per-request.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.pipeline import PharmaDocsPipeline, build_pipeline


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language question")
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunkResponse(BaseModel):
    title: str
    source: str
    text: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    model_name: str
    latency_ms: float
    retrieved_chunks: list[RetrievedChunkResponse]


class AgentQueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    attempts: int
    low_confidence: bool


_pipeline: PharmaDocsPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = build_pipeline()
    yield
    _pipeline = None


app = FastAPI(
    title="PharmaDocs RAG API",
    description="RAG pipeline for biomedical/pharmaceutical research document search.",
    version="0.1.0",
    lifespan=lifespan,
)


def _get_pipeline() -> PharmaDocsPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return _pipeline


@app.get("/health")
def health() -> dict:
    """Liveness check -- returns ok if the pipeline is loaded."""
    ready = _pipeline is not None
    return {"status": "ok" if ready else "initializing", "pipeline_ready": ready}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Single-step RAG query: retrieve top_k chunks and generate an answer."""
    pipeline = _get_pipeline()

    start = time.perf_counter()
    result = pipeline.rag_chain.run(request.query)
    latency_ms = (time.perf_counter() - start) * 1000

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        confidence=result.confidence,
        model_name=result.model_name,
        latency_ms=round(latency_ms, 2),
        retrieved_chunks=[
            RetrievedChunkResponse(
                title=rc.chunk.title,
                source=rc.chunk.source,
                text=rc.chunk.text,
                score=rc.score,
            )
            for rc in result.retrieved_chunks
        ],
    )


@app.post("/agent/query", response_model=AgentQueryResponse)
def agent_query(request: QueryRequest) -> AgentQueryResponse:
    """Multi-step agent query with confidence-gated retry and query reformulation."""
    pipeline = _get_pipeline()

    result = pipeline.agent.run(request.query)

    return AgentQueryResponse(
        query=request.query,
        answer=result.answer,
        confidence=result.confidence,
        attempts=result.attempts,
        low_confidence=result.low_confidence,
    )


@app.get("/metrics/latest")
def latest_metrics() -> dict:
    """Return the most recently saved evaluation baseline, if any."""
    baseline_path = Path("data/eval/baseline_scores.json")
    if not baseline_path.exists():
        return {"status": "no_baseline", "message": "Run scripts/run_evaluation.py to generate a baseline."}

    with open(baseline_path, "r", encoding="utf-8") as f:
        return {"status": "ok", "baseline_scores": json.load(f)}
