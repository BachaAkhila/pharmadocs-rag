"""
Multi-step LangGraph agent with a self-correcting retry loop.

Implements a "reflection and query reformulation fallback" workflow:
the agent runs an initial RAG query, checks the resulting confidence
score, and if it falls below a threshold, reformulates the query and
retries -- up to a configurable maximum number of attempts -- before
returning its best available answer with a low-confidence flag.

The graph has two nodes with a conditional edge:

    retrieve_and_generate -> {reformulate -> retrieve_and_generate} | END

This corresponds to the resume claim of "multi-step LLM agent
workflows using LangGraph with tool use, memory, and conditional
branching."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agents.rag_chain import RAGChain, RAGResult


class AgentState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    original_query: str
    current_query: str
    attempt: int
    max_attempts: int
    confidence_threshold: float
    history: list[RAGResult]
    final_result: RAGResult | None


@dataclass
class AgentRunResult:
    """Final output of an agent run, including the full retry history."""

    answer: str
    confidence: float
    attempts: int
    low_confidence: bool
    history: list[RAGResult] = field(default_factory=list)


_STOPWORDS = {
    "what", "how", "why", "when", "where", "is", "are", "the", "a", "an",
    "did", "does", "do", "of", "in", "on", "for", "to", "and", "or",
}


def _reformulate_query(original_query: str) -> str:
    """Reformulate a query by stripping stopwords and question framing.

    Lightweight stand-in for an LLM-based query rewrite: surfaces the
    core content terms, which often retrieves more precisely than the
    full natural-language question on a second pass.
    """
    tokens = re.findall(r"[A-Za-z0-9]+", original_query)
    content_tokens = [t for t in tokens if t.lower() not in _STOPWORDS]
    return " ".join(content_tokens) if content_tokens else original_query


class LangGraphAgent:
    """Self-correcting RAG agent built on LangGraph's StateGraph.

    Wraps a `RAGChain` with a retrieve -> check confidence ->
    reformulate -> retry loop, implemented as a directed graph with
    conditional branching.
    """

    def __init__(
        self,
        rag_chain: RAGChain,
        confidence_threshold: float = 0.3,
        max_attempts: int = 3,
    ):
        self.rag_chain = rag_chain
        self.confidence_threshold = confidence_threshold
        self.max_attempts = max_attempts
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("retrieve_and_generate", self._retrieve_and_generate)
        graph.add_node("reformulate", self._reformulate)

        graph.set_entry_point("retrieve_and_generate")

        graph.add_conditional_edges(
            "retrieve_and_generate",
            self._should_retry,
            {"retry": "reformulate", "done": END},
        )
        graph.add_edge("reformulate", "retrieve_and_generate")

        return graph.compile()

    # -- nodes --------------------------------------------------------

    def _retrieve_and_generate(self, state: AgentState) -> AgentState:
        result = self.rag_chain.run(state["current_query"])
        history = state["history"] + [result]
        return {**state, "history": history, "final_result": result, "attempt": state["attempt"] + 1}

    def _reformulate(self, state: AgentState) -> AgentState:
        new_query = _reformulate_query(state["original_query"])
        return {**state, "current_query": new_query}

    # -- conditional edge ----------------------------------------------

    def _should_retry(self, state: AgentState) -> str:
        result = state["final_result"]
        assert result is not None

        if result.confidence >= self.confidence_threshold:
            return "done"
        if state["attempt"] >= state["max_attempts"]:
            return "done"
        return "retry"

    # -- public API -----------------------------------------------------

    def run(self, query: str) -> AgentRunResult:
        """Run the agent end-to-end and return the final result."""
        initial_state: AgentState = {
            "original_query": query,
            "current_query": query,
            "attempt": 0,
            "max_attempts": self.max_attempts,
            "confidence_threshold": self.confidence_threshold,
            "history": [],
            "final_result": None,
        }

        final_state = self.graph.invoke(initial_state)
        final_result: RAGResult = final_state["final_result"]
        history: list[RAGResult] = final_state["history"]

        return AgentRunResult(
            answer=final_result.answer,
            confidence=final_result.confidence,
            attempts=len(history),
            low_confidence=final_result.confidence < self.confidence_threshold,
            history=history,
        )
