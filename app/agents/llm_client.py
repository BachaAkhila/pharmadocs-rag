"""
LLM interface for answer generation.

Provides a common `LLMClient` interface with a `MockLLMClient`
implementation that generates extractive, template-based answers
directly from retrieved context -- no API key or external model
required. This keeps the project runnable end-to-end out of the box
while preserving the same interface an `OpenAIClient` (GPT-4) or
local Ollama client would implement.

Swapping `MockLLMClient` for `OpenAIClient` in `app/agents/rag_chain.py`
is a one-line change.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.retrieval.vectorstore import RetrievedChunk


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""

    answer: str
    model_name: str
    finish_reason: str = "stop"


class LLMClient(ABC):
    """Common interface for all LLM backends used in the RAG chain."""

    @abstractmethod
    def generate(self, query: str, context_chunks: list[RetrievedChunk]) -> LLMResponse:
        """Generate an answer grounded in the provided context chunks."""
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Deterministic, extractive answer generator.

    Constructs answers by extracting the most query-relevant sentence
    from each retrieved chunk, rather than calling a generative model.
    This makes the pipeline fully reproducible and free of external
    dependencies, useful for testing retrieval and evaluation layers
    in isolation from generation quality.

    In production this is replaced by `OpenAIClient` (GPT-4).
    """

    def __init__(self):
        self.model_name = "mock-extractive-v1"

    def generate(self, query: str, context_chunks: list[RetrievedChunk]) -> LLMResponse:
        if not context_chunks:
            return LLMResponse(
                answer="I don't have enough information in the knowledge base to answer this question.",
                model_name=self.model_name,
                finish_reason="no_context",
            )

        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))

        best_sentence = None
        best_overlap = -1
        best_source = None

        for retrieved in context_chunks:
            sentences = re.split(r"(?<=[.!?])\s+", retrieved.chunk.text)
            for sentence in sentences:
                terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                overlap = len(query_terms & terms)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sentence = sentence.strip()
                    best_source = retrieved.chunk.title

        if best_sentence is None:
            best_sentence = context_chunks[0].chunk.text.split(".")[0] + "."
            best_source = context_chunks[0].chunk.title

        answer = f"{best_sentence}\n\n(Source: {best_source})"
        return LLMResponse(answer=answer, model_name=self.model_name)


class OpenAIClient(LLMClient):
    """GPT-4 backed answer generation via the OpenAI API.

    Not enabled by default in this portfolio build -- requires an
    `OPENAI_API_KEY` environment variable and the `openai` package.
    Documents the production swap-in path referenced in the project
    README ("RAG pipeline using LangChain, FAISS, GPT-4").
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAIClient. Install with: pip install openai"
            ) from exc

        self.client = OpenAI()
        self.model_name = model_name

    def generate(self, query: str, context_chunks: list[RetrievedChunk]) -> LLMResponse:
        context_text = "\n\n".join(f"[{c.chunk.title}]\n{c.chunk.text}" for c in context_chunks)

        system_prompt = (
            "You are a research assistant. Answer the user's question using ONLY "
            "the provided context. If the context does not contain the answer, "
            "say so explicitly. Cite the source document title for any claim."
        )
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )

        choice = response.choices[0]
        return LLMResponse(
            answer=choice.message.content,
            model_name=self.model_name,
            finish_reason=choice.finish_reason,
        )


def get_default_llm_client() -> LLMClient:
    """Factory returning the default (dependency-free) LLM client."""
    return MockLLMClient()
