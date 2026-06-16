"""
Embedding backends for the retrieval pipeline.

Provides a common `EmbeddingModel` interface with two implementations:

- `TfidfEmbedding`: a lightweight, dependency-free embedding based on
  hashed TF-IDF vectors. Used as the default so the project runs
  end-to-end without API keys or large model downloads.
- `SentenceTransformerEmbedding`: a wrapper around `sentence-transformers`
  (e.g. BAAI/bge-small-en, all-MiniLM-L6-v2) for production-quality
  dense embeddings, matching the embedding configuration tuning
  described in the project README.

Swapping backends is a one-line change at the call site, which keeps
the rest of the pipeline (vector store, retrieval, agents) agnostic
to the embedding implementation.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np


class EmbeddingModel(ABC):
    """Common interface for all embedding backends."""

    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dim) float32 array of embeddings."""
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        """Convenience wrapper for embedding a single query string."""
        return self.embed([text])[0]


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class TfidfEmbedding(EmbeddingModel):
    """Hashed TF-IDF embedding.

    Tokens are hashed into a fixed-size vector space, weighted by
    term frequency and a corpus-level inverse document frequency
    computed at fit time. This avoids any external model downloads
    or API calls while still producing meaningful similarity scores
    for retrieval over a small document corpus.

    Not intended as a substitute for transformer-based embeddings in
    production -- see `SentenceTransformerEmbedding` for that path.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self._idf: np.ndarray | None = None

    def fit(self, corpus: list[str]) -> "TfidfEmbedding":
        """Compute inverse document frequencies from a reference corpus."""
        doc_freq = np.zeros(self.dim, dtype=np.float64)
        n_docs = len(corpus)

        for text in corpus:
            seen = set()
            for token in _tokenize(text):
                idx = self._hash(token)
                if idx not in seen:
                    doc_freq[idx] += 1
                    seen.add(idx)

        # smoothed IDF
        self._idf = np.log((1 + n_docs) / (1 + doc_freq)) + 1.0
        return self

    def _hash(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(h, 16) % self.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._idf is None:
            raise RuntimeError("TfidfEmbedding must be fit() before use")

        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = _tokenize(text)
            if not tokens:
                continue
            counts = Counter(self._hash(t) for t in tokens)
            total = len(tokens)
            for idx, count in counts.items():
                tf = count / total
                vectors[i, idx] = tf * self._idf[idx]

            # L2 normalize for cosine similarity via inner product
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm

        return vectors


class SentenceTransformerEmbedding(EmbeddingModel):
    """Dense embeddings via sentence-transformers.

    Used for the embedding configuration comparison (e.g.
    all-MiniLM-L6-v2 vs BAAI/bge-small-en) referenced in the README.
    Requires the `sentence-transformers` package and will download
    model weights on first use -- not enabled by default in this
    portfolio build to keep setup dependency-free.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedding. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)


def get_default_embedding_model(corpus_for_fit: list[str] | None = None) -> EmbeddingModel:
    """Factory returning the default (dependency-free) embedding model.

    If `corpus_for_fit` is provided, the TF-IDF model is fit on it
    immediately so it is ready for use.
    """
    model = TfidfEmbedding(dim=512)
    if corpus_for_fit is not None:
        model.fit(corpus_for_fit)
    return model
