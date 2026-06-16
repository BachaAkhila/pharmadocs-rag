"""
RAGAS-style evaluation framework.

Implements three of the core RAGAS metrics -- faithfulness, answer
relevancy, and context recall -- using lexical overlap heuristics
rather than an LLM-as-judge, so the evaluation suite runs without API
calls. Each metric returns a score in [0, 1]; higher is better.

`run_evaluation` runs these metrics across a labeled evaluation set
and `RegressionGate` compares results against a stored baseline to
flag quality regressions before a model or prompt change is promoted
-- the "regression gates ensuring every model and prompt change is
validated before production" referenced in the project README.

In production, these heuristic metrics are replaced by the `ragas`
package's LLM-judged equivalents (same function signatures, swap the
implementation body).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.agents.rag_chain import RAGResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass
class EvalScores:
    """Per-query evaluation scores."""

    query: str
    faithfulness: float
    answer_relevancy: float
    context_recall: float

    @property
    def overall(self) -> float:
        return round((self.faithfulness + self.answer_relevancy + self.context_recall) / 3, 4)


def faithfulness(result: RAGResult) -> float:
    """Measures whether the generated answer is grounded in retrieved context.

    Heuristic: fraction of content tokens in the answer that also
    appear somewhere in the retrieved context. An answer that
    introduces many terms absent from the context is likely
    hallucinated.

    Production equivalent: `ragas.metrics.faithfulness`, which uses an
    LLM to decompose the answer into atomic claims and verify each
    against the context.
    """
    answer_tokens = _tokens(result.answer)
    if not answer_tokens:
        return 0.0

    context_tokens: set[str] = set()
    for chunk in result.retrieved_chunks:
        context_tokens |= _tokens(chunk.chunk.text)

    if not context_tokens:
        return 0.0

    grounded = answer_tokens & context_tokens
    return round(len(grounded) / len(answer_tokens), 4)


def answer_relevancy(result: RAGResult, reference_answer: str | None = None) -> float:
    """Measures whether the answer addresses the query.

    Heuristic: token overlap between the query and the answer. If a
    reference (gold) answer is provided, overlap with the reference
    is also considered and averaged in.

    Production equivalent: `ragas.metrics.answer_relevancy`, which
    generates synthetic questions from the answer and measures their
    similarity to the original query via embeddings.
    """
    query_tokens = _tokens(result.query)
    answer_tokens = _tokens(result.answer)

    if not query_tokens or not answer_tokens:
        return 0.0

    query_overlap = len(query_tokens & answer_tokens) / len(query_tokens)

    if reference_answer:
        ref_tokens = _tokens(reference_answer)
        if ref_tokens:
            ref_overlap = len(ref_tokens & answer_tokens) / len(ref_tokens)
            return round((query_overlap + ref_overlap) / 2, 4)

    return round(query_overlap, 4)


def context_recall(result: RAGResult, reference_answer: str) -> float:
    """Measures whether retrieved context contains the information
    needed to produce the reference answer.

    Heuristic: fraction of reference-answer tokens that appear in the
    union of retrieved chunk texts.

    Production equivalent: `ragas.metrics.context_recall`, which uses
    an LLM to attribute each sentence in the reference answer to a
    retrieved context chunk.
    """
    ref_tokens = _tokens(reference_answer)
    if not ref_tokens:
        return 0.0

    context_tokens: set[str] = set()
    for chunk in result.retrieved_chunks:
        context_tokens |= _tokens(chunk.chunk.text)

    if not context_tokens:
        return 0.0

    covered = ref_tokens & context_tokens
    return round(len(covered) / len(ref_tokens), 4)


def evaluate_result(result: RAGResult, reference_answer: str | None = None) -> EvalScores:
    """Compute all three RAGAS-style metrics for a single RAG result."""
    return EvalScores(
        query=result.query,
        faithfulness=faithfulness(result),
        answer_relevancy=answer_relevancy(result, reference_answer),
        context_recall=context_recall(result, reference_answer) if reference_answer else 0.0,
    )


def load_golden_set(path: str | Path = "data/eval/golden_set.json") -> list[dict]:
    """Load the labeled evaluation set (queries + reference answers)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(agent_run_fn, golden_set: list[dict]) -> list[EvalScores]:
    """Run evaluation over the golden set.

    Args:
        agent_run_fn: callable taking a query string and returning a
            `RAGResult` (e.g. `RAGChain.run`).
        golden_set: list of {"query": ..., "reference_answer": ...} dicts.

    Returns:
        list of EvalScores, one per golden set entry.
    """
    scores = []
    for item in golden_set:
        result = agent_run_fn(item["query"])
        scores.append(evaluate_result(result, item.get("reference_answer")))

    logger.info("Evaluated %d queries", len(scores))
    return scores


def summarize_scores(scores: list[EvalScores]) -> dict[str, float]:
    """Aggregate per-query scores into mean metric values."""
    if not scores:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0, "overall": 0.0}

    n = len(scores)
    return {
        "faithfulness": round(sum(s.faithfulness for s in scores) / n, 4),
        "answer_relevancy": round(sum(s.answer_relevancy for s in scores) / n, 4),
        "context_recall": round(sum(s.context_recall for s in scores) / n, 4),
        "overall": round(sum(s.overall for s in scores) / n, 4),
    }


class RegressionGate:
    """Compares evaluation results against a stored baseline.

    Used to block a model or prompt change from being promoted to
    production if any metric regresses beyond a configurable
    tolerance -- the "regression gates" referenced in the README.
    """

    def __init__(self, baseline_path: str | Path = "data/eval/baseline_scores.json", tolerance: float = 0.05):
        self.baseline_path = Path(baseline_path)
        self.tolerance = tolerance

    def load_baseline(self) -> dict[str, float] | None:
        if not self.baseline_path.exists():
            return None
        with open(self.baseline_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_baseline(self, summary: dict[str, float]) -> None:
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved new baseline scores to %s", self.baseline_path)

    def check(self, current_summary: dict[str, float]) -> tuple[bool, list[str]]:
        """Check current scores against the baseline.

        Returns:
            (passed, failures) where `passed` is True if no metric
            regressed beyond `tolerance`, and `failures` lists
            human-readable descriptions of any regressions.
        """
        baseline = self.load_baseline()
        if baseline is None:
            logger.info("No baseline found; treating current scores as the new baseline")
            self.save_baseline(current_summary)
            return True, []

        failures = []
        for metric, current_value in current_summary.items():
            baseline_value = baseline.get(metric)
            if baseline_value is None:
                continue
            if current_value < baseline_value - self.tolerance:
                failures.append(
                    f"{metric} regressed: {baseline_value:.4f} -> {current_value:.4f} "
                    f"(tolerance={self.tolerance})"
                )

        passed = len(failures) == 0
        return passed, failures
