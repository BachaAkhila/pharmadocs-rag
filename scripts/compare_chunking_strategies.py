"""
Compare chunking strategies (semantic vs fixed-size) via the RAGAS
evaluation suite, logging both to MLflow for side-by-side comparison.

This corresponds to the "chunking strategy optimization" referenced
in the project README's experience bullets -- a concrete, runnable
A/B comparison rather than an unverifiable claim.

Usage:
    python scripts/compare_chunking_strategies.py
"""

from __future__ import annotations

from app.evaluation.ragas_eval import load_golden_set, run_evaluation, summarize_scores
from app.monitoring.experiment_tracking import log_evaluation_run
from app.pipeline import build_pipeline


def main() -> None:
    golden_set = load_golden_set()
    results = {}

    for strategy in ["semantic", "fixed"]:
        print(f"\n=== Chunking strategy: {strategy} ===")
        pipeline = build_pipeline(chunking_strategy=strategy)

        scores = run_evaluation(pipeline.rag_chain.run, golden_set)
        summary = summarize_scores(scores)
        results[strategy] = summary

        print(f"  faithfulness:     {summary['faithfulness']:.4f}")
        print(f"  answer_relevancy: {summary['answer_relevancy']:.4f}")
        print(f"  context_recall:   {summary['context_recall']:.4f}")
        print(f"  overall:          {summary['overall']:.4f}")

        log_evaluation_run(
            run_name=f"chunking-comparison-{strategy}",
            params={"chunking_strategy": strategy, "embedding_model": "tfidf-512", "llm": "mock-extractive-v1"},
            summary=summary,
            scores=scores,
        )

    print("\n=== Comparison ===")
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "overall"]:
        semantic_val = results["semantic"][metric]
        fixed_val = results["fixed"][metric]
        delta = semantic_val - fixed_val
        winner = "semantic" if delta > 0 else ("fixed" if delta < 0 else "tie")
        print(f"  {metric:18s} semantic={semantic_val:.4f}  fixed={fixed_val:.4f}  "
              f"delta={delta:+.4f}  ({winner})")


if __name__ == "__main__":
    main()
