"""
Standalone evaluation script.

Builds the pipeline, runs the RAGAS-style evaluation suite against the
golden set, logs results to MLflow, and checks the regression gate
against the stored baseline.

Exits with a non-zero status code if the regression gate fails, so
this script can be used directly as a CI/CD step that blocks
deployment on quality regressions.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --chunking fixed --top-k 3
    python scripts/run_evaluation.py --update-baseline
"""

from __future__ import annotations

import argparse
import sys
import time

from app.evaluation.ragas_eval import (
    RegressionGate,
    load_golden_set,
    run_evaluation,
    summarize_scores,
)
from app.monitoring.experiment_tracking import log_evaluation_run
from app.pipeline import build_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAGAS-style evaluation with regression gating")
    parser.add_argument("--chunking", default="semantic", choices=["semantic", "fixed"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=0.05, help="Regression tolerance per metric")
    parser.add_argument("--update-baseline", action="store_true", help="Overwrite the stored baseline with this run's scores")
    parser.add_argument("--run-name", default=None, help="MLflow run name (defaults to '<chunking>-top<k>')")
    args = parser.parse_args()

    run_name = args.run_name or f"{args.chunking}-top{args.top_k}"

    print(f"Building pipeline (chunking={args.chunking}, top_k={args.top_k})...")
    pipeline = build_pipeline(chunking_strategy=args.chunking, top_k=args.top_k)

    golden_set = load_golden_set()
    print(f"Running evaluation over {len(golden_set)} queries...")

    start = time.perf_counter()
    scores = run_evaluation(pipeline.rag_chain.run, golden_set)
    elapsed_ms = (time.perf_counter() - start) * 1000
    mean_latency_ms = elapsed_ms / len(golden_set)

    summary = summarize_scores(scores)

    print()
    print("Per-query results:")
    for s in scores:
        print(f"  overall={s.overall:.3f}  faithfulness={s.faithfulness:.3f}  "
              f"relevancy={s.answer_relevancy:.3f}  recall={s.context_recall:.3f}  | {s.query[:60]}")

    print()
    print("Summary:", summary)
    print(f"Mean latency per query: {mean_latency_ms:.2f} ms")

    log_evaluation_run(
        run_name=run_name,
        params={
            "chunking_strategy": args.chunking,
            "top_k": args.top_k,
            "embedding_model": "tfidf-512",
            "llm": "mock-extractive-v1",
        },
        summary=summary,
        scores=scores,
        mean_latency_ms=mean_latency_ms,
    )

    gate = RegressionGate(tolerance=args.tolerance)

    if args.update_baseline:
        gate.save_baseline(summary)
        print("\nBaseline updated.")
        return 0

    passed, failures = gate.check(summary)

    print()
    if passed:
        print("REGRESSION GATE: PASSED")
        return 0
    else:
        print("REGRESSION GATE: FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
