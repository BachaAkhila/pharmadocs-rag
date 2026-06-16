"""
MLflow experiment tracking for the RAG evaluation pipeline.

Each evaluation run (a "release candidate" -- a specific combination
of chunking strategy, retrieval configuration, and LLM backend) is
logged as an MLflow run, recording:

  - parameters: chunking strategy, top_k, confidence threshold,
    embedding model, LLM model name
  - metrics: faithfulness, answer_relevancy, context_recall, overall,
    mean latency
  - artifacts: per-query evaluation scores (JSON)

This gives a versioned, reproducible audit trail of every pipeline
configuration evaluated -- corresponding to the "MLflow with versioned
artifacts and audit trails" claim in the project README, applied here
to the RAG/evaluation pipeline.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow

from app.evaluation.ragas_eval import EvalScores


def log_evaluation_run(
    run_name: str,
    params: dict,
    summary: dict[str, float],
    scores: list[EvalScores],
    mean_latency_ms: float | None = None,
    experiment_name: str = "pharmadocs_rag_evaluation",
    tracking_uri: str = "sqlite:///mlruns.db",
) -> str:
    """Log a single evaluation run to MLflow.

    Args:
        run_name: human-readable name for this run (e.g.
            "semantic-chunking-bge-small").
        params: configuration parameters for this run (chunking
            strategy, embedding model, top_k, etc).
        summary: aggregate metric dict from `summarize_scores`.
        scores: per-query EvalScores, logged as a JSON artifact.
        mean_latency_ms: optional mean end-to-end latency to log
            alongside quality metrics.
        experiment_name: MLflow experiment to log under.
        tracking_uri: local directory (or remote URI) for the MLflow
            tracking store.

    Returns:
        The MLflow run ID.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(summary)

        if mean_latency_ms is not None:
            mlflow.log_metric("mean_latency_ms", mean_latency_ms)

        with tempfile.TemporaryDirectory() as tmpdir:
            scores_path = Path(tmpdir) / "per_query_scores.json"
            with open(scores_path, "w", encoding="utf-8") as f:
                json.dump([dict(s.__dict__, overall=s.overall) for s in scores], f, indent=2)
            mlflow.log_artifact(str(scores_path))

        return run.info.run_id


def get_best_run(
    experiment_name: str = "pharmadocs_rag_evaluation",
    metric: str = "overall",
    tracking_uri: str = "sqlite:///mlruns.db",
) -> dict | None:
    """Return the run with the highest value for `metric` in the experiment.

    Used to identify the best-performing pipeline configuration across
    all logged experiments (e.g. comparing chunking strategies or
    embedding models).
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )

    if not runs:
        return None

    best = runs[0]
    return {
        "run_id": best.info.run_id,
        "run_name": best.data.tags.get("mlflow.runName"),
        "params": best.data.params,
        "metrics": best.data.metrics,
    }
