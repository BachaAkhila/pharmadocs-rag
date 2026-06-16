"""
Airflow DAG: drift monitoring and automated re-evaluation.

Schedule: daily.

Tasks:
    1. collect_query_logs   -- load recent query confidence/latency
                                logs (simulated here from a JSON log
                                file written by the API).
    2. check_drift           -- run PSI + KS-test against the
                                reference window (see
                                app.monitoring.drift_detection).
    3. branch_on_drift       -- conditional branch: if drift detected,
                                proceed to re-evaluation; otherwise
                                skip to end.
    4. run_regression_suite  -- re-run the RAGAS evaluation suite and
                                check the regression gate (see
                                scripts/run_evaluation.py).

This corresponds to the "drift detection via PSI and KS-test triggers
Airflow DAG retraining automatically" claim in the project README.
In this project, "retraining" maps to re-running the evaluation
regression suite against the current pipeline configuration -- since
the RAG pipeline has no trainable model weights, "retraining" in the
classical sense does not apply, but the automated quality-gate
re-check plays the equivalent role.

Note: this DAG file is provided for portfolio/demo purposes and is not
wired to a live Airflow scheduler in this repository. It documents the
intended orchestration structure and can be dropped into an Airflow
dags/ folder with a working Airflow installation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

try:
    from airflow import DAG
    from airflow.operators.python import BranchPythonOperator, PythonOperator
    from airflow.operators.empty import EmptyOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

from app.monitoring.drift_detection import check_pipeline_drift

logger = logging.getLogger(__name__)

QUERY_LOG_PATH = Path("data/monitoring/query_logs.jsonl")
CURRENT_WINDOW_DAYS = 1


def collect_query_logs(**context) -> dict:
    """Load recent query logs and split into reference/current windows.

    In production this would query a logging store (e.g. CloudWatch
    Logs, BigQuery) for confidence and latency values from API
    responses over the last N days. Here it reads a local JSONL log
    file written by the FastAPI app (one record per query).
    """
    if not QUERY_LOG_PATH.exists():
        logger.warning("No query log found at %s; skipping drift check", QUERY_LOG_PATH)
        return {"has_data": False}

    with open(QUERY_LOG_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    cutoff = datetime.utcnow() - timedelta(days=CURRENT_WINDOW_DAYS)
    reference = [r for r in records if datetime.fromisoformat(r["timestamp"]) < cutoff]
    current = [r for r in records if datetime.fromisoformat(r["timestamp"]) >= cutoff]

    return {
        "has_data": len(reference) > 0 and len(current) > 0,
        "reference_confidences": [r["confidence"] for r in reference],
        "current_confidences": [r["confidence"] for r in current],
        "reference_latencies": [r["latency_ms"] for r in reference],
        "current_latencies": [r["latency_ms"] for r in current],
    }


def check_drift_task(**context) -> str:
    """Run drift checks and return the next task id for branching."""
    ti = context["ti"]
    data = ti.xcom_pull(task_ids="collect_query_logs")

    if not data["has_data"]:
        return "skip_reevaluation"

    reports = check_pipeline_drift(
        reference_confidences=np.array(data["reference_confidences"]),
        current_confidences=np.array(data["current_confidences"]),
        reference_latencies=np.array(data["reference_latencies"]),
        current_latencies=np.array(data["current_latencies"]),
    )

    for report in reports:
        logger.info(
            "Drift check [%s]: PSI=%.4f KS_p=%.4f drifted=%s",
            report.signal_name, report.psi, report.ks_pvalue, report.drifted,
        )

    drifted = any(r.drifted for r in reports)
    return "run_regression_suite" if drifted else "skip_reevaluation"


def run_regression_suite(**context) -> None:
    """Re-run the RAGAS evaluation suite and check the regression gate.

    Shells out to scripts/run_evaluation.py, which exits non-zero if
    the regression gate fails. A non-zero exit here fails this Airflow
    task, which (with retries/alerting configured) surfaces as a
    pipeline alert.
    """
    import subprocess

    result = subprocess.run(["python", "scripts/run_evaluation.py"], capture_output=True, text=True)

    logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError("Regression gate failed -- see logs for details")


if AIRFLOW_AVAILABLE:
    default_args = {
        "owner": "ml-platform",
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    }

    with DAG(
        dag_id="pharmadocs_drift_monitoring",
        description="Daily drift check with automated regression re-evaluation",
        schedule_interval="@daily",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        default_args=default_args,
        tags=["mlops", "rag", "drift-monitoring"],
    ) as dag:

        t1 = PythonOperator(task_id="collect_query_logs", python_callable=collect_query_logs)
        t2 = BranchPythonOperator(task_id="check_drift", python_callable=check_drift_task)
        t3 = PythonOperator(task_id="run_regression_suite", python_callable=run_regression_suite)
        t4 = EmptyOperator(task_id="skip_reevaluation")

        t1 >> t2 >> [t3, t4]
