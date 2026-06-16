"""
Drift detection for the RAG pipeline using PSI and the KS-test.

In a production RAG system, "drift" shows up as shifts in:
  - the distribution of retrieval confidence scores (are queries
    becoming harder to answer from the existing knowledge base?)
  - the distribution of response latencies (is the index or LLM
    backend degrading?)

This module computes the Population Stability Index (PSI) and the
Kolmogorov-Smirnov (KS) test statistic between a reference window
(e.g. last week's queries) and a current window (e.g. today's
queries) for both signals. When either statistic exceeds a configured
threshold, `check_drift` flags the corresponding signal as drifted.

This corresponds to the "drift detection via PSI and KS-test
triggers automated retraining" claim in the project README. In this
project, "retraining" maps to triggering a re-evaluation of the
RAGAS regression suite and, if needed, a re-index of the document
corpus (e.g. with updated embeddings).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp


@dataclass
class DriftReport:
    """Result of a drift check for a single signal (e.g. "confidence")."""

    signal_name: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    drifted: bool


def population_stability_index(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Compute the Population Stability Index between two distributions.

    PSI quantifies how much a distribution has shifted between a
    reference and current sample, using a common binning scheme
    derived from the reference distribution's quantiles.

    Interpretation (standard industry thresholds):
        PSI < 0.1  -> no significant shift
        0.1 <= PSI < 0.25 -> moderate shift, worth monitoring
        PSI >= 0.25 -> significant shift, investigate
    """
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)

    # quantile-based bin edges from the reference distribution
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(np.quantile(reference, quantiles))

    if len(bin_edges) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    # avoid division by zero / log(0) with a small epsilon
    eps = 1e-4
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return round(float(psi), 4)


def check_drift(
    reference: np.ndarray,
    current: np.ndarray,
    signal_name: str,
    psi_threshold: float = 0.25,
    ks_pvalue_threshold: float = 0.05,
) -> DriftReport:
    """Run PSI and KS-test for a single signal and determine drift status.

    A signal is flagged as drifted if EITHER:
      - PSI >= psi_threshold, OR
      - the KS-test p-value < ks_pvalue_threshold (distributions
        significantly different)
    """
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)

    psi = population_stability_index(reference, current)
    ks_stat, ks_pvalue = ks_2samp(reference, current)

    drifted = bool((psi >= psi_threshold) or (ks_pvalue < ks_pvalue_threshold))

    return DriftReport(
        signal_name=signal_name,
        psi=psi,
        ks_statistic=round(float(ks_stat), 4),
        ks_pvalue=round(float(ks_pvalue), 4),
        drifted=drifted,
    )


def check_pipeline_drift(
    reference_confidences: np.ndarray,
    current_confidences: np.ndarray,
    reference_latencies: np.ndarray,
    current_latencies: np.ndarray,
) -> list[DriftReport]:
    """Run drift checks across the two monitored RAG pipeline signals.

    Returns a list of DriftReport, one per signal ("confidence" and
    "latency_ms"). Downstream, if any report has `drifted=True`, an
    Airflow DAG (see `dags/drift_monitoring_dag.py`) triggers a
    re-evaluation of the RAGAS regression suite.
    """
    return [
        check_drift(reference_confidences, current_confidences, signal_name="confidence"),
        check_drift(reference_latencies, current_latencies, signal_name="latency_ms", psi_threshold=0.25),
    ]
