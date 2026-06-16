import numpy as np

from app.monitoring.drift_detection import check_drift, check_pipeline_drift, population_stability_index


def test_psi_near_zero_for_identical_distributions():
    rng = np.random.default_rng(42)
    reference = rng.normal(0.7, 0.1, 1000)
    current = reference.copy()

    psi = population_stability_index(reference, current)

    assert psi < 0.01


def test_psi_high_for_shifted_distributions():
    rng = np.random.default_rng(42)
    reference = rng.normal(0.7, 0.1, 1000)
    current = rng.normal(0.3, 0.1, 1000)

    psi = population_stability_index(reference, current)

    assert psi > 0.25


def test_check_drift_flags_shifted_distribution():
    rng = np.random.default_rng(42)
    reference = rng.normal(0.7, 0.05, 500)
    current = rng.normal(0.3, 0.05, 500)

    report = check_drift(reference, current, signal_name="confidence")

    assert report.drifted is True
    assert report.signal_name == "confidence"


def test_check_drift_does_not_flag_stable_distribution():
    rng = np.random.default_rng(1)
    reference = rng.normal(50, 5, 500)
    current = rng.normal(50, 5, 500)

    report = check_drift(reference, current, signal_name="latency_ms")

    assert bool(report.drifted) is False


def test_check_pipeline_drift_returns_two_reports():
    rng = np.random.default_rng(1)
    ref_conf = rng.normal(0.7, 0.1, 200)
    cur_conf = rng.normal(0.7, 0.1, 200)
    ref_lat = rng.normal(50, 5, 200)
    cur_lat = rng.normal(50, 5, 200)

    reports = check_pipeline_drift(ref_conf, cur_conf, ref_lat, cur_lat)

    assert len(reports) == 2
    assert {r.signal_name for r in reports} == {"confidence", "latency_ms"}
