"""Tests for the PSI-based drift detection and Prometheus metrics logic."""
import src.api.monitoring as mon
from src.api.drift import load_reference


def _reset_window():
    with mon._window_lock:
        mon._score_window.clear()


def test_empty_window_returns_no_alert():
    _reset_window()
    report = mon.get_drift_report()
    assert report["alert"] is False
    assert report["sample_count"] == 0
    assert report["psi"] is None


def test_record_prediction_populates_window():
    _reset_window()
    mon.record_prediction(prob=0.45, grade="E", decision="Decline", latency=0.01)
    report = mon.get_drift_report()
    assert report["sample_count"] == 1
    assert report["score_mean"] == 0.45


def test_below_min_sample_has_no_psi():
    _reset_window()
    for _ in range(10):  # below _MIN_PSI_SAMPLE
        mon.record_prediction(prob=0.5, grade="E", decision="Decline", latency=0.01)
    report = mon.get_drift_report()
    assert report["psi"] is None
    assert report["alert"] is False


def test_sampling_from_reference_distribution_gives_low_psi():
    # Feed a window drawn from the reference distribution's own shape
    # (reproduced mean/std) — PSI against itself should be small.
    _reset_window()
    reference = load_reference()
    mean, std = reference["reference_mean"], reference["reference_std"]
    import numpy as np

    rng = np.random.default_rng(0)
    scores = np.clip(rng.normal(mean, std, 200), 0, 1)
    for s in scores:
        mon.record_prediction(prob=float(s), grade="C", decision="Review", latency=0.01)
    report = mon.get_drift_report()
    assert report["psi"] is not None
    assert report["psi_status"] in {"stable", "moderate"}


def test_drift_alert_triggered_on_shifted_distribution():
    # A window concentrated at the low end is a clear shift away from the
    # reference distribution (reference mean ~0.42) -> significant PSI.
    _reset_window()
    for _ in range(200):
        mon.record_prediction(prob=0.01, grade="A", decision="Approve", latency=0.01)
    report = mon.get_drift_report()
    assert report["psi"] is not None
    assert report["psi_status"] == "significant"
    assert report["alert"] is True
    assert "PSI" in report["alert_reason"]


def test_high_risk_rate_calculation():
    _reset_window()
    # 30 high-risk (>=0.35), 70 low-risk
    for _ in range(30):
        mon.record_prediction(prob=0.40, grade="E", decision="Decline", latency=0.01)
    for _ in range(70):
        mon.record_prediction(prob=0.05, grade="A", decision="Approve", latency=0.01)
    report = mon.get_drift_report()
    assert abs(report["high_risk_rate"] - 0.30) < 0.02
