"""Tests for the drift detection and Prometheus metrics logic."""
import src.api.monitoring as mon


def _reset_window():
    with mon._window_lock:
        mon._score_window.clear()


def test_empty_window_returns_no_alert():
    _reset_window()
    report = mon.get_drift_report()
    assert report["alert"] is False
    assert report["sample_count"] == 0


def test_record_prediction_populates_window():
    _reset_window()
    mon.record_prediction(prob=0.45, grade="E", decision="Decline", latency=0.01)
    report = mon.get_drift_report()
    assert report["sample_count"] == 1
    assert report["score_mean"] == 0.45


def test_drift_alert_triggered_on_high_scores():
    _reset_window()
    for _ in range(50):
        mon.record_prediction(prob=0.50, grade="E", decision="Decline", latency=0.01)
    report = mon.get_drift_report()
    assert report["alert"] is True
    assert report["alert_reason"] is not None


def test_no_alert_on_normal_scores():
    _reset_window()
    for _ in range(50):
        mon.record_prediction(prob=0.05, grade="A", decision="Approve", latency=0.01)
    report = mon.get_drift_report()
    assert report["alert"] is False


def test_high_risk_rate_calculation():
    _reset_window()
    # 30 high-risk (>=0.35), 70 low-risk
    for _ in range(30):
        mon.record_prediction(prob=0.40, grade="E", decision="Decline", latency=0.01)
    for _ in range(70):
        mon.record_prediction(prob=0.05, grade="A", decision="Approve", latency=0.01)
    report = mon.get_drift_report()
    assert abs(report["high_risk_rate"] - 0.30) < 0.02
