"""
Prometheus metrics and in-memory prediction window for drift detection.

Metrics exported at /metrics:
  - prediction_latency_seconds   (histogram)
  - prediction_score_distribution (histogram — tracks score drift over time)
  - prediction_requests_total    (counter, labelled by risk_grade and decision)
  - high_risk_rate_ratio         (gauge — rolling % of E-grade predictions)
"""
import time
import collections
import threading
from typing import Deque

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "End-to-end latency of /predict requests",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

PREDICTION_SCORE = Histogram(
    "prediction_score",
    "Distribution of default probability scores",
    buckets=[0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0],
)

REQUESTS_TOTAL = Counter(
    "prediction_requests_total",
    "Total prediction requests by risk grade and decision",
    ["risk_grade", "decision"],
)

HIGH_RISK_GAUGE = Gauge(
    "high_risk_rate_ratio",
    "Rolling proportion of Decline-grade (E) predictions over last 1000 requests",
)

# Rolling window for lightweight drift detection (no external DB needed)
_window_lock = threading.Lock()
_score_window: Deque[float] = collections.deque(maxlen=1000)


def record_prediction(prob: float, grade: str, decision: str, latency: float) -> None:
    REQUEST_LATENCY.observe(latency)
    PREDICTION_SCORE.observe(prob)
    REQUESTS_TOTAL.labels(risk_grade=grade, decision=decision).inc()

    with _window_lock:
        _score_window.append(prob)
        if _score_window:
            high_risk_rate = sum(1 for s in _score_window if s >= 0.35) / len(_score_window)
            HIGH_RISK_GAUGE.set(high_risk_rate)


def get_drift_report(window_hours: int = 1) -> dict:
    """Return a lightweight score-distribution snapshot for the rolling window."""
    with _window_lock:
        scores = list(_score_window)

    if not scores:
        return {
            "score_mean": None,
            "score_std": None,
            "high_risk_rate": None,
            "sample_count": 0,
            "window_hours": window_hours,
            "alert": False,
            "alert_reason": None,
        }

    import statistics
    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    high_risk_rate = sum(1 for s in scores if s >= 0.35) / len(scores)

    # Simple drift alert: mean score > 2 standard deviations above 0.08 (population default rate)
    EXPECTED_MEAN = 0.08
    alert = mean > EXPECTED_MEAN + 2 * 0.05  # flag if mean default prob > 18%
    alert_reason = f"Mean score {mean:.3f} exceeds alert threshold 0.18" if alert else None

    return {
        "score_mean": round(mean, 4),
        "score_std": round(std, 4),
        "high_risk_rate": round(high_risk_rate, 4),
        "sample_count": len(scores),
        "window_hours": window_hours,
        "alert": alert,
        "alert_reason": alert_reason,
    }


def prometheus_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
