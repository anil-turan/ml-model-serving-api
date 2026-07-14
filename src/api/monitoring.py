"""
Prometheus metrics and in-memory prediction window for drift detection.

Metrics exported at /metrics:
  - prediction_latency_seconds   (histogram)
  - prediction_score_distribution (histogram — tracks score drift over time)
  - prediction_requests_total    (counter, labelled by risk_grade and decision)
  - high_risk_rate_ratio         (gauge — rolling % of E-grade predictions)
  - prediction_score_psi         (gauge — PSI of the rolling window vs the
                                   reference test-set score distribution)

Drift alerting is PSI-based (see drift.py) rather than a fixed mean-score
threshold — this repo's model uses scale_pos_weight for class imbalance, so
predict_proba's mean (~0.42) isn't the true default rate (~8%), which made a
mean-threshold alert unreliable (see drift.py's module docstring).
"""
import collections
import statistics
import threading

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.api.drift import classify_psi, psi_from_reference

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

PSI_GAUGE = Gauge(
    "prediction_score_psi",
    "Population Stability Index of the rolling prediction window vs the reference test-set distribution",
)

# Minimum window size before PSI is meaningful (histogram bins would be too sparse below this).
_MIN_PSI_SAMPLE = 30

# Rolling window for lightweight drift detection (no external DB needed)
_window_lock = threading.Lock()
_score_window: collections.deque[float] = collections.deque(maxlen=1000)


def record_prediction(prob: float, grade: str, decision: str, latency: float) -> None:
    REQUEST_LATENCY.observe(latency)
    PREDICTION_SCORE.observe(prob)
    REQUESTS_TOTAL.labels(risk_grade=grade, decision=decision).inc()

    with _window_lock:
        _score_window.append(prob)
        window = list(_score_window)

    high_risk_rate = sum(1 for s in window if s >= 0.35) / len(window)
    HIGH_RISK_GAUGE.set(high_risk_rate)

    if len(window) >= _MIN_PSI_SAMPLE:
        PSI_GAUGE.set(psi_from_reference(window))


def get_drift_report(window_hours: int = 1) -> dict:
    """Return a PSI-based drift snapshot for the rolling window vs the reference distribution."""
    with _window_lock:
        scores = list(_score_window)

    if not scores:
        return {
            "score_mean": None,
            "score_std": None,
            "high_risk_rate": None,
            "sample_count": 0,
            "window_hours": window_hours,
            "psi": None,
            "psi_status": None,
            "alert": False,
            "alert_reason": None,
        }

    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    high_risk_rate = sum(1 for s in scores if s >= 0.35) / len(scores)

    if len(scores) >= _MIN_PSI_SAMPLE:
        psi_value = psi_from_reference(scores)
        psi_status = classify_psi(psi_value)
        alert = psi_status == "significant"
        alert_reason = f"PSI {psi_value:.4f} vs reference distribution ({psi_status} shift)" if alert else None
    else:
        psi_value, psi_status, alert, alert_reason = None, None, False, None

    return {
        "score_mean": round(mean, 4),
        "score_std": round(std, 4),
        "high_risk_rate": round(high_risk_rate, 4),
        "sample_count": len(scores),
        "window_hours": window_hours,
        "psi": round(psi_value, 4) if psi_value is not None else None,
        "psi_status": psi_status,
        "alert": alert,
        "alert_reason": alert_reason,
    }


def prometheus_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
