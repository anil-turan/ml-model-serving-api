# ML Model Serving API

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED)](https://www.docker.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-E6522C)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-F46800)](https://grafana.com/)
[![tests](https://img.shields.io/badge/coverage-92%25-brightgreen)](tests/)

Production-ready FastAPI service for credit default risk scoring. Deploys the LightGBM + Optuna model from [credit-risk-ml-pipeline](../credit-risk-ml-pipeline) behind a REST API with full observability.

**Model:** LightGBM + Optuna · ROC-AUC **0.754** · KS **37.9**  
**Stack:** FastAPI · Docker · Prometheus · Grafana · GitHub Actions · pytest (91% coverage)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client / Browser                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ POST /predict
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FastAPI (uvicorn)                             │
│                                                                   │
│  /predict  ──►  Pydantic validation                              │
│                 ──►  Feature engineering (_engineer_features)     │
│                      ──►  Selector → Preprocessor → LightGBM     │
│                           ──►  Risk grade (A–E) + decision        │
│                                                                   │
│  /health   ──►  Model metadata + uptime                          │
│  /drift    ──►  Rolling score distribution + drift alert          │
│  /metrics  ──►  Prometheus scrape endpoint                        │
└──────────────────────────┬───────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   Prometheus          Grafana          Model bundle
   (metrics)         (dashboard)     (best_model_bundle.pkl)
```

---

## Project Structure

```
ml-model-serving-api/
├── src/api/
│   ├── main.py          # FastAPI app + lifespan (model loading)
│   ├── schemas.py       # Pydantic request/response models
│   ├── predictor.py     # Model loading + inference pipeline
│   ├── monitoring.py    # Prometheus metrics + drift reporting
│   ├── drift.py         # From-scratch PSI (Siddiqi 2006) vs reference distribution
│   └── reference_score_distribution.json  # Binned reference scores (see scripts/)
├── scripts/
│   └── build_reference_distribution.py    # Regenerates the reference distribution above
├── tests/
│   ├── conftest.py      # Shared fixtures (mock bundle, TestClient)
│   ├── test_api.py      # Endpoint integration tests
│   ├── test_monitoring.py # Drift reporting unit tests
│   ├── test_drift.py    # PSI module unit tests
│   └── test_schemas.py  # Pydantic validation tests
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/         # Provisioned datasource + dashboard
├── .github/
│   └── workflows/ci.yml # Test → lint → Docker build
└── pyproject.toml
```

---

## Quickstart

**Option A — Local (no Docker)**

```bash
pip install -e ".[dev]"

# Point to the trained model bundle from Project 1
export MODEL_BUNDLE_PATH=../credit-risk-ml-pipeline/outputs/best_model_bundle.pkl

uvicorn src.api.main:app --reload
# → http://localhost:8000/docs
```

**Option B — Docker Compose (API + Prometheus + Grafana)**

```bash
cd docker
docker-compose up --build
```

| Service    | URL                          |
|------------|------------------------------|
| API docs   | http://localhost:8000/docs   |
| Prometheus | http://localhost:9090        |
| Grafana    | http://localhost:3000 (admin/admin) |

---

## API Reference

### `GET /health`

```json
{
  "status": "ok",
  "model_version": "lgb-optuna-auc0.754",
  "test_roc_auc": 0.754,
  "uptime_seconds": 142.3
}
```

### `POST /predict`

**Request:**
```json
{
  "AMT_INCOME_TOTAL": 90000,
  "AMT_CREDIT": 450000,
  "AMT_ANNUITY": 22500,
  "AMT_GOODS_PRICE": 400000,
  "DAYS_BIRTH": -9000,
  "DAYS_EMPLOYED": -180,
  "CNT_FAM_MEMBERS": 1,
  "NAME_EDUCATION_TYPE": "Secondary / secondary special",
  "NAME_INCOME_TYPE": "Working",
  "ORGANIZATION_TYPE": "Business Entity Type 3",
  "EXT_SOURCE_1": 0.25,
  "EXT_SOURCE_2": 0.30,
  "EXT_SOURCE_3": 0.20
}
```

**Response:**
```json
{
  "default_probability": 0.3821,
  "risk_grade": "D",
  "decision": "Review",
  "model_version": "lgb-optuna-auc0.754",
  "prediction_id": "f3a2c1d0-...",
  "timestamp": "2026-06-27T10:30:00.000Z"
}
```

**Risk grades:**

| Grade | Probability | Decision |
|-------|-------------|----------|
| A | < 5% | Approve |
| B | 5–10% | Approve |
| C | 10–20% | Review |
| D | 20–35% | Review |
| E | ≥ 35% | Decline |

### `GET /drift`

Rolling score distribution over the last 1000 predictions, with a **PSI-based** drift alert (Population Stability Index vs. a reference distribution reproduced from the model's held-out test set — see [Monitoring](#monitoring) below).

```json
{
  "score_mean": 0.4187,
  "score_std": 0.1932,
  "high_risk_rate": 0.301,
  "sample_count": 247,
  "window_hours": 1,
  "psi": 0.0421,
  "psi_status": "stable",
  "alert": false,
  "alert_reason": null
}
```

`psi_status` is `stable` (< 0.10), `moderate` (0.10–0.25), or `significant` (≥ 0.25 — `alert: true`). `psi`/`psi_status` are `null` until the rolling window has at least 30 samples.

### `GET /metrics`

Prometheus text format. Scraped at `/metrics`.

**Key metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `prediction_latency_seconds` | Histogram | End-to-end /predict latency |
| `prediction_score` | Histogram | Default probability distribution |
| `prediction_requests_total` | Counter | Request count by risk_grade + decision |
| `high_risk_rate_ratio` | Gauge | Rolling % of Decline-grade (E) predictions |
| `prediction_score_psi` | Gauge | PSI of the rolling window vs. the reference test-set score distribution |

---

## Testing

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

**Coverage: 92%** (26 tests) — all endpoints, PSI drift logic, and schema validation tested with a mock bundle (no pkl file needed to run tests).

| Module | Coverage |
|--------|----------|
| `schemas.py` | 100% |
| `monitoring.py` | 100% |
| `drift.py` | 100% |
| `predictor.py` | 90% |
| `main.py` | 78% |

---

## CI/CD

GitHub Actions pipeline (`.github/workflows/ci.yml`):

```
push to main / PR
      │
      ▼
  ┌─────────────┐
  │  pytest     │  --cov-fail-under=85
  │  ruff lint  │
  │  mypy check │
  └──────┬──────┘
         │ on main only
         ▼
  ┌─────────────┐
  │Docker build │  validates Dockerfile + dependencies
  └─────────────┘
```

---

## Monitoring

**Grafana dashboard** (auto-provisioned at startup):

- Request rate (req/min)
- P50 / P95 / P99 latency
- Score distribution over time
- Predictions by risk grade (pie chart)
- High-risk rate gauge with threshold alert at 30%

**Drift detection logic (PSI-based, deepened 2026-07-14):**
- Maintains a rolling window of the last 1,000 predictions (in-memory, thread-safe)
- Compares the window against a **reference score distribution** reproduced from the
  credit-risk-ml-pipeline test split (`scripts/build_reference_distribution.py` — same
  `train_test_split(test_size=0.20, random_state=42, stratify=TARGET)` used for that
  project's fairness audit, scored through the deployed bundle; AUC reproduces to 0.7540,
  matching the bundle's stored 0.754)
- PSI (Siddiqi 2006, from scratch — same implementation/alarm bands as the
  customer-churn-prediction and ato-detection-lstm projects): **< 0.10 stable, 0.10–0.25
  moderate, ≥ 0.25 significant** → `alert: true`
- Exposed via `/drift` (JSON) and `/metrics` (`prediction_score_psi` gauge, for
  Prometheus/Grafana alerting) — see [`src/api/drift.py`](src/api/drift.py)

**Why PSI, not a mean-score threshold:** the original version of this drift check alerted
when the rolling mean exceeded a fixed 0.18. Reproducing the reference distribution
surfaced why that was unreliable — see **Design Decisions** below.

Tests: `pytest tests/test_drift.py tests/test_monitoring.py` — 11 tests, all passing.

---

## Design Decisions

**Why no database?** The rolling window uses a thread-safe `collections.deque(maxlen=1000)` — sufficient for demonstrating monitoring patterns without requiring a running Postgres or Redis instance. In production, swap for a time-series store (InfluxDB, TimescaleDB).

**Why no MLflow?** MLflow model registry is additive infrastructure. The model bundle pkl is the canonical artefact for this project. An `mlflow ui` integration would add another service with no learning return for this scope.

**Why `monkeypatch` in tests not `unittest.mock.patch` context managers?** Context managers in pytest fixtures exit before the test body runs, leaving the mock inactive. `monkeypatch` from pytest stays active for the full test lifetime.

**Why the model's predicted probability isn't the true default rate:** reproducing the
deployed bundle's test-set predictions (for the PSI reference distribution, see Monitoring
above) turned up a real finding — the reference mean predicted probability is **~42%**, far
above the **~8%** true default rate in the same test set, even though the model's AUC (0.754)
reproduces exactly. The cause: the bundle's LightGBM was trained with
`scale_pos_weight=11.39` to improve ranking on this imbalanced problem, which is a legitimate
technique for AUC but destroys probability calibration — `predict_proba` should be read as a
**risk-ranking score**, not a literal probability. This is why the risk-grade thresholds (5%,
10%, 20%, 35%) are themselves uncalibrated cutoffs tuned on the raw score, and why the old
mean-threshold drift check (alerting above 0.18) was unreliable: 0.18 sits *below* the
reference mean of 0.42, so it would have fired continuously even with zero drift. PSI-based
drift detection compares the distribution's *shape* over time instead, which isn't affected
by this miscalibration.

---

## AI Tools Used

Built with [Claude Code](https://claude.com/claude-code):
- API structure, Pydantic schemas, and Prometheus metrics designed with senior DS review
- Test fixtures designed to avoid the pytest fixture / mock lifetime pitfall
- All code reviewed and verified manually before commit

---

## License

MIT — Copyright (c) 2026 Anil Turan
