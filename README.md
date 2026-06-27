# ML Model Serving API

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
│   └── monitoring.py    # Prometheus metrics + drift detection
├── tests/
│   ├── conftest.py      # Shared fixtures (mock bundle, TestClient)
│   ├── test_api.py      # Endpoint integration tests
│   ├── test_monitoring.py # Drift detection unit tests
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

Rolling score distribution over the last 1000 predictions. Raises an alert if mean score exceeds 18% (2 SD above the 8.1% population default rate).

```json
{
  "score_mean": 0.1124,
  "score_std": 0.0873,
  "high_risk_rate": 0.082,
  "sample_count": 247,
  "window_hours": 1,
  "alert": false,
  "alert_reason": null
}
```

### `GET /metrics`

Prometheus text format. Scraped at `/metrics`.

**Key metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `prediction_latency_seconds` | Histogram | End-to-end /predict latency |
| `prediction_score` | Histogram | Default probability distribution |
| `prediction_requests_total` | Counter | Request count by risk_grade + decision |
| `high_risk_rate_ratio` | Gauge | Rolling % of Decline-grade (E) predictions |

---

## Testing

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

**Coverage: 91%** — all endpoints, drift logic, and schema validation tested with a mock bundle (no pkl file needed to run tests).

| Module | Coverage |
|--------|----------|
| `schemas.py` | 100% |
| `monitoring.py` | 100% |
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

**Drift detection logic:**
- Maintains a rolling window of the last 1,000 predictions (in-memory, thread-safe)
- Alerts when mean default probability > 18% (2 SD above the 8.1% training distribution)
- Exposed via `/drift` for polling or external alertmanager integration

---

## Design Decisions

**Why no database?** The rolling window uses a thread-safe `collections.deque(maxlen=1000)` — sufficient for demonstrating monitoring patterns without requiring a running Postgres or Redis instance. In production, swap for a time-series store (InfluxDB, TimescaleDB).

**Why no MLflow?** MLflow model registry is additive infrastructure. The model bundle pkl is the canonical artefact for this project. An `mlflow ui` integration would add another service with no learning return for this scope.

**Why `monkeypatch` in tests not `unittest.mock.patch` context managers?** Context managers in pytest fixtures exit before the test body runs, leaving the mock inactive. `monkeypatch` from pytest stays active for the full test lifetime.

---

## AI Tools Used

Built with [Claude Code](https://claude.com/claude-code):
- API structure, Pydantic schemas, and Prometheus metrics designed with senior DS review
- Test fixtures designed to avoid the pytest fixture / mock lifetime pitfall
- All code reviewed and verified manually before commit

---

## License

MIT — Copyright (c) 2026 Anil Turan
