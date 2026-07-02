"""
Model loading and inference.

Loads best_model_bundle.pkl from the credit-risk-ml-pipeline project.
The bundle contains the full inference pipeline:
  selector → ColumnTransformer (StandardScaler + OHE) → LightGBM
"""
import datetime
import os
import pickle
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.schemas import LoanApplicationRequest, PredictionResponse

# Path to the trained model bundle from Project 1
_BUNDLE_PATH = Path(
    os.getenv(
        "MODEL_BUNDLE_PATH",
        str(Path(__file__).resolve().parents[3] / "credit-risk-ml-pipeline" / "outputs" / "best_model_bundle.pkl"),
    )
)

_GRADE_THRESHOLDS = [
    (0.05, "A", "Approve"),
    (0.10, "B", "Approve"),
    (0.20, "C", "Review"),
    (0.35, "D", "Review"),
    (1.01, "E", "Decline"),
]

_bundle: dict | None = None


def load_bundle() -> dict:
    global _bundle
    if _bundle is None:
        if not _BUNDLE_PATH.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {_BUNDLE_PATH}. "
                "Run notebooks/03_modeling.ipynb in the credit-risk-ml-pipeline project first."
            )
        with open(_BUNDLE_PATH, "rb") as f:
            _bundle = pickle.load(f)
    return _bundle


def _grade(prob: float) -> tuple[str, str]:
    for threshold, grade, decision in _GRADE_THRESHOLDS:
        if prob < threshold:
            return grade, decision
    return "E", "Decline"


def predict(request: LoanApplicationRequest) -> PredictionResponse:
    bundle = load_bundle()

    df = pd.DataFrame([request.model_dump()])

    # Apply domain feature engineering (same transformations as training)
    df = _engineer_features(df)

    df_sel = bundle["selector"].transform(df)
    df_proc = bundle["preprocessor"].transform(df_sel)
    prob = float(bundle["model"].predict_proba(df_proc)[0, 1])

    grade, decision = _grade(prob)
    auc = bundle.get("test_roc_auc", "n/a")

    return PredictionResponse(
        default_probability=round(prob, 4),
        risk_grade=grade,
        decision=decision,
        model_version=f"lgb-optuna-auc{auc}",
        prediction_id=str(uuid.uuid4()),
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
    )


def get_bundle_meta() -> dict:
    bundle = load_bundle()
    return {
        "model_version": f"lgb-optuna-auc{bundle.get('test_roc_auc', 'n/a')}",
        "test_roc_auc": bundle.get("test_roc_auc"),
        "test_pr_auc": bundle.get("test_pr_auc"),
    }


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the feature engineering from CreditRiskFeatureEngineer in Project 1."""
    df = df.copy()
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYMENT_YEARS"] = (-df["DAYS_EMPLOYED"]).clip(upper=50 * 365) / 365.25
    df["EMPLOYMENT_AGE_RATIO"] = df["EMPLOYMENT_YEARS"] / (df["AGE_YEARS"] + 1e-6)
    df["DTI_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1e-6)
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1e-6)
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1e-6)
    df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1e-6)
    df["INCOME_PER_FAMILY"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1e-6)
    df["AMT_ANNUITY_LOG"] = np.log1p(df["AMT_ANNUITY"])
    bins = [0, 25, 35, 45, 55, 65, 200]
    labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
    df["AGE_GROUP"] = pd.cut(df["AGE_YEARS"], bins=bins, labels=labels, right=False).astype(str)
    return df
