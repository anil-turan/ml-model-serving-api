"""Rebuild the reference score distribution used by PSI-based drift detection.

Reproduces the exact held-out test split used for the fairness audit in
credit-risk-ml-pipeline/notebooks/06_fairness.ipynb (train_test_split,
test_size=0.20, random_state=42, stratified on TARGET), scores it through the
deployed bundle, and saves a binned summary (not raw scores) to
src/api/reference_score_distribution.json.

Run this again only if the deployed model bundle changes.

Usage:
    python scripts/build_reference_distribution.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PIPELINE_REPO = Path(__file__).resolve().parents[2] / "credit-risk-ml-pipeline"
sys.path.insert(0, str(PIPELINE_REPO))  # unpickling the bundle needs src.credit_risk importable

BUNDLE_PATH = PIPELINE_REPO / "outputs" / "best_model_bundle.pkl"
FEATURES_PATH = PIPELINE_REPO / "data" / "processed" / "features_engineered.csv"
OUT_PATH = Path(__file__).resolve().parents[1] / "src" / "api" / "reference_score_distribution.json"

# Same bucket edges as PREDICTION_SCORE in src/api/monitoring.py, for consistent PSI binning.
BIN_EDGES = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0]


def main() -> None:
    bundle = pickle.load(open(BUNDLE_PATH, "rb"))
    selector, preprocessor, model = bundle["selector"], bundle["preprocessor"], bundle["model"]

    df = pd.read_csv(FEATURES_PATH)
    X, y = df.drop(columns=["TARGET"]), df["TARGET"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    X_test_proc = preprocessor.transform(selector.transform(X_test))
    proba = model.predict_proba(X_test_proc)[:, 1]

    reproduced_auc = roc_auc_score(y_test, proba)
    print(f"Bundle stored test AUC: {bundle['test_roc_auc']}")
    print(f"Reproduced test AUC:    {reproduced_auc:.4f}")
    print(f"n_test: {len(proba)}  mean: {proba.mean():.4f}  std: {proba.std():.4f}")
    print(f"True default rate (y_test): {y_test.mean():.4f}")

    counts, _ = np.histogram(proba, bins=BIN_EDGES)
    proportions = (counts / counts.sum()).tolist()

    payload = {
        "source": "credit-risk-ml-pipeline train_test_split(test_size=0.20, random_state=42, stratify=TARGET), scored through best_model_bundle.pkl",
        "bin_edges": BIN_EDGES,
        "bin_proportions": proportions,
        "n_reference": int(len(proba)),
        "reference_mean": float(proba.mean()),
        "reference_std": float(proba.std()),
        "true_default_rate": float(y_test.mean()),
        "note": (
            "reference_mean is far above true_default_rate because the deployed model was "
            "trained with scale_pos_weight=11.39 to improve ranking (AUC) on this imbalanced "
            "problem — predict_proba is NOT a calibrated probability. Drift detection below "
            "compares against this reference distribution's own shape, not against the true "
            "default rate, so this miscalibration doesn't distort the PSI calculation."
        ),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
