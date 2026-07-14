"""PSI-based score drift detection, comparing live predictions against a
fixed reference distribution reproduced from the model's held-out test set.

Population Stability Index (PSI), Siddiqi (2006) — same from-scratch
implementation and alarm bands used elsewhere in this portfolio
(customer-churn-prediction, ato-detection-lstm): bins both distributions on
shared edges and sums a symmetric, weighted log-ratio of bin proportions.

    PSI < 0.10            stable
    0.10 <= PSI < 0.25    moderate shift
    PSI >= 0.25           significant shift — investigate / consider retraining

The reference distribution here is the *predicted score* distribution on the
credit-risk-ml-pipeline test split (see scripts/build_reference_distribution.py),
not the true default rate. The deployed LightGBM model was trained with
scale_pos_weight=11.39 to improve ranking on this imbalanced problem, so
predict_proba is not a calibrated probability (reference mean ~0.42 vs a true
default rate of ~8%). PSI compares the *shape* of the distribution against
itself over time, so this miscalibration doesn't affect the drift signal —
but it does mean "mean predicted score" alone is a misleading health metric,
which is why the previous version of this module's alert logic (mean > 0.18)
was unreliable: 0.18 sits *below* the reference mean, so it would fire under
zero drift. See scripts/build_reference_distribution.py's output for the numbers.
"""

import json
from pathlib import Path

import numpy as np

_EPS = 1e-6  # avoids log(0)/div-by-0 when a bin is empty in one distribution

PSI_STABLE_MAX = 0.10
PSI_MODERATE_MAX = 0.25

_REFERENCE_PATH = Path(__file__).parent / "reference_score_distribution.json"
_reference: dict | None = None


def load_reference() -> dict:
    global _reference
    if _reference is None:
        _reference = json.loads(_REFERENCE_PATH.read_text())
    return _reference


def psi_from_reference(current_scores: list[float] | np.ndarray) -> float:
    """PSI between the live score window and the stored reference distribution."""
    reference = load_reference()
    edges = np.asarray(reference["bin_edges"], dtype=float)
    ref_pct = np.asarray(reference["bin_proportions"], dtype=float) + _EPS

    current = np.asarray(current_scores, dtype=float)
    cur_counts, _ = np.histogram(current, bins=edges)
    cur_pct = cur_counts / cur_counts.sum() + _EPS

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def classify_psi(psi_value: float) -> str:
    if psi_value < PSI_STABLE_MAX:
        return "stable"
    if psi_value < PSI_MODERATE_MAX:
        return "moderate"
    return "significant"
