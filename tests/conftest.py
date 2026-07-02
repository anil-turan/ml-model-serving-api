"""Shared fixtures for all tests."""
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

SAMPLE_APPLICATION = {
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
    "EXT_SOURCE_3": 0.20,
}

LOW_RISK_APPLICATION = {
    "AMT_INCOME_TOTAL": 200000,
    "AMT_CREDIT": 180000,
    "AMT_ANNUITY": 9000,
    "AMT_GOODS_PRICE": 160000,
    "DAYS_BIRTH": -18000,
    "DAYS_EMPLOYED": -3650,
    "CNT_FAM_MEMBERS": 3,
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_INCOME_TYPE": "Working",
    "ORGANIZATION_TYPE": "Government",
    "EXT_SOURCE_1": 0.85,
    "EXT_SOURCE_2": 0.90,
    "EXT_SOURCE_3": 0.80,
}


def _make_mock_bundle():
    """Build a minimal fake bundle that mimics the real pkl structure."""
    selector = MagicMock()
    selector.transform.side_effect = lambda df: df  # pass through

    preprocessor = MagicMock()
    preprocessor.transform.side_effect = lambda df: np.zeros((len(df), 52))

    model = MagicMock()
    # Return high-risk proba for the default sample
    model.predict_proba.return_value = np.array([[0.65, 0.35]])

    return {
        "selector": selector,
        "preprocessor": preprocessor,
        "model": model,
        "num_cols": [],
        "cat_cols": [],
        "best_params": {},
        "test_roc_auc": 0.754,
        "test_pr_auc": 0.243,
    }


@pytest.fixture
def mock_bundle():
    return _make_mock_bundle()


@pytest.fixture
def client(mock_bundle, monkeypatch):
    """TestClient with the model bundle mocked out so no pkl file is needed.

    monkeypatch keeps the patch alive for the full duration of the test,
    unlike a context manager which exits after the fixture returns.
    """
    import src.api.main as main_module
    import src.api.predictor as pred_module

    monkeypatch.setattr(pred_module, "_bundle", mock_bundle)
    monkeypatch.setattr(pred_module, "_BUNDLE_PATH", Path("/fake/bundle.pkl"))
    monkeypatch.setattr(main_module, "_model_ready", True)

    return TestClient(main_module.app)
