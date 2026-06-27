"""Pydantic schema validation tests."""
import pytest
from pydantic import ValidationError

from src.api.schemas import LoanApplicationRequest

BASE = {
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
}


def test_valid_application():
    app = LoanApplicationRequest(**BASE)
    assert app.EXT_SOURCE_1 == 0.5  # default


def test_negative_income_rejected():
    with pytest.raises(ValidationError):
        LoanApplicationRequest(**{**BASE, "AMT_INCOME_TOTAL": -1})


def test_positive_days_birth_rejected():
    with pytest.raises(ValidationError):
        LoanApplicationRequest(**{**BASE, "DAYS_BIRTH": 100})


def test_credit_income_ratio_too_high():
    with pytest.raises(ValidationError):
        LoanApplicationRequest(**{**BASE, "AMT_CREDIT": 90000 * 25})


def test_ext_source_out_of_range():
    with pytest.raises(ValidationError):
        LoanApplicationRequest(**{**BASE, "EXT_SOURCE_1": 1.5})


def test_default_occupation_type():
    app = LoanApplicationRequest(**BASE)
    assert app.OCCUPATION_TYPE == "Unknown"
