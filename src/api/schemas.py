from pydantic import BaseModel, Field, model_validator
from typing import Optional
import datetime


class LoanApplicationRequest(BaseModel):
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Annual income in local currency")
    AMT_CREDIT: float = Field(..., gt=0, description="Loan amount requested")
    AMT_ANNUITY: float = Field(..., gt=0, description="Monthly repayment amount")
    AMT_GOODS_PRICE: float = Field(..., gt=0, description="Price of goods the loan is for")
    DAYS_BIRTH: int = Field(..., lt=0, description="Days since birth (negative integer)")
    DAYS_EMPLOYED: int = Field(..., description="Days since employment start (negative = employed, 365243 = unemployed)")
    CNT_FAM_MEMBERS: float = Field(..., ge=1, description="Number of family members")
    NAME_EDUCATION_TYPE: str = Field(..., description="Highest education level achieved")
    NAME_INCOME_TYPE: str = Field(..., description="Primary income source type")
    ORGANIZATION_TYPE: str = Field(..., description="Type of employer organisation")
    OCCUPATION_TYPE: str = Field(default="Unknown", description="Applicant occupation type")
    EXT_SOURCE_1: float = Field(default=0.5, ge=0, le=1, description="External credit score 1 (0–1)")
    EXT_SOURCE_2: float = Field(default=0.5, ge=0, le=1, description="External credit score 2 (0–1)")
    EXT_SOURCE_3: float = Field(default=0.5, ge=0, le=1, description="External credit score 3 (0–1)")

    @model_validator(mode="after")
    def check_credit_income_ratio(self) -> "LoanApplicationRequest":
        ratio = self.AMT_CREDIT / self.AMT_INCOME_TOTAL
        if ratio > 20:
            raise ValueError(
                f"AMT_CREDIT / AMT_INCOME_TOTAL = {ratio:.1f} exceeds plausible limit of 20"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    default_probability: float = Field(..., description="Predicted probability of loan default (0–1)")
    risk_grade: str = Field(..., description="Risk grade A–E")
    decision: str = Field(..., description="Lending decision: Approve / Review / Decline")
    model_version: str = Field(..., description="Model identifier used for this prediction")
    prediction_id: str = Field(..., description="Unique ID for this prediction (for audit trail)")
    timestamp: str = Field(..., description="UTC timestamp of the prediction")


class HealthResponse(BaseModel):
    status: str
    model_version: str
    test_roc_auc: Optional[float]
    uptime_seconds: float


class DriftReport(BaseModel):
    score_mean: Optional[float]
    score_std: Optional[float]
    high_risk_rate: Optional[float]
    sample_count: int
    window_hours: int
    alert: bool
    alert_reason: Optional[str]
