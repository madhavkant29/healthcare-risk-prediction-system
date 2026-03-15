from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
from datetime import datetime


# ── Auth ────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    username: str
    password: str


# ── Patient ──────────────────────────────────────────────

class PatientCreate(BaseModel):
    patient_ref: str


class PatientOut(BaseModel):
    id: int
    patient_ref: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Prediction ───────────────────────────────────────────

class SHAPFeature(BaseModel):
    feature: str
    shap_value: float


class SHAPExplanation(BaseModel):
    predicted_class: str
    base_value: float
    top_features: list[SHAPFeature]


class PredictionRequest(BaseModel):
    features: dict  # raw 163-feature dict from frontend/CSV
    patient_ref: Optional[str] = None  # optional — for linking to patient record

    @field_validator("features")
    @classmethod
    def features_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("Features dict cannot be empty")
        return v


class PredictionResult(BaseModel):
    predicted_class: str                  # "NO", "<30", ">30"
    confidence: float
    probabilities: dict[str, float]       # {"NO": 0.x, "<30": 0.x, ">30": 0.x}
    shap_explanation: Optional[SHAPExplanation] = None
    prediction_id: Optional[int] = None


class BatchPredictionRequest(BaseModel):
    records: list[dict]                   # list of feature dicts


class BatchPredictionResult(BaseModel):
    total: int
    results: list[PredictionResult]


class PredictionHistory(BaseModel):
    id: int
    predicted_class: str
    confidence: float
    probabilities: dict
    created_at: datetime
    patient_ref: Optional[str] = None

    model_config = {"from_attributes": True}