from sqlalchemy.orm import Session
from models.orm import Patient, Prediction
from models.schemas import PredictionResult
from typing import Optional
import hashlib


def anonymise_ref(raw_ref: str) -> str:
    """One-way hash patient reference for storage."""
    return hashlib.sha256(raw_ref.encode()).hexdigest()[:16]


def get_or_create_patient(db: Session, patient_ref: str) -> Patient:
    anon_ref = anonymise_ref(patient_ref)
    patient = db.query(Patient).filter(Patient.patient_ref == anon_ref).first()
    if not patient:
        patient = Patient(patient_ref=anon_ref)
        db.add(patient)
        db.commit()
        db.refresh(patient)
    return patient


def save_prediction(
    db: Session,
    user_id: int,
    result: dict,
    input_features: dict,
    patient_ref: Optional[str] = None,
) -> Prediction:
    patient_id = None
    if patient_ref:
        patient = get_or_create_patient(db, patient_ref)
        patient_id = patient.id

    prediction = Prediction(
        user_id=user_id,
        patient_id=patient_id,
        input_features=input_features,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        shap_values=result.get("shap_explanation"),
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


def get_prediction_history(
    db: Session,
    user_id: int,
    limit: int = 50,
    offset: int = 0,
) -> list[Prediction]:
    return (
        db.query(Prediction)
        .filter(Prediction.user_id == user_id)
        .order_by(Prediction.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_prediction_by_id(
    db: Session, prediction_id: int, user_id: int
) -> Optional[Prediction]:
    return (
        db.query(Prediction)
        .filter(
            Prediction.id == prediction_id,
            Prediction.user_id == user_id,
        )
        .first()
    )


def get_stats(db: Session, user_id: int) -> dict:
    predictions = db.query(Prediction).filter(Prediction.user_id == user_id).all()
    if not predictions:
        return {"total": 0, "by_class": {}, "avg_confidence": 0}

    by_class = {}
    total_conf = 0
    for p in predictions:
        by_class[p.predicted_class] = by_class.get(p.predicted_class, 0) + 1
        total_conf += p.confidence

    return {
        "total": len(predictions),
        "by_class": by_class,
        "avg_confidence": round(total_conf / len(predictions), 4),
    }