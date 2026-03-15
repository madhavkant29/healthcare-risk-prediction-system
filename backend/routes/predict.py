from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models.schemas import (
    PredictionRequest,
    PredictionResult,
    BatchPredictionRequest,
    BatchPredictionResult,
    SHAPExplanation,
    SHAPFeature,
)
from services.prediction_service import prediction_service
from services.patient_service import save_prediction
from routes.deps import get_current_user
from models.orm import User

router = APIRouter(prefix="/predict", tags=["predict"])


def _build_result(raw: dict, prediction_id: int = None) -> PredictionResult:
    shap_exp = None
    if raw.get("shap_explanation"):
        s = raw["shap_explanation"]
        shap_exp = SHAPExplanation(
            predicted_class=s["predicted_class"],
            base_value=s["base_value"],
            top_features=[SHAPFeature(**f) for f in s["top_features"]],
        )
    return PredictionResult(
        predicted_class=raw["predicted_class"],
        confidence=raw["confidence"],
        probabilities=raw["probabilities"],
        shap_explanation=shap_exp,
        prediction_id=prediction_id,
    )


@router.post("", response_model=PredictionResult)
def predict_single(
    req: PredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not prediction_service.is_loaded:
        try:
            prediction_service.load()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))

    raw = prediction_service.predict_single(req.features)

    saved = save_prediction(
        db,
        user_id=current_user.id,
        result=raw,
        input_features=req.features,
        patient_ref=req.patient_ref,
    )

    return _build_result(raw, prediction_id=saved.id)


@router.post("/batch", response_model=BatchPredictionResult)
def predict_batch(
    req: BatchPredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not req.records:
        raise HTTPException(status_code=400, detail="No records provided")

    if not prediction_service.is_loaded:
        try:
            prediction_service.load()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))

    raw_results = prediction_service.predict_batch(req.records)

    results = []
    for raw, features in zip(raw_results, req.records):
        saved = save_prediction(
            db,
            user_id=current_user.id,
            result=raw,
            input_features=features,
        )
        results.append(_build_result(raw, prediction_id=saved.id))

    return BatchPredictionResult(total=len(results), results=results)