from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from database import get_db
from models.schemas import PredictionHistory
from services.patient_service import get_prediction_history, get_stats, get_prediction_by_id
from routes.deps import get_current_user
from models.orm import User

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("/history", response_model=list[PredictionHistory])
def history(
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    records = get_prediction_history(db, current_user.id, limit=limit, offset=offset)
    results = []
    for r in records:
        patient_ref = r.patient.patient_ref if r.patient else None
        results.append(
            PredictionHistory(
                id=r.id,
                predicted_class=r.predicted_class,
                confidence=r.confidence,
                probabilities=r.probabilities,
                created_at=r.created_at,
                patient_ref=patient_ref,
            )
        )
    return results


@router.get("/stats")
def stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_stats(db, current_user.id)


@router.get("/history/{prediction_id}")
def get_single(
    prediction_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    record = get_prediction_by_id(db, prediction_id, current_user.id)
    if not record:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Prediction not found")
    return record