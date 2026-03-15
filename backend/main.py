from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import settings
from database import init_db
from services.prediction_service import prediction_service
from routes import auth, predict, patients


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    try:
        prediction_service.load()
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("API will start but /predict will return 503 until model is trained.")
    yield
    # Shutdown (nothing to clean up)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Readmission risk prediction API for clinical decision support.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(patients.router)


@app.get("/health", tags=["health"])
def health():
    return {
        "status": "ok",
        "model_loaded": prediction_service.is_loaded,
        "version": settings.APP_VERSION,
    }