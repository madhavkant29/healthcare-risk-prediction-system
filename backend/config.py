from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Healthcare Risk Prediction API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Security
    SECRET_KEY: str = "change-this-in-production-use-openssl-rand-hex-32"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 8  # 8 hours

    # Database
    DATABASE_URL: str = "sqlite:///./healthcare.db"

    # Model paths
    MODEL_PATH: str = str(
        Path(__file__).parent.parent / "ml_pipeline" / "models" / "model.pkl"
    )
    PREPROCESSOR_PATH: str = str(
        Path(__file__).parent.parent / "ml_pipeline" / "models" / "preprocessor.pkl"
    )

    # CORS — restrict in production
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()