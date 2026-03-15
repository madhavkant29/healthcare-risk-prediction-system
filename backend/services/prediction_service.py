"""
Prediction service:
- Loads model.pkl and preprocessor.pkl once at startup
- Applies same preprocessing as training pipeline
- Runs XGBoost inference
- Computes SHAP explanation for predicted class
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import sys

from config import settings

# Allow importing from ml_pipeline
ML_PIPELINE_DIR = Path(__file__).parent.parent.parent / "ml_pipeline"
sys.path.insert(0, str(ML_PIPELINE_DIR))
from explainability import load_explainer, explain_single  # noqa: E402


class PredictionService:
    _instance = None

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names: list = []
        self.label_names: list = []
        self.label_map: dict = {}
        self.continuous_cols: list = []
        self.explainer = None
        self._loaded = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self):
        if self._loaded:
            return

        model_path = Path(settings.MODEL_PATH)
        prep_path = Path(settings.PREPROCESSOR_PATH)

        if not model_path.exists():
            raise FileNotFoundError(
                f"model.pkl not found at {model_path}. "
                "Run ml_pipeline/train.py first."
            )
        if not prep_path.exists():
            raise FileNotFoundError(
                f"preprocessor.pkl not found at {prep_path}. "
                "Run ml_pipeline/train.py first."
            )

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        with open(prep_path, "rb") as f:
            prep_data = pickle.load(f)

        self.model = model_data["model"]
        self.feature_names = prep_data["feature_names"]
        self.label_names = prep_data["label_names"]
        self.label_map = prep_data["label_map"]
        self.continuous_cols = prep_data["continuous_cols"]
        self.scaler = prep_data["scaler"]
        self.explainer = load_explainer(self.model)
        self._loaded = True

        print(f"Model loaded. CV Macro F1: {model_data.get('cv_macro_f1', 'N/A')}")

    def _preprocess(self, features: dict) -> pd.DataFrame:
        """Align input features to training columns and apply scaler."""
        row = pd.DataFrame([features])

        # Align columns to training feature set — fill missing with 0
        row = row.reindex(columns=self.feature_names, fill_value=0)

        # Apply scaling to continuous cols
        row[self.continuous_cols] = self.scaler.transform(row[self.continuous_cols])

        return row

    def predict_single(
        self, features: dict, include_shap: bool = True
    ) -> dict:
        if not self._loaded:
            self.load()

        X = self._preprocess(features)
        X_arr = X.values

        proba = self.model.predict_proba(X_arr)[0]
        class_idx = int(np.argmax(proba))
        predicted_class = self.label_names[class_idx]
        confidence = float(proba[class_idx])
        probabilities = {
            name: round(float(p), 4)
            for name, p in zip(self.label_names, proba)
        }

        shap_result = None
        if include_shap:
            try:
                shap_result = explain_single(
                    self.explainer,
                    X,
                    self.feature_names,
                    self.label_names,
                    class_idx,
                )
            except Exception as e:
                print(f"SHAP computation failed: {e}")

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "shap_explanation": shap_result,
        }

    def predict_batch(self, records: list[dict]) -> list[dict]:
        """Batch prediction without SHAP (for speed)."""
        if not self._loaded:
            self.load()

        results = []
        for record in records:
            result = self.predict_single(record, include_shap=False)
            results.append(result)
        return results

    @property
    def is_loaded(self) -> bool:
        return self._loaded


prediction_service = PredictionService.get_instance()