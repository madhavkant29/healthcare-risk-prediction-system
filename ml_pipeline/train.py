"""
Training pipeline:
- Stratified 5-fold CV with SMOTE applied inside each fold
- XGBoost multi-class (softprob)
- Saves best model.pkl to ml_pipeline/models/
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from preprocess import (
    run_preprocessing,
    apply_smote,
    LABEL_NAMES,
    DATA_DIR,
    MODEL_DIR,
)

XGBOOST_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 6,
    "n_estimators": 400,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "use_label_encoder": False,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
}

N_FOLDS = 5


def train_with_cv(X_train: pd.DataFrame, y_train: np.ndarray):
    """
    Stratified K-fold CV. SMOTE is applied inside each fold
    to prevent data leakage into validation splits.
    Returns the model retrained on the full training set.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_scores = []

    print(f"\nRunning {N_FOLDS}-fold stratified CV...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_fold_train = X_train.iloc[train_idx].values
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train.iloc[val_idx].values
        y_fold_val = y_train[val_idx]

        # SMOTE only on this fold's training data
        X_fold_train_res, y_fold_train_res = apply_smote(X_fold_train, y_fold_train)

        model = XGBClassifier(**XGBOOST_PARAMS)
        model.fit(
            X_fold_train_res,
            y_fold_train_res,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False,
        )

        preds = model.predict(X_fold_val)
        score = f1_score(y_fold_val, preds, average="macro")
        fold_scores.append(score)
        print(f"  Fold {fold}: Macro F1 = {score:.4f}")

    print(f"\n  CV Macro F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    # Retrain on full training set with SMOTE
    print("\nRetraining on full training set...")
    X_full, y_full = apply_smote(X_train.values, y_train)
    final_model = XGBClassifier(**XGBOOST_PARAMS)
    final_model.fit(X_full, y_full, verbose=False)

    return final_model, np.mean(fold_scores)


def save_model(model: XGBClassifier, cv_score: float, feature_names: list):
    model_path = MODEL_DIR / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_names": feature_names,
                "label_names": LABEL_NAMES,
                "cv_macro_f1": cv_score,
                "params": XGBOOST_PARAMS,
            },
            f,
        )
    print(f"Model saved to {model_path}")
    return model_path


def run_training(filepath: str | Path = None):
    if filepath is None:
        filepath = DATA_DIR / "train.csv"

    X_train, X_test, y_train, y_test = run_preprocessing(filepath)

    model, cv_score = train_with_cv(X_train, y_train)

    save_model(model, cv_score, list(X_train.columns))

    # Quick test set evaluation
    from evaluate import evaluate_model
    print("\nTest set evaluation:")
    evaluate_model(model, X_test, y_test)

    return model, X_test, y_test


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    run_training(filepath)