"""
Evaluation:
- Per-class precision, recall, F1
- Macro F1, ROC-AUC (OvR)
- Confusion matrix
- Prints a clean report
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

LABEL_NAMES = ["NO", "<30", ">30"]


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame | np.ndarray,
    y_test: np.ndarray,
    label_names: list = LABEL_NAMES,
) -> dict:
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    try:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = None

    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    print(f"\nMacro F1     : {macro_f1:.4f}")
    print(f"Weighted F1  : {weighted_f1:.4f}")
    if roc_auc:
        print(f"ROC-AUC (OvR): {roc_auc:.4f}")

    print("\nPer-class report:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    print("Confusion matrix:")
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    print(cm_df.to_string())
    print("=" * 50)

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=label_names, output_dict=True
        ),
    }


if __name__ == "__main__":
    import pickle, sys
    from pathlib import Path

    MODEL_DIR = Path(__file__).parent / "models"

    model_data = pickle.load(open(MODEL_DIR / "model.pkl", "rb"))
    prep_data = pickle.load(open(MODEL_DIR / "preprocessor.pkl", "rb"))

    print("Loaded model. CV Macro F1 during training:", model_data["cv_macro_f1"])