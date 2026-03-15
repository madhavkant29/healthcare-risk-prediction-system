"""
SHAP explainability:
- TreeExplainer for XGBoost (exact, fast)
- Per-prediction waterfall values (for API)
- Global beeswarm plot (for notebooks/dashboard)
"""

import numpy as np
import pandas as pd
import pickle
import shap
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"


def load_explainer(model) -> shap.TreeExplainer:
    explainer = shap.TreeExplainer(model)
    return explainer


def explain_single(
    explainer: shap.TreeExplainer,
    X_row: pd.DataFrame | np.ndarray,
    feature_names: list,
    label_names: list,
    predicted_class_idx: int,
) -> dict:
    """
    Returns SHAP values for a single prediction.
    Returns top contributing features for the predicted class.
    """
    if isinstance(X_row, pd.DataFrame):
        X_arr = X_row.values
    else:
        X_arr = X_row

    shap_values = explainer.shap_values(X_arr)

    # shap_values shape: (n_samples, n_features, n_classes) for multi-class
    if isinstance(shap_values, list):
        # older SHAP returns list of arrays, one per class
        class_shap = shap_values[predicted_class_idx][0]
    else:
        class_shap = shap_values[0, :, predicted_class_idx]

    feature_shap = dict(zip(feature_names, class_shap.tolist()))

    # Top 10 by absolute SHAP value
    sorted_features = sorted(
        feature_shap.items(), key=lambda x: abs(x[1]), reverse=True
    )[:10]

    return {
        "predicted_class": label_names[predicted_class_idx],
        "top_features": [
            {"feature": k, "shap_value": round(v, 5)}
            for k, v in sorted_features
        ],
        "base_value": float(explainer.expected_value[predicted_class_idx])
        if hasattr(explainer.expected_value, "__len__")
        else float(explainer.expected_value),
    }


def explain_batch(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame | np.ndarray,
    feature_names: list,
) -> np.ndarray:
    """Returns raw SHAP values for a batch. Shape: (n_samples, n_features, n_classes)."""
    if isinstance(X, pd.DataFrame):
        X = X.values
    return explainer.shap_values(X)


def plot_global_importance(
    explainer: shap.TreeExplainer,
    X_sample: pd.DataFrame,
    feature_names: list,
    save_path: Path = None,
):
    """Beeswarm plot — run from notebook."""
    import matplotlib.pyplot as plt

    shap_vals = explainer.shap_values(X_sample.values)

    if isinstance(shap_vals, list):
        # Average absolute SHAP across classes
        mean_shap = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
    else:
        mean_shap = np.mean(np.abs(shap_vals), axis=2)

    shap.summary_plot(
        mean_shap,
        X_sample,
        feature_names=feature_names,
        show=save_path is None,
    )
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    model_data = pickle.load(open(MODEL_DIR / "model.pkl", "rb"))
    prep_data = pickle.load(open(MODEL_DIR / "preprocessor.pkl", "rb"))

    model = model_data["model"]
    feature_names = prep_data["feature_names"]
    label_names = prep_data["label_names"]

    explainer = load_explainer(model)
    print("Explainer ready. Expected value:", explainer.expected_value)