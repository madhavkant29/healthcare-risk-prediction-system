"""
Preprocessing pipeline:
- Load raw CSV
- Encode target labels
- Min-max scale continuous features
- SMOTE oversample minority classes (per fold only)
- Stratified train/test split
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

CONTINUOUS_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

LABEL_MAP = {"NO": 0, "<30": 1, ">30": 2}
LABEL_NAMES = ["NO", "<30", ">30"]


def load_data(filepath: str | Path) -> pd.DataFrame:
    # Auto-detect separator by reading the first line
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline()

    if first_line.count("\t") > first_line.count(","):
        sep = "\t"
    else:
        sep = ","

    print(f"  Detected separator: {'TAB' if sep == chr(9) else 'COMMA'}")
    df = pd.read_csv(filepath, sep=sep)
    print(f"  Columns found: {list(df.columns[:5])} ...")
    return df


def encode_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    y = df["readmitted"].map(LABEL_MAP).values
    X = df.drop(columns=["readmitted"])
    return X, y


def build_preprocessor(X_train: pd.DataFrame) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(X_train[CONTINUOUS_COLS])
    return scaler


def apply_preprocessor(X: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    X = X.copy()
    X[CONTINUOUS_COLS] = scaler.transform(X[CONTINUOUS_COLS])
    return X


def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


def split_data(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
):
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def run_preprocessing(filepath: str | Path):
    """
    Full preprocessing run. Returns split data and saves preprocessor.pkl.
    Call this once before training.
    """
    print(f"Loading data from {filepath}...")
    df = load_data(filepath)
    print(f"  Shape: {df.shape}")

    unique, counts = np.unique(df["readmitted"], return_counts=True)
    print("  Class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"    {cls}: {cnt} ({cnt/len(df)*100:.1f}%)")

    X, y = encode_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    scaler = build_preprocessor(X_train)
    X_train = apply_preprocessor(X_train, scaler)
    X_test = apply_preprocessor(X_test, scaler)

    preprocessor_path = MODEL_DIR / "preprocessor.pkl"
    with open(preprocessor_path, "wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "feature_names": list(X_train.columns),
                "label_map": LABEL_MAP,
                "label_names": LABEL_NAMES,
                "continuous_cols": CONTINUOUS_COLS,
            },
            f,
        )
    print(f"  Preprocessor saved to {preprocessor_path}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR / "train.csv"
    run_preprocessing(filepath)