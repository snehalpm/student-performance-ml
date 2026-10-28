"""
Shared utilities for Student Performance Prediction and At-Risk Analysis.
Handles data loading, feature partitioning, preprocessing pipelines, and defaults.
"""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Default paths
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "student-mat.csv"


def load_data(data_path: Path | str | None = None) -> pd.DataFrame:
    """
    Load student performance dataset from CSV (semicolon-separated).
    Raises a descriptive FileNotFoundError if the file is missing.
    """
    path = Path(data_path) if data_path else DEFAULT_DATA_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{path.resolve()}'. "
            "Please ensure 'student-mat.csv' is placed inside the 'data/' directory."
        )
    return pd.read_csv(path, sep=";")


def get_feature_lists(df: pd.DataFrame, include_g1_g2: bool = True) -> tuple[list[str], list[str]]:
    """
    Partition dataframe columns into numeric and categorical features,
    excluding target variables (G3 and at_risk).
    Optionally includes or excludes G1 and G2 for early-warning evaluation.
    """
    # Categorical features in the UCI Student Performance dataset
    categorical_cols = [
        "school", "sex", "address", "famsize", "Pstatus",
        "Mjob", "Fjob", "reason", "guardian", "schoolsup",
        "famsup", "paid", "activities", "nursery", "higher",
        "internet", "romantic"
    ]

    # Base numeric features
    numeric_cols = [
        "age", "Medu", "Fedu", "traveltime", "studytime",
        "failures", "famrel", "freetime", "goout", "Dalc",
        "Walc", "health", "absences"
    ]

    if include_g1_g2:
        numeric_cols.extend(["G1", "G2"])

    # Ensure all identified columns exist in df
    available_cat = [c for c in categorical_cols if c in df.columns]
    available_num = [c for c in numeric_cols if c in df.columns]

    return available_num, available_cat


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Create a ColumnTransformer that one-hot encodes categorical variables
    and optionally scales numeric features with StandardScaler.
    Prevents data leakage when used within an sklearn Pipeline.
    """
    num_transformer = StandardScaler() if scale_numeric else "passthrough"
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_features),
            ("cat", cat_transformer, categorical_features),
        ],
        remainder="drop",
    )


def clean_feature_name(raw_name: str) -> str:
    """
    Convert raw transformer feature names into human-readable labels.
    E.g., 'num__G2' -> 'G2', 'cat__Mjob_services' -> 'Mjob: services'.
    """
    if raw_name.startswith("num__"):
        return raw_name.replace("num__", "")
    if raw_name.startswith("cat__"):
        cleaned = raw_name.replace("cat__", "")
        # Split on the first underscore to format as Feature: Value
        parts = cleaned.split("_", 1)
        if len(parts) == 2:
            return f"{parts[0]}: {parts[1]}"
        return cleaned
    return raw_name


def get_default_student() -> dict:
    """
    Provide sensible default values (median/mode) for all features,
    useful for initializing user inputs in the Streamlit prediction form.
    """
    return {
        "school": "GP",
        "sex": "F",
        "age": 16,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 2,
        "Fedu": 2,
        "Mjob": "other",
        "Fjob": "other",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 1,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 1,
        "health": 3,
        "absences": 2,
        "G1": 11,
        "G2": 11,
    }
