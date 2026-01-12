from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RAW_DIR

# Cached, preprocessed OpenML download (optional)
ADULT_CACHE_PATH = RAW_DIR / "adult_openml.parquet"


def _standardize_column_names(features_df: pd.DataFrame) -> pd.DataFrame:
    """Make column names consistent and explainer-friendly."""
    cleaned_df = features_df.copy()
    cleaned_df.columns = [
        str(col).strip().replace(" ", "_").replace("-", "_") for col in cleaned_df.columns
    ]
    return cleaned_df


def _clean_string_columns(features_df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and replace common missing tokens in string columns."""
    cleaned_df = features_df.copy()

    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == "object" or str(cleaned_df[col].dtype).startswith("string"):
            cleaned_df[col] = cleaned_df[col].astype("string").str.strip()
            cleaned_df[col] = cleaned_df[col].replace("?", pd.NA)

    return cleaned_df


def _load_adult_from_openml() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load Adult dataset via OpenML.

    Target convention:
      y=1 means income > 50K
      y=0 means income <= 50K
    """
    from sklearn.datasets import fetch_openml

    openml_dataset = fetch_openml("adult", version=2, as_frame=True)
    features_df = openml_dataset.data
    target_series = openml_dataset.target.astype(str).str.strip()

    y = (target_series == ">50K").astype(int).to_numpy()
    return features_df, y


def load_adult_raw(
    prefer_openml: bool = True,
    cache_path: Path = ADULT_CACHE_PATH,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load Adult as a raw pandas DataFrame + binary target array.

    Strategy:
      1) Try OpenML (if prefer_openml=True)
      2) Save a local cache to data/raw/ (optional)
      3) If OpenML fails, load from cache (if present)

    Returns
    -------
    features_df : pd.DataFrame
        Raw features (mixed numeric/categorical).
    y : np.ndarray
        Binary target, 1 => >50K
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if prefer_openml:
        try:
            features_df, y = _load_adult_from_openml()
            features_df = _standardize_column_names(features_df)
            features_df = _clean_string_columns(features_df)

            # cache for offline reproducibility (optional)
            try:
                cached_df = features_df.copy()
                cached_df["__target__"] = y
                cached_df.to_parquet(cache_path, index=False)
            except Exception:
                # Parquet may require optional deps (pyarrow/fastparquet). Safe to ignore.
                pass

            return features_df, y
        except Exception:
            pass  # fall back to cache below

    if cache_path.exists():
        cached_df = pd.read_parquet(cache_path)
        if "__target__" not in cached_df.columns:
            raise ValueError(f"Cache file {cache_path} exists but lacks '__target__' column.")

        y = cached_df["__target__"].to_numpy().astype(int)
        features_df = cached_df.drop(columns=["__target__"])

        features_df = _standardize_column_names(features_df)
        features_df = _clean_string_columns(features_df)
        return features_df, y

    raise FileNotFoundError(
        "Could not load Adult dataset. OpenML failed and no local cache exists at: "
        f"{cache_path}"
    )


def infer_feature_types(features_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Infer numeric vs categorical feature columns by dtype.
    """
    categorical_columns: list[str] = []
    numeric_columns: list[str] = []

    for col in features_df.columns:
        dtype_str = str(features_df[col].dtype)
        is_categorical = (
            features_df[col].dtype == "object"
            or dtype_str.startswith("string")
            or dtype_str.startswith("category")
        )
        if is_categorical:
            categorical_columns.append(col)
        else:
            numeric_columns.append(col)

    return numeric_columns, categorical_columns


def load_adult_splits(
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
    prefer_openml: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """
    Split Adult into train/validation/test.

    Returns
    -------
    X_train_df, X_val_df, X_test_df : pd.DataFrame
    y_train, y_val, y_test : np.ndarray
    meta : dict
        Helpful metadata for preprocessing + explainers.
    """
    features_df, y = load_adult_raw(prefer_openml=prefer_openml)
    numeric_columns, categorical_columns = infer_feature_types(features_df)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        features_df,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_train_df,
        y_train,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train,
    )

    # Helpful for LIME/Anchors: list possible categorical values from TRAIN only
    categorical_value_map: dict[str, list[str]] = {}
    for col in categorical_columns:
        unique_values = X_train_df[col].dropna().astype(str).unique().tolist()
        unique_values.sort()
        categorical_value_map[col] = unique_values

    meta: dict[str, Any] = {
        "dataset": "adult",
        "target_name": "income_gt_50k",
        "class_names": ["<=50K", ">50K"],
        "feature_names": list(features_df.columns),
        "numeric_cols": numeric_columns,
        "categorical_cols": categorical_columns,
        "categorical_names": categorical_value_map,
    }

    return X_train_df, X_val_df, X_test_df, y_train, y_val, y_test, meta
