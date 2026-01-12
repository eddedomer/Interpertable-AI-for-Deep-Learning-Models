from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_california(
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the California Housing dataset as pandas DataFrames.

    Returns
    -------
    X_train_df, X_val_df, X_test_df : pd.DataFrame
        Raw feature tables.
    y_train, y_val, y_test : np.ndarray
        Continuous target values (median house value).
    """
    dataset = fetch_california_housing(as_frame=True)

    features_df: pd.DataFrame = dataset.data.copy()
    target: np.ndarray = dataset.target.to_numpy()

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        features_df,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_train_df,
        y_train,
        test_size=validation_size,
        random_state=random_state,
    )

    return X_train_df, X_val_df, X_test_df, y_train, y_val, y_test
