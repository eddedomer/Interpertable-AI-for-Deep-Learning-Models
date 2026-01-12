from dataclasses import dataclass
from typing import Callable, Optional, Any, Dict
import numpy as np
import pandas as pd

@dataclass
class TabularPredictor:
    feature_names: list[str]
    predict: Callable[[pd.DataFrame], np.ndarray]                  # regression OR labels
    predict_proba: Optional[Callable[[pd.DataFrame], np.ndarray]]  # classification
    meta: Dict[str, Any] = None

def make_keras_tabular_predictor(preprocess, keras_model, feature_names, task: str, meta=None) -> TabularPredictor:
    """
    preprocess: sklearn ColumnTransformer (fit)
    keras_model: fitted tf.keras.Model
    task: "regression" or "classification"
    """
    import numpy as np

    def _to_X(df: pd.DataFrame):
        return preprocess.transform(df)

    def predict(df: pd.DataFrame) -> np.ndarray:
        X = _to_X(df)
        y = keras_model.predict(X, verbose=0).reshape(-1)
        if task == "classification":
            return (y >= 0.5).astype(int)
        return y

    def predict_proba(df: pd.DataFrame) -> np.ndarray:
        X = _to_X(df)
        p1 = keras_model.predict(X, verbose=0).reshape(-1)
        p1 = np.clip(p1, 1e-7, 1 - 1e-7)
        return np.column_stack([1 - p1, p1])

    return TabularPredictor(
        feature_names=feature_names,
        predict=predict,
        predict_proba=predict_proba if task == "classification" else None,
        meta=meta or {}
    )
