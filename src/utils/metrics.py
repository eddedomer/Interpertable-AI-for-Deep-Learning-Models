import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score
)

def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "R2": float(r2_score(y_true, y_pred)),
    }

def classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """
    y_proba: shape (n, 2) or (n,) for positive class prob
    """
    y_true = np.asarray(y_true)

    if y_proba.ndim == 2:
        p = y_proba[:, 1]
    else:
        p = y_proba

    y_pred = (p >= threshold).astype(int)
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred)),
    }
    # AUC requires both classes present in y_true
    if len(np.unique(y_true)) == 2:
        out["AUC"] = float(roc_auc_score(y_true, p))
    return out
