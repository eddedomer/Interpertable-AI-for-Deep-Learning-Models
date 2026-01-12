"""
src/models/california_mlp.py

Regression MLP for California Housing.

Matches your src/data/california.py:
- load_california(...) returns pandas DataFrames + numpy arrays

We:
- impute missing numeric values (median)
- standardize numeric features
- train a small Keras MLP regressor
- save model + preprocessing transformer + feature names
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import json
import pickle

import numpy as np
import tensorflow as tf

try:
    import keras
    from keras import layers
except Exception:  # pragma: no cover
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CaliforniaMLPTrainingSettings:
    max_epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10

    hidden_layer_units: Tuple[int, ...] = (128, 64)
    dropout_rate: float = 0.10

    random_seed: int = 42

    test_size: float = 0.2
    validation_size: float = 0.2


def _ensure_directory_exists(directory_path: Path) -> None:
    directory_path.mkdir(parents=True, exist_ok=True)


def _build_california_mlp_regressor(
    input_feature_count: int,
    hidden_layer_units: Tuple[int, ...],
    dropout_rate: float,
) -> keras.Model:
    model_input = keras.Input(shape=(input_feature_count,), name="tabular_features")

    hidden_representation = model_input
    for layer_index, units_in_layer in enumerate(hidden_layer_units, start=1):
        hidden_representation = layers.Dense(
            units_in_layer,
            activation="relu",
            name=f"dense_{layer_index}_{units_in_layer}_units",
        )(hidden_representation)

        if dropout_rate and dropout_rate > 0:
            hidden_representation = layers.Dropout(
                rate=dropout_rate,
                name=f"dropout_{layer_index}_{int(dropout_rate * 100)}pct",
            )(hidden_representation)

    regression_output = layers.Dense(
        1, activation="linear", name="predicted_median_house_value"
    )(hidden_representation)

    return keras.Model(model_input, regression_output, name="california_mlp")


def _load_and_preprocess_california_as_numpy(
    training_settings: CaliforniaMLPTrainingSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline, list[str]]:
    from src.data.california import load_california  # <-- your real loader

    (
        train_features_df,
        validation_features_df,
        test_features_df,
        train_targets,
        validation_targets,
        test_targets,
    ) = load_california(
        test_size=training_settings.test_size,
        validation_size=training_settings.validation_size,
        random_state=training_settings.random_seed,
    )

    feature_name_list = list(train_features_df.columns)

    preprocessing_pipeline = Pipeline(
        steps=[
            ("numeric_imputer", SimpleImputer(strategy="median")),
            ("numeric_scaler", StandardScaler()),
        ]
    )

    train_features_array = preprocessing_pipeline.fit_transform(train_features_df)
    validation_features_array = preprocessing_pipeline.transform(validation_features_df)
    test_features_array = preprocessing_pipeline.transform(test_features_df)

    train_features_array = np.asarray(train_features_array, dtype=np.float32)
    validation_features_array = np.asarray(validation_features_array, dtype=np.float32)
    test_features_array = np.asarray(test_features_array, dtype=np.float32)

    train_targets_array = np.asarray(train_targets, dtype=np.float32).reshape(-1, 1)
    validation_targets_array = np.asarray(validation_targets, dtype=np.float32).reshape(-1, 1)
    test_targets_array = np.asarray(test_targets, dtype=np.float32).reshape(-1, 1)

    return (
        train_features_array,
        train_targets_array,
        validation_features_array,
        validation_targets_array,
        test_features_array,
        test_targets_array,
        preprocessing_pipeline,
        feature_name_list,
    )


def train_california_mlp(
    output_directory: Path | str = "artifacts/california_mlp",
    training_settings: CaliforniaMLPTrainingSettings = CaliforniaMLPTrainingSettings(),
) -> Dict[str, float]:
    """
    Saves:
    - best_model.keras
    - training_history.json
    - evaluation_metrics.json
    - preprocessing.pkl
    - feature_names.json
    """
    output_directory_path = Path(output_directory)
    _ensure_directory_exists(output_directory_path)

    tf.keras.utils.set_random_seed(training_settings.random_seed)

    (
        train_features_array,
        train_targets_array,
        validation_features_array,
        validation_targets_array,
        test_features_array,
        test_targets_array,
        preprocessing_pipeline,
        feature_name_list,
    ) = _load_and_preprocess_california_as_numpy(training_settings)

    input_feature_count = int(train_features_array.shape[1])

    california_mlp_model = _build_california_mlp_regressor(
        input_feature_count=input_feature_count,
        hidden_layer_units=training_settings.hidden_layer_units,
        dropout_rate=training_settings.dropout_rate,
    )

    california_mlp_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=training_settings.learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )

    best_model_path = output_directory_path / "best_model.keras"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=training_settings.early_stopping_patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    training_history = california_mlp_model.fit(
        train_features_array,
        train_targets_array,
        validation_data=(validation_features_array, validation_targets_array),
        epochs=training_settings.max_epochs,
        batch_size=training_settings.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    evaluation_values = california_mlp_model.evaluate(test_features_array, test_targets_array, verbose=0)
    metric_name_to_value: Dict[str, float] = {
        metric_name: float(metric_value)
        for metric_name, metric_value in zip(california_mlp_model.metrics_names, evaluation_values)
    }

    (output_directory_path / "training_history.json").write_text(json.dumps(training_history.history, indent=2))
    (output_directory_path / "evaluation_metrics.json").write_text(json.dumps(metric_name_to_value, indent=2))
    (output_directory_path / "feature_names.json").write_text(json.dumps(feature_name_list, indent=2))

    with open(output_directory_path / "preprocessing.pkl", "wb") as file_handle:
        pickle.dump(preprocessing_pipeline, file_handle)

    return metric_name_to_value
