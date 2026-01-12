"""
src/models/adult_mlp.py

Binary classification MLP for the Adult Census Income dataset.

This version matches your src/data/adult.py which provides:
- load_adult_splits(...)

It handles:
- pandas DataFrames (mixed numeric + categorical)
- missing values
- one-hot encoding for categoricals
- scaling for numerics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any

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

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class AdultMLPTrainingSettings:
    max_epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    early_stopping_patience: int = 6

    hidden_layer_units: Tuple[int, ...] = (128, 64)
    dropout_rate: float = 0.20

    random_seed: int = 42

    # Data split params (should match your load_adult_splits defaults)
    test_size: float = 0.2
    validation_size: float = 0.2
    prefer_openml: bool = True


def _ensure_directory_exists(directory_path: Path) -> None:
    directory_path.mkdir(parents=True, exist_ok=True)


def _build_adult_mlp_model(
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

    probability_output = layers.Dense(
        1,
        activation="sigmoid",
        name="positive_class_probability",
    )(hidden_representation)

    return keras.Model(model_input, probability_output, name="adult_mlp")


def _create_one_hot_encoder() -> OneHotEncoder:
    """
    sklearn changed the argument name from `sparse` -> `sparse_output`.
    This helper keeps compatibility across versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessing_pipeline(
    numeric_column_names: list[str],
    categorical_column_names: list[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("numeric_imputer", SimpleImputer(strategy="median")),
            ("numeric_scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("categorical_imputer", SimpleImputer(strategy="most_frequent")),
            ("categorical_one_hot", _create_one_hot_encoder()),
        ]
    )

    preprocessing_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_column_names),
            ("categorical", categorical_pipeline, categorical_column_names),
        ],
        remainder="drop",
    )

    return preprocessing_transformer


def _extract_preprocessed_feature_names(
    preprocessing_transformer: ColumnTransformer,
    numeric_column_names: list[str],
    categorical_column_names: list[str],
) -> list[str]:
    """
    Produces human-readable feature names after preprocessing:
    - numeric columns keep their names
    - categorical columns become one-hot expanded names
    """
    expanded_feature_names: list[str] = []
    expanded_feature_names.extend(numeric_column_names)

    # Access the fitted OneHotEncoder inside the ColumnTransformer
    fitted_one_hot_encoder: OneHotEncoder = (
        preprocessing_transformer.named_transformers_["categorical"]
        .named_steps["categorical_one_hot"]
    )

    try:
        one_hot_feature_names = fitted_one_hot_encoder.get_feature_names_out(categorical_column_names).tolist()
    except Exception:
        # Older sklearn fallback
        one_hot_feature_names = fitted_one_hot_encoder.get_feature_names(categorical_column_names).tolist()

    expanded_feature_names.extend(one_hot_feature_names)
    return expanded_feature_names


def _load_and_preprocess_adult_as_numpy(
    training_settings: AdultMLPTrainingSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], ColumnTransformer, list[str]]:
    """
    Loads Adult via your src/data/adult.py, then preprocesses:
    - imputes missing values
    - scales numeric
    - one-hot encodes categoricals
    Returns numpy arrays ready for Keras.
    """
    from src.data.adult import load_adult_splits  # <-- your real function

    (
        train_features_df,
        validation_features_df,
        test_features_df,
        train_labels,
        validation_labels,
        test_labels,
        metadata_dict,
    ) = load_adult_splits(
        test_size=training_settings.test_size,
        validation_size=training_settings.validation_size,
        random_state=training_settings.random_seed,
        prefer_openml=training_settings.prefer_openml,
    )

    numeric_column_names: list[str] = metadata_dict["numeric_cols"]
    categorical_column_names: list[str] = metadata_dict["categorical_cols"]

    preprocessing_transformer = _build_preprocessing_pipeline(
        numeric_column_names=numeric_column_names,
        categorical_column_names=categorical_column_names,
    )

    # Fit on train only
    train_features_array = preprocessing_transformer.fit_transform(train_features_df)
    validation_features_array = preprocessing_transformer.transform(validation_features_df)
    test_features_array = preprocessing_transformer.transform(test_features_df)

    # Ensure float32 for TensorFlow
    train_features_array = np.asarray(train_features_array, dtype=np.float32)
    validation_features_array = np.asarray(validation_features_array, dtype=np.float32)
    test_features_array = np.asarray(test_features_array, dtype=np.float32)

    train_labels_array = np.asarray(train_labels, dtype=np.float32).reshape(-1, 1)
    validation_labels_array = np.asarray(validation_labels, dtype=np.float32).reshape(-1, 1)
    test_labels_array = np.asarray(test_labels, dtype=np.float32).reshape(-1, 1)

    expanded_feature_names = _extract_preprocessed_feature_names(
        preprocessing_transformer=preprocessing_transformer,
        numeric_column_names=numeric_column_names,
        categorical_column_names=categorical_column_names,
    )

    return (
        train_features_array,
        train_labels_array,
        validation_features_array,
        validation_labels_array,
        test_features_array,
        test_labels_array,
        metadata_dict,
        preprocessing_transformer,
        expanded_feature_names,
    )


def train_adult_mlp(
    output_directory: Path | str = "artifacts/adult_mlp",
    training_settings: AdultMLPTrainingSettings = AdultMLPTrainingSettings(),
) -> Dict[str, float]:
    """
    Trains an MLP for Adult and saves:
    - best_model.keras
    - training_history.json
    - evaluation_metrics.json
    - preprocessing.pkl
    - expanded_feature_names.json
    - dataset_metadata.json
    """
    output_directory_path = Path(output_directory)
    _ensure_directory_exists(output_directory_path)

    tf.keras.utils.set_random_seed(training_settings.random_seed)

    (
        train_features_array,
        train_labels_array,
        validation_features_array,
        validation_labels_array,
        test_features_array,
        test_labels_array,
        metadata_dict,
        preprocessing_transformer,
        expanded_feature_names,
    ) = _load_and_preprocess_adult_as_numpy(training_settings)

    input_feature_count = int(train_features_array.shape[1])

    adult_mlp_model = _build_adult_mlp_model(
        input_feature_count=input_feature_count,
        hidden_layer_units=training_settings.hidden_layer_units,
        dropout_rate=training_settings.dropout_rate,
    )

    adult_mlp_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=training_settings.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
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

    training_history = adult_mlp_model.fit(
        train_features_array,
        train_labels_array,
        validation_data=(validation_features_array, validation_labels_array),
        epochs=training_settings.max_epochs,
        batch_size=training_settings.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    evaluation_values = adult_mlp_model.evaluate(test_features_array, test_labels_array, verbose=0)
    metric_name_to_value: Dict[str, float] = {
        metric_name: float(metric_value)
        for metric_name, metric_value in zip(adult_mlp_model.metrics_names, evaluation_values)
    }

    # Save artifacts helpful for explainability
    (output_directory_path / "training_history.json").write_text(
        json.dumps(training_history.history, indent=2)
    )
    (output_directory_path / "evaluation_metrics.json").write_text(
        json.dumps(metric_name_to_value, indent=2)
    )
    (output_directory_path / "expanded_feature_names.json").write_text(
        json.dumps(expanded_feature_names, indent=2)
    )
    (output_directory_path / "dataset_metadata.json").write_text(
        json.dumps(metadata_dict, indent=2)
    )

    with open(output_directory_path / "preprocessing.pkl", "wb") as file_handle:
        pickle.dump(preprocessing_transformer, file_handle)

    return metric_name_to_value
