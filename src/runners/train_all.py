"""
src/runners/train_all.py

Runs all three training scripts:
- Adult MLP
- California MLP
- Fashion MLP

Saves a single summary JSON in artifacts/train_all/summary.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import time
import json
import tensorflow as tf

from src.models.adult_mlp import train_adult_mlp, AdultMLPTrainingSettings
from src.models.california_mlp import train_california_mlp, CaliforniaMLPTrainingSettings
from src.models.fashion_cnn import train_fashion_cnn, FashionCNNTrainingSettings


@dataclass(frozen=True)
class TrainAllSettings:
    output_directory: str = "artifacts/train_all"
    random_seed: int = 42

    adult_settings: AdultMLPTrainingSettings = AdultMLPTrainingSettings()
    california_settings: CaliforniaMLPTrainingSettings = CaliforniaMLPTrainingSettings()
    fashion_settings: FashionCNNTrainingSettings = FashionCNNTrainingSettings()


def _ensure_directory_exists(directory_path: Path) -> None:
    directory_path.mkdir(parents=True, exist_ok=True)


def train_all(train_all_settings: TrainAllSettings = TrainAllSettings()) -> Dict[str, Dict[str, float]]:
    """
    Trains all models and returns a nested metrics dict like:
    {
      "adult": {"loss": ..., "accuracy": ..., "auc": ...},
      "california": {"loss": ..., "mae": ...},
      "fashion": {"loss": ..., "accuracy": ...}
    }
    """
    tf.keras.utils.set_random_seed(train_all_settings.random_seed)

    train_all_output_directory = Path(train_all_settings.output_directory)
    _ensure_directory_exists(train_all_output_directory)

    adult_output_directory = train_all_output_directory / "adult_mlp"
    california_output_directory = train_all_output_directory / "california_mlp"
    fashion_output_directory = train_all_output_directory / "fashion_cnn"

    start_time_seconds = time.perf_counter()
    adult_metrics = train_adult_mlp(
        output_directory=adult_output_directory,
        training_settings=train_all_settings.adult_settings,
    )
    adult_time_seconds = time.perf_counter() - start_time_seconds
    print(f"Adult training time: {adult_time_seconds:.1f}s")

    start_time_seconds = time.perf_counter()
    california_metrics = train_california_mlp(
        output_directory=california_output_directory,
        training_settings=train_all_settings.california_settings,
    )
    california_time_seconds = time.perf_counter() - start_time_seconds
    print(f"California training time: {california_time_seconds:.1f}s")

    start_time_seconds = time.perf_counter()
    fashion_metrics = train_fashion_cnn(
        output_directory=fashion_output_directory,
        training_settings=train_all_settings.fashion_settings,
    )
    fashion_time_seconds = time.perf_counter() - start_time_seconds
    print(f"Fashion training time: {fashion_time_seconds:.1f}s")

    all_metrics = {
        "adult": adult_metrics,
        "california": california_metrics,
        "fashion": fashion_metrics,
    }

    (train_all_output_directory / "summary.json").write_text(
        json.dumps(all_metrics, indent=2)
    )

    return all_metrics


if __name__ == "__main__":
    # Minimal CLI behavior: just run with defaults
    train_all()
