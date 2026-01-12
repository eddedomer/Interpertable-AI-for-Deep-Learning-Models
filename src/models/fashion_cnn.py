"""
src/models/fashion_cnn.py

CNN for Fashion-MNIST using PyTorch + torchvision.

Matches your src/data/fashion.py:
- load_fashion(...) returns torchvision datasets
- FASHION_MNIST_CLASS_NAMES lists class names

We:
- create a train/val split from the torchvision train dataset
- train a simple CNN with early stopping
- save the best state_dict as best_model.pt
- save history + test metrics as json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import json
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split


@dataclass(frozen=True)
class FashionCNNTrainingSettings:
    max_epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: int = 5

    validation_fraction: float = 0.1
    normalize: bool = False

    random_seed: int = 42
    num_workers: int = 0
    pin_memory: bool = False


class FashionCNN(nn.Module):
    def __init__(self, class_count: int = 10) -> None:
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 28 -> 14

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 14 -> 7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=128, out_features=class_count),
        )

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(image_batch)
        logits = self.classifier(features)
        return logits


def _ensure_directory_exists(directory_path: Path) -> None:
    directory_path.mkdir(parents=True, exist_ok=True)


def _pick_training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _compute_accuracy(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()

    correct_prediction_count = 0
    total_sample_count = 0

    for image_batch, label_batch in data_loader:
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        logits = model(image_batch)
        predicted_labels = torch.argmax(logits, dim=1)

        correct_prediction_count += int((predicted_labels == label_batch).sum().item())
        total_sample_count += int(label_batch.numel())

    if total_sample_count == 0:
        return 0.0
    return correct_prediction_count / total_sample_count


def train_fashion_cnn(
    output_directory: Path | str = "artifacts/fashion_cnn",
    training_settings: FashionCNNTrainingSettings = FashionCNNTrainingSettings(),
) -> Dict[str, float]:
    """
    Saves:
    - best_model.pt
    - training_history.json
    - evaluation_metrics.json
    - class_names.json
    """
    output_directory_path = Path(output_directory)
    _ensure_directory_exists(output_directory_path)

    # Reproducibility
    torch.manual_seed(training_settings.random_seed)

    from src.data.fashion import load_fashion, FASHION_MNIST_CLASS_NAMES

    full_train_dataset, test_dataset = load_fashion(normalize=training_settings.normalize)

    validation_sample_count = int(len(full_train_dataset) * training_settings.validation_fraction)
    training_sample_count = len(full_train_dataset) - validation_sample_count

    split_generator = torch.Generator().manual_seed(training_settings.random_seed)
    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        lengths=[training_sample_count, validation_sample_count],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_settings.batch_size,
        shuffle=True,
        num_workers=training_settings.num_workers,
        pin_memory=training_settings.pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training_settings.batch_size,
        shuffle=False,
        num_workers=training_settings.num_workers,
        pin_memory=training_settings.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_settings.batch_size,
        shuffle=False,
        num_workers=training_settings.num_workers,
        pin_memory=training_settings.pin_memory,
    )

    device = _pick_training_device()

    model = FashionCNN(class_count=len(FASHION_MNIST_CLASS_NAMES)).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_settings.learning_rate,
        weight_decay=training_settings.weight_decay,
    )

    best_model_path = output_directory_path / "best_model.pt"
    best_validation_accuracy = -1.0
    epochs_without_improvement = 0

    training_history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_accuracy": [],
        "epoch_seconds": [],
    }

    for epoch_index in range(1, training_settings.max_epochs + 1):
        epoch_start_time = time.perf_counter()

        model.train()
        running_loss_sum = 0.0
        seen_sample_count = 0

        for image_batch, label_batch in train_loader:
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(image_batch)
            batch_loss = loss_function(logits, label_batch)

            batch_loss.backward()
            optimizer.step()

            batch_size = int(label_batch.numel())
            running_loss_sum += float(batch_loss.item()) * batch_size
            seen_sample_count += batch_size

        average_training_loss = running_loss_sum / max(seen_sample_count, 1)
        validation_accuracy = _compute_accuracy(model, validation_loader, device)

        epoch_seconds = time.perf_counter() - epoch_start_time

        training_history["train_loss"].append(float(average_training_loss))
        training_history["val_accuracy"].append(float(validation_accuracy))
        training_history["epoch_seconds"].append(float(epoch_seconds))

        # Early stopping on validation accuracy
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= training_settings.early_stopping_patience:
            break

    # Load best weights for evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_accuracy = _compute_accuracy(model, test_loader, device)

    evaluation_metrics = {
        "test_accuracy": float(test_accuracy),
        "best_val_accuracy": float(best_validation_accuracy),
        "device": str(device),
    }

    (output_directory_path / "training_history.json").write_text(json.dumps(training_history, indent=2))
    (output_directory_path / "evaluation_metrics.json").write_text(json.dumps(evaluation_metrics, indent=2))
    (output_directory_path / "class_names.json").write_text(json.dumps(FASHION_MNIST_CLASS_NAMES, indent=2))

    return evaluation_metrics
