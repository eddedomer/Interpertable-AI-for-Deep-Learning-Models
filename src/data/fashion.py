from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import RAW_DIR


FASHION_MNIST_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_fashion(
    data_root: str | None = None,
    download: bool = True,
    normalize: bool = False,
) -> Tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    """
    Load Fashion-MNIST as torchvision datasets.

    Returns
    -------
    train_dataset, test_dataset : torchvision.datasets.FashionMNIST
    """
    if data_root is None:
        data_root = str(RAW_DIR / "fashion_mnist")

    transform_steps = [transforms.ToTensor()]
    if normalize:
        transform_steps.append(transforms.Normalize((0.5,), (0.5,)))

    transform = transforms.Compose(transform_steps)

    train_dataset = datasets.FashionMNIST(
        root=data_root,
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=data_root,
        train=False,
        download=download,
        transform=transform,
    )
    return train_dataset, test_dataset


def make_fashion_loaders(
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
    normalize: bool = False,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Convenience helper that returns DataLoaders + class names.
    """
    train_dataset, test_dataset = load_fashion(normalize=normalize)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, FASHION_MNIST_CLASS_NAMES
