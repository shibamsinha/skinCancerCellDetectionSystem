"""PyTorch dataset, class-imbalance handling and DataLoader construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from skincancer.preprocessing.image_loader import load_image
from skincancer.preprocessing.transforms import ImageTransform
from skincancer.utils.seed import worker_init_fn


class LesionDataset(Dataset):
    """Reads images lazily from disk; only file paths and labels are held in memory."""

    def __init__(self, records: pd.DataFrame, images_dir: str | Path, transform: ImageTransform) -> None:
        self.image_ids = records["image_id"].tolist()
        self.labels = records["label"].astype(int).to_numpy()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = load_image(self.images_dir / f"{self.image_ids[idx]}.jpg")
        return self.transform(img), int(self.labels[idx])


def class_weights(labels: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency class weights, normalised so they average to 1."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"Every class needs at least one sample, got counts {counts.tolist()}")
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def oversampling_sampler(labels: np.ndarray, generator: torch.Generator) -> WeightedRandomSampler:
    """Sample with replacement so both classes are drawn equally often.

    Each repeated draw of a minority image gets a fresh random augmentation,
    which mirrors the source paper's augmentation-based class balancing.
    """
    per_class = class_weights(labels).numpy()
    sample_weights = per_class[labels]
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    loss_weights: torch.Tensor | None  # set only for imbalance_strategy == "class_weight"


def build_dataloaders(
    splits: pd.DataFrame,
    images_dir: str | Path,
    train_transform: ImageTransform,
    eval_transform: ImageTransform,
    batch_size: int,
    num_workers: int,
    imbalance_strategy: str,
    generator: torch.Generator,
) -> DataLoaders:
    """Create train/val/test loaders. Augmentation is attached to train only."""
    parts = {s: splits[splits["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    train_ds = LesionDataset(parts["train"], images_dir, train_transform)
    val_ds = LesionDataset(parts["val"], images_dir, eval_transform)
    test_ds = LesionDataset(parts["test"], images_dir, eval_transform)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers else None,
        pin_memory=False,  # not supported on MPS
    )
    if num_workers:
        common["prefetch_factor"] = 4

    sampler = None
    loss_weights = None
    if imbalance_strategy == "oversample":
        sampler = oversampling_sampler(train_ds.labels, generator)
    elif imbalance_strategy == "class_weight":
        loss_weights = class_weights(train_ds.labels)

    # Only the training loader keeps its workers alive between epochs; on an
    # 8 GB machine persistent workers for all three loaders waste memory.
    train = DataLoader(
        train_ds, sampler=sampler, shuffle=sampler is None, generator=generator,
        persistent_workers=num_workers > 0, **common,
    )
    val = DataLoader(val_ds, shuffle=False, **common)
    test = DataLoader(test_ds, shuffle=False, **common)
    return DataLoaders(train=train, val=val, test=test, loss_weights=loss_weights)
