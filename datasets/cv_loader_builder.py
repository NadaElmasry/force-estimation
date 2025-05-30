# datasets/cv_loader_builder.py
from __future__ import annotations
from typing import Callable, List, Tuple, Optional, Sequence
from pathlib import Path
import joblib
from PIL import Image
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
from torchvision.transforms import Compose








def build_cv_loaders(
    dataset: torch.utils.data.Dataset,
    k_folds: int = 5,
    batch_size: int = 32,
    collate_fn: Optional[Callable] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    random_state: int = 0,
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Splits <dataset> into `k_folds` folds and returns a list:
        loaders[i] == (train_loader_i, val_loader_i)
    """
    if k_folds < 2:
        raise ValueError("k_folds must be >= 2")

    kf = KFold(
        n_splits=k_folds,
        shuffle=True,
        random_state=random_state
    )

    indices: Sequence[int] = np.arange(len(dataset))
    pairs: List[Tuple[DataLoader, DataLoader]] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
        tr_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        tr_loader = DataLoader(
            tr_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            collate_fn=collate_fn
        )
        pairs.append((tr_loader, val_loader))

    return pairs



