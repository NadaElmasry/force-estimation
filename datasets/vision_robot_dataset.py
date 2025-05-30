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
import random








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





# -----------------------------------------------------------------------------
#  VisionRobotDataset (images + state + force)  -----------------
# -----------------------------------------------------------------------------
class VisionRobotDataset(Dataset):
    """Return per‑frame (or per‑sequence) samples for vs / v modes.

    If `seq_length > 1` the dataset outputs windows `(T, C, H, W)` and
    `(T, S)`; otherwise single frames.  The training script's custom collate_fn
    will stack them into `(B, T, …)`.
    """

    def __init__(
        self,
        *,
        robot_features: np.ndarray,        # (N, S)
        force_targets: np.ndarray,         # (N, 3)
        img_left_paths: List[str],
        img_right_paths: List[str],
        path: str,
        img_transforms: Optional[Compose] = None,
        seq_length: int = 1,
        feature_scaler_path: Optional[str] = None,
        target_scaler_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        # Validate input shapes
        print(f"\nInitializing VisionRobotDataset:")
        print(f"- Robot features shape: {robot_features.shape}")
        print(f"- Force targets shape: {force_targets.shape}")
        print(f"- Number of left images: {len(img_left_paths)}")
        print(f"- Number of right images: {len(img_right_paths)}")
        print(f"- Sequence length: {seq_length}")
        
        assert len(robot_features) == len(img_right_paths) == len(force_targets), \
            f"Sample count mismatch: features={len(robot_features)}, " \
            f"right_imgs={len(img_right_paths)}, forces={len(force_targets)}"
        
        assert force_targets.shape[1] == 3, \
            f"Force targets should have 3 components (x,y,z), got {force_targets.shape[1]}"
        
        self.seq_length = seq_length
        self.root = Path(path)
        self.transforms = img_transforms

        # --- scalers ---------------------------------------------------------
        if feature_scaler_path and Path(feature_scaler_path).is_file():
            print(f"Loading feature scaler from {feature_scaler_path}")
            fscaler: StandardScaler = joblib.load(feature_scaler_path)
            self.robot_features = torch.from_numpy(fscaler.transform(robot_features)).float()
            print(f"- Scaled features range: [{self.robot_features.min():.3f}, {self.robot_features.max():.3f}]")
        else:
            print("No feature scaler provided, using raw features")
            self.robot_features = torch.from_numpy(robot_features).float()

        if target_scaler_path and Path(target_scaler_path).is_file():
            print(f"Loading target scaler from {target_scaler_path}")
            tscaler: MinMaxScaler = joblib.load(target_scaler_path)
            self.force_targets = torch.from_numpy(tscaler.transform(force_targets)).float()
            print(f"- Scaled targets range: [{self.force_targets.min():.3f}, {self.force_targets.max():.3f}]")
        else:
            print("No target scaler provided, using raw targets")
            self.force_targets = torch.from_numpy(force_targets).float()

        self.img_left_paths = img_left_paths
        self.img_right_paths = img_right_paths
        self.N = len(img_right_paths)

        # Validate image paths
        self._validate_image_paths()
        print(f"Dataset initialization complete. Total samples: {len(self)}\n")

    def _validate_image_paths(self) -> None:
        """Validate that image paths exist and have correct dimensions."""
        print("Validating image paths...")
        for i, (left, right) in enumerate(zip(self.img_left_paths[:5], self.img_right_paths[:5])):
            left_path = self.root / left
            right_path = self.root / right
            assert left_path.exists(), f"Left image not found: {left_path}"
            assert right_path.exists(), f"Right image not found: {right_path}"
            
            if i == 0:  # Check dimensions for first pair
                l_img = Image.open(left_path)
                r_img = Image.open(right_path)
                print(f"- Image dimensions: Left={l_img.size}, Right={r_img.size}")
                l_img.close()
                r_img.close()
        print("Image path validation complete")

    def __len__(self) -> int:
        return self.N - self.seq_length + 1 if self.seq_length > 1 else self.N

    def _load_pair(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        try:
            l = Image.open(self.root / self.img_left_paths[idx]).convert("RGB")
            r = Image.open(self.root / self.img_right_paths[idx]).convert("RGB")
            return l, r
        except Exception as e:
            print(f"Error loading images at index {idx}:")
            print(f"Left path: {self.root / self.img_left_paths[idx]}")
            print(f"Right path: {self.root / self.img_right_paths[idx]}")
            raise e

    def __getitem__(self, idx):
        if self.seq_length == 1:
            img_l, img_r = self._load_pair(idx)
            if self.transforms:
                img_l, img_r = self.transforms(img_l), self.transforms(img_r)
                
            sample = {
                "img_left": img_l,
                "img_right": img_r,
                "features": self.robot_features[idx],
                "target": self.force_targets[idx],
            }
            
            # Log shapes periodically (every 1000th sample)
            if idx % 1000 == 0:
                shapes = {
                    "img_left": tuple(img_l.shape),
                    "img_right": tuple(img_r.shape),
                    "features": tuple(sample["features"].shape),
                    "target": tuple(sample["target"].shape)
                }
                print(f"\nSample {idx} shapes:", shapes)
            
            return sample

        # -------- sequence window (T frames) --------
        idxs = range(idx, idx + self.seq_length)
        imgs_l, imgs_r = zip(*(self._load_pair(i) for i in idxs))
        if self.transforms:
            imgs_l = [self.transforms(im) for im in imgs_l]
            imgs_r = [self.transforms(im) for im in imgs_r]
        imgs_l_tensor = torch.stack(imgs_l)  # (T, C, H, W)
        imgs_r_tensor = torch.stack(imgs_r)  # (T, C, H, W)
        
        sample = {
            "img_left": imgs_l_tensor,
            "img_right": imgs_r_tensor,
            "features": self.robot_features[idx : idx + self.seq_length],   # (T, S)
            "target": self.force_targets[idx : idx + self.seq_length],      # (T, 3)
        }
        
        # Log shapes periodically for sequences
        if idx % 1000 == 0:
            shapes = {
                "img_left": tuple(sample["img_left"].shape),
                "img_right": tuple(sample["img_right"].shape),
                "features": tuple(sample["features"].shape),
                "target": tuple(sample["target"].shape)
            }
            print(f"\nSequence {idx} shapes:", shapes)
        
        return sample

def custom_collate_fn(batch):
    """
    Stacks dict fields to a single batch.
    Works for both single‑frame and sequence inputs.

    Returns
    -------
    dict with keys:
        img_right  – (B, T, C, H, W) or (B, C, H, W)
        forces     – (B, 3)   (last step if sequence)
        robot_state– (B, T, S) or (B, S)
    """
    try:
        imgs = torch.stack([b["img_right"] for b in batch])
        forces_lst = [b["target"] for b in batch]          # (T,3) or (3,)
        
        if forces_lst[0].dim() == 2:                       # sequence → pick last
            forces = torch.stack([f[-1] for f in forces_lst])
        else:
            forces = torch.stack(forces_lst)
            
        robot_state = torch.stack([b["features"] for b in batch])
        
        # Log batch shapes periodically
        if random.random() < 0.01:  # ~1% of batches
            print("\nBatch shapes:")
            print(f"- Images: {imgs.shape}")
            print(f"- Forces: {forces.shape}")
            print(f"- Robot state: {robot_state.shape}")
        
        return {
            "img_right": imgs,
            "forces": forces,
            "robot_state": robot_state
        }
    except Exception as e:
        print("\nError in collate_fn:")
        print("Batch contents:")
        for i, b in enumerate(batch):
            print(f"Item {i} shapes:")
            for k, v in b.items():
                print(f"  {k}: {tuple(v.shape) if torch.is_tensor(v) else type(v)}")
        raise e

