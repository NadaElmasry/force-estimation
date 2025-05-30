# from typing import List, Optional
# import torch
# import numpy as np
# import joblib
# import os

# from torch.utils.data import Dataset
# from torchvision import transforms

# from PIL import Image
# from pathlib import Path
# from sklearn.preprocessing import StandardScaler, MinMaxScaler


# import constants


# class FeatureDataset(Dataset):
#     """
#     Dataset class to handle sequences of robot state features and force targets.
#     """

#     def __init__(self, robot_features: np.ndarray, force_targets: np.ndarray, seq_length: int, feature_scaler_path: Optional[str] = None) -> None:
#         self.robot_features = torch.from_numpy(robot_features).float()
#         self.force_targets = torch.from_numpy(force_targets).float()
#         self.seq_length = seq_length
#         self.num_samples = len(robot_features) - seq_length + 1

#         if feature_scaler_path:
#             assert os.path.isfile(
#                 feature_scaler_path), f"{feature_scaler_path=}"
#             self.feature_scaler = joblib.load(feature_scaler_path)
#             self.robot_features = torch.from_numpy(
#                 self.feature_scaler.transform(robot_features)
#             ).float()
#         else:
#             self.feature_scaler = None

#     def __len__(self) -> int:
#         return self.num_samples

#     def __getitem__(self, idx):
#         start = idx
#         end = start + self.seq_length
#         return {
#             "features": self.robot_features[start:end],
#             "target": self.force_targets[start:end]
#         }


# class SequentialDataset(Dataset):
#     """
#     Dataset class to handle sequences of robot state features and force targets.
#     """

#     def __init__(self,
#                  robot_features_list: List[np.ndarray],
#                  force_targets_list: List[np.ndarray],
#                  seq_length: int,
#                  normalize_targets: bool,
#                  feature_scaler_path: Optional[str] = None,
#                  target_scaler_path: Optional[str] = None) -> None:
#         assert isinstance(robot_features_list, list)
#         assert isinstance(force_targets_list, list)
#         self.robot_features = []
#         self.force_targets = []
#         self.seq_length = seq_length

#         for robot_features, force_targets in zip(robot_features_list, force_targets_list):
#             self.robot_features.append(
#                 torch.from_numpy(robot_features).float())
#             self.force_targets.append(torch.from_numpy(force_targets).float())

#         self.num_samples_per_run = [
#             len(features) - seq_length + 1 for features in self.robot_features]
#         self.cumulative_samples = np.cumsum(self.num_samples_per_run)

#         if feature_scaler_path:
#             assert os.path.isfile(
#                 feature_scaler_path), f"{feature_scaler_path=}"
#             self.feature_scaler = joblib.load(feature_scaler_path)
#             for i in range(len(self.robot_features)):
#                 self.robot_features[i] = torch.from_numpy(
#                     self.feature_scaler.transform(
#                         self.robot_features[i].numpy())
#                 ).float()
#         else:
#             self.feature_scaler = StandardScaler()
#             self._fit_scaler()
#             self._transform_features()
#             joblib.dump(self.feature_scaler, constants.FEATURE_SCALER_FN)

#         if normalize_targets:
#             if target_scaler_path:
#                 assert os.path.isfile(
#                     target_scaler_path), f"{target_scaler_path=}"
#                 self.target_scaler = joblib.load(target_scaler_path)
#                 self._transform_targets()
#             else:
#                 self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
#                 self._fit_target_scaler()
#                 self._transform_targets()
#                 joblib.dump(self.target_scaler, constants.TARGET_SCALER_FN)
#         else:
#             self.target_scaler = None

#     def _fit_scaler(self):
#         all_features = np.concatenate(
#             [features.numpy() for features in self.robot_features])
#         self.feature_scaler.fit(all_features)

#     def _transform_features(self):
#         for i in range(len(self.robot_features)):
#             self.robot_features[i] = torch.from_numpy(
#                 self.feature_scaler.transform(self.robot_features[i].numpy())
#             ).float()

#     def _fit_target_scaler(self):
#         all_targets = np.concatenate([targets.numpy()
#                                      for targets in self.force_targets])
#         self.target_scaler.fit(all_targets)

#     def _transform_targets(self):
#         for i in range(len(self.force_targets)):
#             self.force_targets[i] = torch.from_numpy(
#                 self.target_scaler.transform(self.force_targets[i].numpy())
#             ).float()

#     def __len__(self) -> int:
#         return sum(self.num_samples_per_run)

#     def __getitem__(self, idx):
#         run_idx = np.searchsorted(self.cumulative_samples, idx, side='right')
#         if run_idx == 0:
#             start = idx
#         else:
#             start = idx - self.cumulative_samples[run_idx - 1]

#         end = start + self.seq_length
#         return {
#             "features": self.robot_features[run_idx][start:end],
#             "target": self.force_targets[run_idx][start:end]
#         }


# class AutoEncoderDataset(Dataset):
#     """
#     Dataset class to store left and right images to train an auto encoder
#     """

#     def __init__(self,
#                  img_left_paths: List[str],
#                  img_right_paths: List[str],
#                  path: str,
#                  transforms: Optional[transforms.Compose] = None) -> None:
#         assert len(img_left_paths) == len(img_right_paths)
#         self.img_paths = img_left_paths + img_right_paths
#         assert len(self.img_paths) == len(
#             img_left_paths) + len(img_right_paths)
#         self.transforms = transforms
#         self.path = Path(path)

#     def __len__(self) -> int:
#         return len(self.img_paths)

#     def __getitem__(self, idx):
#         img_path = self.path / self.img_paths[idx]
#         img = Image.open(img_path)

#         assert img.size[0] == img.size[1] == 256, \
#             f"{img.size=}, {img_path=}"

#         if self.transforms:
#             img = self.transforms(img)

#         return {"img": img, "target": img}


# class VisionRobotDataset(Dataset):
#     """
#     Dataset class to store left and right images and robot data.
#     Optionally applies pre-fitted StandardScaler to robot features and MinMaxScaler to force targets.
#     """

#     def __init__(self,
#                  robot_features: np.ndarray,
#                  force_targets: np.ndarray,
#                  img_left_paths: List[str],
#                  img_right_paths: List[str],
#                  path: str,
#                  img_transforms: Optional[transforms.Compose] = None,
#                  feature_scaler_path: Optional[str] = None,
#                  target_scaler_path: Optional[str] = None) -> None:
#         self.num_samples, self.num_robot_features = robot_features.shape
#         assert force_targets.shape[0] == self.num_samples, \
#             f"force_labels size: \
#             {force_targets.shape} does not match samples nr: {self.num_samples}"
#         assert len(img_left_paths) == self.num_samples
#         assert len(img_right_paths) == self.num_samples

#         self.robot_features = torch.from_numpy(robot_features).float()
#         self.force_targets = torch.from_numpy(force_targets).float()
#         self.img_left_paths = img_left_paths
#         self.img_right_paths = img_right_paths
#         self.transforms = img_transforms
#         self.path = Path(path)

#         if feature_scaler_path:
#             assert os.path.isfile(
#                 feature_scaler_path), f"{feature_scaler_path=}"
#             self.feature_scaler = joblib.load(feature_scaler_path)
#             self.robot_features = torch.from_numpy(
#                 self.feature_scaler.transform(robot_features)
#             ).float()
#         else:
#             self.feature_scaler = None

#         if target_scaler_path:
#             assert os.path.isfile(
#                 target_scaler_path), f"{target_scaler_path=}"
#             self.target_scaler = joblib.load(target_scaler_path)
#             self.force_targets = torch.from_numpy(
#                 self.target_scaler.transform(force_targets)
#             ).float()
#         else:
#             self.target_scaler = None

#     def __len__(self) -> int:
#         return self.num_samples

#     def __getitem__(self, idx):
#         img_left_path = self.path / self.img_left_paths[idx]
#         img_right_path = self.path / self.img_right_paths[idx]
#         img_left = Image.open(img_left_path)
#         img_right = Image.open(img_right_path)

#         assert img_left.size[0] == img_left.size[1] == 256, \
#             f"{img_left.size=}, {img_left_path=}"
#         assert img_right.size[0] == img_right.size[1] == 256, \
#             f"{img_left.size=}, {img_right.size=}"

#         if self.transforms:
#             img_left = self.transforms(img_left)
#             img_right = self.transforms(img_right)

#         return {"img_left": img_left, "img_right": img_right, "features": self.robot_features[idx], "target": self.force_targets[idx]}



# -----------------------------------------------------------------------------
#  datasets_revised.py  ·  May 2025
#  End‑to‑end safe dataset classes used by the training script:
#    • SequentialDataset  (state sequences for s‑mode)
#    • VisionRobotDataset (optional recurrency, custom loader path)
# -----------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from utils import custom_collate_fn, fit_transform
from __future__ import annotations
from typing import List, Optional, Tuple
from pathlib import Path
import joblib, numpy as np, torch
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset
from torchvision.transforms import Compose

# -----------------------------------------------------------------------------
#  SequentialDataset (robot state + force, only state matters for training)  --
# -----------------------------------------------------------------------------
class SequentialDataset(Dataset):
    """Return sliding windows of length **T** over robot‑state + force arrays.

    Each **__getitem__** returns:
        {"features": (T, S), "target": (T, 3)}

    • `robot_features_list` and `force_targets_list` are lists of Numpy arrays
      with shapes (Nᵢ, S) and (Nᵢ, 3) respectively.
    • The dataset does its own StandardScaling across *all* runs unless you pass
      a `feature_scaler_path`.
    """

    def __init__(
        self,
        *,
        robot_features_list: List[np.ndarray],
        force_targets_list: List[np.ndarray],
        seq_length: int,
        normalize_targets: bool = False,
        feature_scaler_path: Optional[str] = None,
        target_scaler_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert len(robot_features_list) == len(force_targets_list), "mismatched runs"
        self.seq_length = seq_length

        # -------- Feature scaling --------
        if feature_scaler_path and Path(feature_scaler_path).is_file():
            self.feature_scaler: StandardScaler = joblib.load(feature_scaler_path)
            self.robot_features = [
                torch.from_numpy(self.feature_scaler.transform(r)).float()
                for r in robot_features_list
            ]
        else:
            self.feature_scaler = StandardScaler()
            self.robot_features = _fit_transform(robot_features_list, self.feature_scaler)
            if feature_scaler_path:
                Path(feature_scaler_path).parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(self.feature_scaler, feature_scaler_path)

        # -------- Target scaling (optional) --------
        if normalize_targets:
            if target_scaler_path and Path(target_scaler_path).is_file():
                self.target_scaler: MinMaxScaler = joblib.load(target_scaler_path)
                self.force_targets = [
                    torch.from_numpy(self.target_scaler.transform(f)).float()
                    for f in force_targets_list
                ]
            else:
                self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
                self.force_targets = _fit_transform(force_targets_list, self.target_scaler)
                if target_scaler_path:
                    Path(target_scaler_path).parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(self.target_scaler, target_scaler_path)
        else:
            self.target_scaler = None
            self.force_targets = [torch.from_numpy(f).float() for f in force_targets_list]

        self.num_samples_per_run = [len(f) - seq_length + 1 for f in self.robot_features]
        self.cum_samples = np.cumsum(self.num_samples_per_run)

    # ------------------------------------------------------------------  magic
    def __len__(self) -> int:  # noqa: D401,E501
        return int(self.cum_samples[-1])

    def __getitem__(self, idx):  # noqa: D401,E501
        run_idx = int(np.searchsorted(self.cum_samples, idx, side="right"))
        start = idx if run_idx == 0 else idx - int(self.cum_samples[run_idx - 1])
        end = start + self.seq_length
        return {
            "features": self.robot_features[run_idx][start:end],  # (T, S)
            "target": self.force_targets[run_idx][start:end],     # (T, 3)
        }


# -----------------------------------------------------------------------------
#  VisionRobotDataset (stereo or mono images + state + force)  -----------------
# -----------------------------------------------------------------------------
class VisionRobotDataset(Dataset):
    """Return per‑frame (or per‑sequence) samples for vs / v modes.

    If `seq_length > 1` the dataset outputs windows `(T, C, H, W)` and
    `(T, S)`; otherwise single frames.  The training script’s custom collate_fn
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
        assert len(robot_features) == len(img_right_paths) == len(force_targets), "sample mismatch"
        self.seq_length = seq_length
        self.root = Path(path)
        self.transforms = img_transforms

        # --- scalers ---------------------------------------------------------
        if feature_scaler_path and Path(feature_scaler_path).is_file():
            fscaler: StandardScaler = joblib.load(feature_scaler_path)
            self.robot_features = torch.from_numpy(fscaler.transform(robot_features)).float()
        else:
            self.robot_features = torch.from_numpy(robot_features).float()

        if target_scaler_path and Path(target_scaler_path).is_file():
            tscaler: MinMaxScaler = joblib.load(target_scaler_path)
            self.force_targets = torch.from_numpy(tscaler.transform(force_targets)).float()
        else:
            self.force_targets = torch.from_numpy(force_targets).float()

        self.img_left_paths = img_left_paths
        self.img_right_paths = img_right_paths
        self.N = len(img_right_paths)

    # ------------------------------------------------------------------  magic
    def __len__(self) -> int:  # noqa: D401,E501
        return self.N - self.seq_length + 1 if self.seq_length > 1 else self.N

    def _load_pair(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        l = Image.open(self.root / self.img_left_paths[idx]).convert("RGB")
        r = Image.open(self.root / self.img_right_paths[idx]).convert("RGB")
        return l, r

    def __getitem__(self, idx):  # noqa: D401,E501
        if self.seq_length == 1:
            img_l, img_r = self._load_pair(idx)
            if self.transforms:
                img_l, img_r = self.transforms(img_l), self.transforms(img_r)
            return {
                "img_left": img_l,
                "img_right": img_r,
                "features": self.robot_features[idx],
                "target": self.force_targets[idx],
            }

        # -------- sequence window (T frames) --------
        idxs = range(idx, idx + self.seq_length)
        imgs_l, imgs_r = zip(*(self._load_pair(i) for i in idxs))
        if self.transforms:
            imgs_l = [self.transforms(im) for im in imgs_l]
            imgs_r = [self.transforms(im) for im in imgs_r]
        imgs_l_tensor = torch.stack(imgs_l)  # (T, C, H, W)
        imgs_r_tensor = torch.stack(imgs_r)  # (T, C, H, W)
        return {
            "img_left": imgs_l_tensor,
            "img_right": imgs_r_tensor,
            "features": self.robot_features[idx : idx + self.seq_length],   # (T, S)
            "target": self.force_targets[idx : idx + self.seq_length],      # (T, 3)
        }






class VisionRobotDataset(Dataset):
    """
    Single‑frame or sliding‑window dataset of images + robot state + forces.
    """

    def __init__(
        self,
        *,
        robot_features: np.ndarray,
        force_targets: np.ndarray,
        img_left_paths: List[str],
        img_right_paths: List[str],
        path: str,
        img_transforms: Optional[Compose] = None,
        seq_length: int = 1,
        feature_scaler_path: Optional[str] = None,
        target_scaler_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.root = Path(path)
        self.seq_length = seq_length
        self.transforms = img_transforms

        # ------- scalers -------------------------------------------------
        if feature_scaler_path and Path(feature_scaler_path).is_file():
            fscaler: StandardScaler = joblib.load(feature_scaler_path)
            robot_features = fscaler.transform(robot_features)
        if target_scaler_path and Path(target_scaler_path).is_file():
            tscaler: MinMaxScaler = joblib.load(target_scaler_path)
            force_targets = tscaler.transform(force_targets)

        self.robot_features = torch.from_numpy(robot_features).float()
        self.force_targets = torch.from_numpy(force_targets).float()
        self.img_left_paths, self.img_right_paths = img_left_paths, img_right_paths
        self.N = len(img_right_paths)
        assert self.N == len(robot_features) == len(force_targets)

    # ------------------------------ magic ------------------------------
    def __len__(self):                # sliding‑window‑aware length
        return self.N - self.seq_length + 1 if self.seq_length > 1 else self.N

    def _load_pair(self, i: int) -> Tuple[Image.Image, Image.Image]:
        L = Image.open(self.root/self.img_left_paths[i]).convert("RGB")
        R = Image.open(self.root/self.img_right_paths[i]).convert("RGB")
        return L, R

    def __getitem__(self, idx):
        if self.seq_length == 1:
            L, R = self._load_pair(idx)
            if self.transforms: L, R = self.transforms(L), self.transforms(R)
            return {"img_left": L, "img_right": R,
                    "features": self.robot_features[idx],
                    "target":   self.force_targets[idx]}
        # ---- sliding window ------------------------------------------
        sl = slice(idx, idx+self.seq_length)
        imgs_LR = [self._load_pair(i) for i in range(idx, idx+self.seq_length)]
        Ls, Rs = zip(*imgs_LR)
        if self.transforms:
            Ls, Rs = [self.transforms(i) for i in Ls], [self.transforms(i) for i in Rs]
        return {"img_left":  torch.stack(Ls),
                "img_right": torch.stack(Rs),
                "features":  self.robot_features[sl],
                "target":    self.force_targets[sl]}
