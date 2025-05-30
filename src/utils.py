from __future__ import annotations

import datetime as _dt
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import constants  # must define FEATURE_COLUMNS, IMAGE_COLUMNS, TARGET_COLUMNS, TIME_COLUMN
import torch
from __future__ import annotations
from typing import Callable, List, Optional, Sequence, Tuple
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold


def build_cv_loaders(
    dataset: torch.utils.data.Dataset,
    k_folds: int = 5,
    batch_size: int = 32,
    collate_fn: Optional[Callable] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 0
) -> List[Tuple[DataLoader, DataLoader]]:
    """Return [(train_loader_i, val_loader_i)] * k_folds."""
    kf = KFold(k_folds, shuffle=True, random_state=seed)
    indices: Sequence[int] = np.arange(len(dataset))
    out: list = []
    for tr_idx, val_idx in kf.split(indices):
        tr, val = Subset(dataset, tr_idx), Subset(dataset, val_idx)
        out.append((
            DataLoader(tr,  batch_size, True, num_workers, pin_memory, collate_fn),
            DataLoader(val, batch_size, False, num_workers, pin_memory, collate_fn)
        ))
    return out
# -----------------------------------------------------------------------------
#  -----------------------  Checkpoint helpers  --------------------------------
# -----------------------------------------------------------------------------

def save_checkpoint(save_path: Path, model_state, is_best: bool, filename: str = "checkpoint.pth") -> None:
    """Write *filename* to *save_path*; copy to *model_best.pth* if *is_best*."""
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, save_path / filename)
    if is_best:
        shutil.copyfile(save_path / filename, save_path / "model_best.pth")


def create_saving_dir(
    root: Path,
    experiment_name: str | Path,
    *,
    architecture: str,
    dataset: str,
    recurrency: bool,
    att_type: Optional[str] = None,
    occ_param: Optional[str] = None,
) -> Path:
    """Replicates original folder structure but guarantees uniqueness via timestamp."""
    timestamp = _dt.datetime.now().strftime("%m-%d_%H-%M-%S")
    arch_dir = ("r" if recurrency else "") + architecture + (att_type or "")
    parts = [root, dataset, arch_dir]
    if occ_param:
        parts.append(occ_param)
    parts.extend([experiment_name, timestamp])
    return Path(*map(str, parts))


# -----------------------------------------------------------------------------
#  -----------------------  Excel → numpy loader  ------------------------------
# -----------------------------------------------------------------------------

def _get_img_paths(col: str, df: pd.DataFrame) -> List[str]:
    paths: List[str] = []
    for server_path in df[col].astype(str).tolist():
        parts = server_path.split("/")[-3:]  # keep last 3 levels
        paths.append("/".join(parts))
    return paths


def _set_derivative(df: pd.DataFrame, src: str, dst: str) -> None:
    dt = df[constants.TIME_COLUMN].diff().replace(0, np.nan)
    df[dst] = df[src].diff().div(dt).fillna(0.0)


def _calculate_velocity(df: pd.DataFrame) -> pd.DataFrame:
    for nr in (1, 2):
        for axis in "xyz":
            _set_derivative(df, f"PSM{nr}_ee_{axis}", f"PSM{nr}_ee_v_{axis}")
        for j in range(1, 7):
            _set_derivative(df, f"PSM{nr}_joint_{j}", f"PSM{nr}_joint_{j}_v")
        _set_derivative(df, f"PSM{nr}_jaw_angle", f"PSM{nr}_jaw_angle_v")
    return df


def _calculate_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    for nr in (1, 2):
        for axis in "xyz":
            _set_derivative(df, f"PSM{nr}_ee_v_{axis}", f"PSM{nr}_ee_a_{axis}")
        for j in range(1, 7):
            _set_derivative(df, f"PSM{nr}_joint_{j}_v", f"PSM{nr}_joint_{j}_a")
        _set_derivative(df, f"PSM{nr}_jaw_angle_v", f"PSM{nr}_jaw_angle_a")
    return df


# -----------------------------------------------------------------------------
#  Main disk‑loading helper -----------------------------------------------------
# -----------------------------------------------------------------------------

def load_dataset(
    *,
    path: str | Path,
    force_policy_runs: List[int],
    no_force_policy_runs: List[int],
    sequential: bool,
    create_plots: bool = False,
    crop_runs: bool = True,
    use_acceleration: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, List[str], List[str]],
    Tuple[List[np.ndarray], List[np.ndarray], List[str], List[str]],
]:
    """Aggregate multiple Excel rollouts into feature/target arrays + image paths.

    Parameters
    ----------
    sequential : bool
        • False ⇒ concatenate *all* runs into single `(N, S)` and `(N, 3)` arrays.
        • True  ⇒ return lists `[run₁, run₂, …]` (used by `SequentialDataset`).
    """

    root = Path(path)
    assert root.is_dir(), f"{root} is not a directory"
    roll_out_dir = root / "roll_out"
    img_root = root / "images"
    assert roll_out_dir.is_dir(), "missing roll_out dir"
    assert img_root.is_dir(), "missing images dir"

    runs: dict[str, List[int]] = {
        "force_policy": force_policy_runs,
        "no_force_policy": no_force_policy_runs,
    }

    X_runs: List[np.ndarray] = []
    y_runs: List[np.ndarray] = []
    left_paths: List[str] = []
    right_paths: List[str] = []

    use_cols = (
        constants.FEATURE_COLUMNS
        + constants.IMAGE_COLUMNS
        + constants.TARGET_COLUMNS
        + [constants.TIME_COLUMN]
    )

    for policy, run_list in runs.items():
        for run in run_list:
            excel_path = roll_out_dir / constants.EXCEL_FILE_NAMES[policy][run]
            df = pd.read_excel(excel_path, usecols=use_cols)
            df = _calculate_velocity(df)
            if use_acceleration:
                df = _calculate_acceleration(df)

            # Drop the time column after derivatives
            df = df.drop(columns=[constants.TIME_COLUMN])
            X_cols = (
                constants.FEATURE_COLUMNS + constants.VELOCITY_COLUMNS + (constants.ACCELERATION_COLUMNS if use_acceleration else [])
            )
            if crop_runs:
                for start, end in constants.START_END_TIMES[policy][run]:
                    window = df.iloc[start:end]
                    _append_window(window, X_cols, X_runs, y_runs, left_paths, right_paths, policy, run)
            else:
                _append_window(df, X_cols, X_runs, y_runs, left_paths, right_paths, policy, run)

    if sequential:
        return X_runs, y_runs, left_paths, right_paths
    return np.concatenate(X_runs), np.concatenate(y_runs), left_paths, right_paths


# -----------------------------------------------------------------------------
#  Helper to slice + append
# -----------------------------------------------------------------------------

def _append_window(
    df: pd.DataFrame,
    X_cols: List[str],
    X_runs: List[np.ndarray],
    y_runs: List[np.ndarray],
    left_paths: List[str],
    right_paths: List[str],
    policy: str,
    run: int,
) -> None:
    X_runs.append(df[X_cols].to_numpy())
    y_runs.append(df[constants.TARGET_COLUMNS].to_numpy())
    left_paths += _get_img_paths("ZED Camera Left", df)
    right_paths += _get_img_paths("ZED Camera Right", df)


# -----------------------------------------------------------------------------
#  Scaling utility used at evaluation time ------------------------------------
# -----------------------------------------------------------------------------

def apply_scaling_to_datasets(
    train_ds,
    val_ds,
    *,
    normalize_targets: bool = False,
    feature_scaler_path: str = "scalers/feature_scaler.gz",
    target_scaler_path: str = "scalers/target_scaler.gz",
):
    """Mutate dataset objects *in‑place* to apply the same scaling."""
    feature_scaler = StandardScaler().fit(train_ds.robot_features.numpy())
    train_ds.robot_features = torch.from_numpy(feature_scaler.transform(train_ds.robot_features.numpy())).float()
    val_ds.robot_features = torch.from_numpy(feature_scaler.transform(val_ds.robot_features.numpy())).float()
    Path(feature_scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_scaler, feature_scaler_path)

    if normalize_targets:
        target_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_ds.force_targets.numpy())
        train_ds.force_targets = torch.from_numpy(target_scaler.transform(train_ds.force_targets.numpy())).float()
        val_ds.force_targets = torch.from_numpy(target_scaler.transform(val_ds.force_targets.numpy())).float()
        Path(target_scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(target_scaler, target_scaler_path)





def custom_collate_fn(batch):
    imgs   = torch.stack([b["img_right"] for b in batch])
    forces = torch.stack([(t[-1] if t.dim()==2 else t) for t in (b["target"] for b in batch)])
    robot  = torch.stack([b["features"] for b in batch])
    return {"img_right": imgs, "forces": forces, "robot_state": robot}


def fit_transform(arrs: List[np.ndarray], scaler) -> List[torch.Tensor]:
    stacked = np.concatenate(arrs)
    scaler.fit(stacked)
    out: List[torch.Tensor] = []
    for a in arrs:
        out.append(torch.from_numpy(scaler.transform(a)).float())
    return out


# ────────────────────────  transforms  ───────────────────────────────────────
def val_tf() -> augmentations.Compose:
    norm = augmentations.Normalize([0.45]*3, [0.225]*3)
    return augmentations.Compose([
        augmentations.CentreCrop(),
        augmentations.SquareResize(),
        augmentations.ArrayToTensor(),
        norm
    ])
