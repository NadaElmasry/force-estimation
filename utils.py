# from __future__ import division
# import shutil
# import torch
# from torch.utils.data import Dataset
# from pathlib import Path
# import numpy as np
# import torch.nn as nn
# import datetime
# import os
# from typing import List, Optional, Tuple, Union
# import pandas as pd
# import argparse
# import joblib

# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import matplotlib.pyplot as plt

# from torch.utils.data import DataLoader
# from torchvision import transforms
# from sklearn.model_selection import KFold
# from dataset import VisionRobotDataset, SequentialDataset
# from models.vision_robot_net import VRNConfig, STATE_MAPPING
# from models.robot_state_transformer import TransformerConfig, EncoderState
# import constants

# def save_checkpoint(save_path: Path, model_state, is_best:bool, filename='checkpoint.pth.tar'):
#     torch.save(model_state, save_path/'{}'.format(filename))

#     if is_best:
#         shutil.copyfile(save_path/'{}'.format(filename),
#                         save_path/'model_best.pth.tar')

# def none_or_str(value):
#     if value=="None":
#         return None
#     return value

# def create_saving_dir(root: Path, 
#                       experiment_name: str, 
#                       architecture: str,
#                       dataset: str, 
#                       recurrency: bool, 
#                       att_type: str = None, 
#                       occ_param: str = None):
    
#     timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")

#     if recurrency:
#         architecture = "r" + architecture
    
#     if att_type is not None:
#         architecture = architecture + att_type.lower()
    
#     if architecture == "fc":
#         if occ_param is not None:
#             save_path = root/"{}".format(dataset)/architecture/occ_param/experiment_name/timestamp
#         else:
#             save_path = root/"{}".format(dataset)/architecture/experiment_name/timestamp
#     else:
#         if occ_param is not None:
#             save_path = root/"{}".format(dataset)/architecture/occ_param/experiment_name/timestamp
#         else:
#             save_path = root/"{}".format(dataset)/architecture/experiment_name/timestamp
    
#     return save_path






# def get_img_paths(cam: str, excel_df: pd.DataFrame) -> List[str]:
#     assert cam in ["Left", "Right"], f"Invalid {cam}"

#     col_name = f"ZED Camera {cam}"
#     img_paths = []
#     for server_path in excel_df[col_name].to_list():
#         dirs = server_path.split("/")
#         new_path = "/".join(dirs[-3:])
#         img_paths.append(new_path)

#     return img_paths


# def load_data(runs: dict[str, List[int]],
#               data_dir: str,
#               sequential: bool,
#               create_plots: bool = False,
#               crop_runs: bool = True,
#               use_acceleration: bool = False) -> Union[Tuple[np.ndarray,
#                                                              np.ndarray,
#                                                              List[str],
#                                                              List[str]],
#                                                        Tuple[List[np.ndarray],
#                                                              List[np.ndarray],
#                                                              List[str],
#                                                              List[str]]]:
#     assert isinstance(runs, dict), f"{runs=}"

#     all_X = []
#     all_y = []
#     all_img_left_paths = []
#     all_img_right_paths = []

#     relevant_cols = constants.FEATURE_COLUMS + constants.IMAGE_COLUMS + \
#         constants.TARGET_COLUMNS + constants.TIME_COLUMN

#     for policy, runs in runs.items():
#         for run in runs:
#             print(f"Loading data for run {run} of policy {policy}")
#             excel_file_name = constants.EXCEL_FILE_NAMES[policy][run]
#             excel_file_path = os.path.join(data_dir, excel_file_name)
#             excel_df = pd.read_excel(excel_file_path, usecols=relevant_cols)
#             excel_df = calculate_velocity(excel_df)
#             if use_acceleration:
#                 excel_df = calculate_acceleration(excel_df)

#             excel_df = excel_df.drop(constants.TIME_COLUMN, axis=1)
#             X_cols = constants.FEATURE_COLUMS + constants.VELOCITY_COLUMNS
#             if use_acceleration:
#                 X_cols += constants.ACCELERATION_COLUMNS

#             if create_plots:
#                 forces_arr = excel_df[constants.TARGET_COLUMNS].to_numpy()
#                 plot_forces(forces_arr, run_nr=run, policy=policy, pdf=False)

#             if crop_runs:
#                 for times in constants.START_END_TIMES[policy][run]:
#                     start = times[0]
#                     end = times[1]
#                     actual_data_df = excel_df.iloc[start:end, :]

#                     X = actual_data_df[X_cols].to_numpy()
#                     y = actual_data_df[constants.TARGET_COLUMNS].to_numpy()
#                     img_left_paths = get_img_paths("Left", actual_data_df)
#                     img_right_paths = get_img_paths("Right", actual_data_df)

#                     all_X.append(X)
#                     all_y.append(y)
#                     all_img_left_paths += img_left_paths
#                     all_img_right_paths += img_right_paths
#             else:
#                 X = excel_df[X_cols].to_numpy()
#                 y = excel_df[constants.TARGET_COLUMNS].to_numpy()
#                 img_left_paths = get_img_paths("Left", excel_df)
#                 img_right_paths = get_img_paths("Right", excel_df)

#                 all_X.append(X)
#                 all_y.append(y)
#                 all_img_left_paths += img_left_paths
#                 all_img_right_paths += img_right_paths

#     if not sequential:
#         all_X = np.concatenate(all_X, axis=0)
#         all_y = np.concatenate(all_y, axis=0)

#     return all_X, all_y, all_img_left_paths, all_img_right_paths


# def calculate_acceleration(df: pd.DataFrame) -> pd.DataFrame:
#     # Ensure velocity is calculated first
#     assert all(column in df.columns for column in constants.VELOCITY_COLUMNS)

#     for nr in [1, 2]:
#         for axis in ['x', 'y', 'z']:
#             velocity_col = f'PSM{nr}_ee_v_{axis}'
#             acceleration_col = f'PSM{nr}_ee_a_{axis}'
#             set_acceleration(df, velocity_col, acceleration_col)
#         for joint in range(1, 7):
#             velocity_col = f'PSM{nr}_joint_{joint}_v'
#             acceleration_col = f'PSM{nr}_joint_{joint}_a'
#             set_acceleration(df, velocity_col, acceleration_col)
#         velocity_col = f'PSM{nr}_jaw_angle_v'
#         acceleration_col = f'PSM{nr}_jaw_angle_a'
#         set_acceleration(df, velocity_col, acceleration_col)

#     return df


# def set_acceleration(df: pd.DataFrame, velocity_col: str, acceleration_col: str):
#     df[acceleration_col] = df[velocity_col].diff() / df["Time (Seconds)"].diff()
#     df.loc[df.index[0], acceleration_col] = 0
#     assert len(df[acceleration_col]) == len(df[velocity_col])
#     assert df[acceleration_col].isnull().sum() == 0


# def calculate_velocity(df: pd.DataFrame) -> pd.DataFrame:
#     for nr in [1, 2]:
#         for axis in ['x', 'y', 'z']:
#             position_col = f'PSM{nr}_ee_{axis}'
#             velocity_col = f'PSM{nr}_ee_v_{axis}'
#             set_velocity(df, position_col, velocity_col)
#         for joint in range(1, 7):
#             position_col = f'PSM{nr}_joint_{joint}'
#             velocity_col = f'PSM{nr}_joint_{joint}_v'
#             set_velocity(df, position_col, velocity_col)
#         position_col = f'PSM{nr}_jaw_angle'
#         velocity_col = f'PSM{nr}_jaw_angle_v'
#         set_velocity(df, position_col, velocity_col)

#     return df


# def set_velocity(df: pd.DataFrame, position_col: str, velocity_col: str):
#     df[velocity_col] = df[position_col].diff() / \
#         df["Time (Seconds)"].diff()
#     # first element is nan, as the velocity cannot be computed
#     df.loc[df.index[0], velocity_col] = 0
#     assert len(df[position_col]) == len(df[velocity_col])
#     assert df[velocity_col].isnull().sum() == 0


# def load_dataset(path: str,
#                  force_policy_runs: List[int],
#                  no_force_policy_runs: List[int],
#                  sequential: bool,
#                  create_plots: bool = False,
#                  crop_runs: bool = True,
#                  use_acceleration: bool = False) -> Tuple[np.ndarray,
#                                                           np.ndarray,
#                                                           List[str],
#                                                           List[str]]:
#     assert os.path.isdir(path), f"{path} is not a directory"
#     assert os.path.exists(os.path.join(path, "images")), \
#         f"{path} does not contain an images directory"
#     assert os.path.exists(os.path.join(path, "roll_out")), \
#         f"{path} does not contain a roll out directory"

#     roll_out_dir = os.path.join(path, "roll_out")

#     runs = {
#         "force_policy": force_policy_runs,
#         "no_force_policy": no_force_policy_runs
#     }

#     return load_data(runs, roll_out_dir, sequential=sequential, create_plots=create_plots, crop_runs=crop_runs, use_acceleration=use_acceleration)


# def apply_scaling_to_datasets(train_dataset: VisionRobotDataset,
#                               test_dataset: VisionRobotDataset,
#                               normalize_targets: Optional[bool] = False) -> None:
#     feature_scaler = StandardScaler()

#     feature_scaler.fit(train_dataset.robot_features.numpy())

#     train_dataset.robot_features = torch.from_numpy(
#         feature_scaler.transform(train_dataset.robot_features.numpy())).float()
#     test_dataset.robot_features = torch.from_numpy(
#         feature_scaler.transform(test_dataset.robot_features.numpy())).float()

#     # Save scaler to file to load it during eval
#     joblib.dump(feature_scaler, constants.FEATURE_SCALER_FN)

#     if normalize_targets:
#         target_scaler = MinMaxScaler(feature_range=(-1, 1))
#         target_scaler.fit(train_dataset.force_targets.numpy())

#         train_dataset.force_targets = torch.from_numpy(
#             target_scaler.transform(train_dataset.force_targets.numpy())).float()
#         test_dataset.force_targets = torch.from_numpy(
#             target_scaler.transform(test_dataset.force_targets.numpy())).float()

#         # Save scaler to file to load it during eval
#         joblib.dump(target_scaler, constants.TARGET_SCALER_FN)






# def create_weights_path(model: str, num_epochs: int, base_dir: str = "weights") -> str:
#     """
#     Creates a directory path for saving weights with a unique run count and specified parameters.

#     """
#     assert isinstance(model, str)
#     assert isinstance(num_epochs, int)
#     if os.path.isdir(model):
#         dir_name = model.split("/")[2]
#         dir_name_split = dir_name.split("_")
#         cnn_name = dir_name_split[2] if len(
#             dir_name_split) > 1 else dir_name_split[0]
#         model_name = f"pretrained_{cnn_name}"
#     else:
#         model_name = model

#     base_path = Path(base_dir)

#     run_name = f"{model_name}_epochs_{num_epochs}"
#     new_dir_path = base_path / run_name
#     return str(new_dir_path)




# def plot_forces(forces: np.ndarray, run_nr: int, policy: str, pdf: bool):
#     assert forces.shape[1] == 3
#     os.makedirs('plots', exist_ok=True)
#     time_axis = np.arange(forces.shape[0])

#     plt.figure()
#     plt.plot(time_axis, forces[:, 0],
#              label='x-axis', linestyle='-', marker='')
#     plt.plot(time_axis, forces[:, 1],
#              label='y-axis', linestyle='-', marker='')
#     plt.plot(time_axis, forces[:, 2],
#              label='z-axis', linestyle='-', marker='')
#     policy_name = "Force Policy" if policy == "force_policy" else "No Force Policy"
#     plt.title(f"Example of Force Data for a single Policy Rollout")
#     plt.xlabel('Time')
#     plt.ylabel('Force [N]')
#     plt.legend()
#     save_path = f"plots/rollout_{policy}_{run_nr}.{'pdf' if pdf else 'png'}"
#     plt.savefig(save_path)
#     plt.close()


# def get_log_dir(args: argparse.Namespace) -> str:
#     if os.path.isdir(args.model):
#         cnn_name = args.model.split("/")[2]
#         model_name = f"finetuned_{cnn_name}"
#     else:
#         model_name = args.model
#     num_ep = args.num_epochs
#     lr = args.lr
#     batch_size = args.batch_size
#     accel = int(args.use_acceleration)
#     normalized = int(args.normalize_targets)
#     pretrained = int(args.use_pretrained)
#     scheduled = "_scheduled" if args.lr_scheduler else ""
#     overfit = "overfit/" if args.overfit else ""
#     state = args.state
#     seq_length = args.seq_length
#     log_dir = f"runs/{model_name}/{overfit}{state}/force_est_{num_ep}ep_" + \
#         f"lr_{lr}{scheduled}_seq_length_{seq_length}_bs_{batch_size}_accel_{accel}_normalized_{normalized}"
#     return log_dir


# def get_run_numbers(args: argparse.Namespace) -> dict[str, List[List[int]]]:
#     train_runs = [args.force_runs, args.no_force_runs]
#     test_runs = train_runs if args.overfit else constants.DEFAULT_TEST_RUNS
#     return {"train": train_runs, "test": test_runs}


# def get_num_robot_features(args: argparse.Namespace) -> int:
#     if args.use_acceleration:
#         return constants.NUM_ROBOT_FEATURES_INCL_ACCEL
#     else:
#         return constants.NUM_ROBOT_FEATURES


# def get_image_transforms(args: argparse.Namespace) -> dict[str, transforms.Compose]:
#     train_transform = constants.RES_NET_TEST_TRANSFORM if args.overfit else constants.RES_NET_TRAIN_TRANSFORM
#     return {"train": train_transform,
#             "test": constants.RES_NET_TEST_TRANSFORM}



# def get_transformer_config(args: argparse.Namespace) -> TransformerConfig:
#     num_robot_features = get_num_robot_features(args)
#     dropout_rate = 0.0 if args.overfit else constants.DROPOUT_RATE
#     encoder_state = EncoderState.LINEAR if args.state == "linear" else EncoderState.CONV
#     return TransformerConfig(
#         num_robot_features=num_robot_features,
#         hidden_layers=constants.HIDDEN_LAYERS,
#         dropout_rate=dropout_rate,
#         use_batch_norm=True,
#         num_heads=constants.NUM_HEADS,
#         num_encoder_layers=constants.NUM_ENCODER_LAYERS,
#         num_decoder_layers=constants.NUM_DECODER_LAYERS,
#         dim_feedforward=constants.DIM_FEEDFORWARD,
#         encoder_state=encoder_state)


# if __name__ == "__main__":
#     all_X, all_y, all_img_left_paths, all_img_right_paths = load_dataset(
#         path="data", force_policy_runs=[3], no_force_policy_runs=[], create_plots=True, sequential=False)

#     assert all_X.shape == (2549, 44), f"{all_X.shape=}"
#     assert all_y.shape == (2549, 3)

#     assert len(all_img_left_paths) == 2549
#     assert len(all_img_right_paths) == 2549



# -----------------------------------------------------------------------------
#  utils_revised.py  ·  May 2025
#  Utilities referenced by the training script: checkpoint I/O, log‑dirs, and
#  *load_dataset()* that prepares numpy arrays + image‑path lists for both
#  state‑only and vision‑state experiments.
# -----------------------------------------------------------------------------
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
