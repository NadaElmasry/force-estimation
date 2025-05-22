#!/usr/bin/env python3
# ---------------------------------------------------------------------
#  evaluate_force_estimator.py  –  2025‑05‑21
# ---------------------------------------------------------------------
import argparse
import os
import time
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from datasets import augmentations
from datasets.dataset import SequentialDataset
from datasets.vision_state_dataset import VisionStateDataset
from logger import AverageMeter
from models.force_estimator import ForceEstimator
from utils import load_dataset, none_or_str

# -----------------------  CLI  -------------------------------------- #
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained ForceEstimator checkpoint")
    p.add_argument("--checkpoint", "-c", required=True,
                   help="*.pth file produced by training script")
    p.add_argument("--run", type=int, required=True,
                   help="policy‑run ID to evaluate on")
    #  model hyper‑params (MUST match the training run!)
    p.add_argument("--architecture", required=True, choices=["cnn", "vit", "fc"])
    p.add_argument("--type", required=True, choices=["v", "vs", "s"])
    p.add_argument("--recurrency", action="store_true")
    p.add_argument("--seq-length", type=int, default=1)
    p.add_argument("--state-size", type=int, default=58)
    p.add_argument("--att-type", default=None)
    #  misc
    p.add_argument("--pdf", action="store_true", help="save plots as PDF")
    p.add_argument("--data-root", default="data", help="root folder with roll_out/")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

# ------------------  eval helpers  ---------------------------------- #
def get_val_transforms():
    norm = augmentations.Normalize([0.45]*3, [0.225]*3)
    return augmentations.Compose([
        augmentations.CentreCrop(),
        augmentations.SquareResize(),
        augmentations.ArrayToTensor(),
        norm
    ])

def custom_collate_fn(batch):
    imgs = torch.stack([b["img_right"] for b in batch]) if "img_right" in batch[0] else None
    target_seq = torch.stack([b["target"] for b in batch])
    # if sequence, take last step
    forces = target_seq[:, -1] if target_seq.dim() == 3 else target_seq
    robot_state = torch.stack([b["features"] for b in batch])
    return {"img_right": imgs, "forces": forces, "robot_state": robot_state}

@torch.no_grad()
def infer(loader: DataLoader, model: nn.Module, args) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    crit = nn.MSELoss()
    rmse_meter = AverageMeter(i=1)
    all_pred, all_gt = [], []

    for batch in loader:
        tgt, pred = forward(args, batch, model)
        rmse_meter.update([torch.sqrt(crit(pred, tgt)).item()])
        all_pred.append(pred.cpu()); all_gt.append(tgt.cpu())

    print(f"[INFO] Eval RMSE: {rmse_meter.avg[0]:.4f}  ({len(loader.dataset)} samples)")
    return torch.cat(all_pred).numpy(), torch.cat(all_gt).numpy()

def forward(args, batch, model):
    if args.type == "s":
        st = batch["robot_state"].to(args.device).float()
        tgt = batch["forces"].to(args.device).float()
        pred = model(st)
    elif args.type == "v":
        img = batch["img_right"].to(args.device)
        tgt = batch["forces"].to(args.device).float()
        pred = model(img, None)
    else:                                           # vs
        img = batch["img_right"].to(args.device)
        st  = batch["robot_state"].to(args.device).float()
        tgt = batch["forces"].to(args.device).float()
        pred = model(img, st)
    return tgt, pred

# ------------------  plotting & utils  ------------------------------ #
def moving_average(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1: return x
    cumsum = np.cumsum(np.pad(x, ((k-1,0),(0,0))), axis=0)
    return (cumsum[k:] - cumsum[:-k]) / k

def save_txt(out_dir: Path, name: str, arr: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir/f"{name}.txt", arr, delimiter=",",
               header="Fx,Fy,Fz", comments='')

def plot(forces_pred, forces_smooth, forces_gt, rmse, run_id, out_dir, pdf=False):
    t = np.arange(len(forces_pred))
    axes = ["X", "Y", "Z"]; ext = "pdf" if pdf else "png"
    out_dir.mkdir(parents=True, exist_ok=True)

    for smooth, label in [(forces_pred, "pred"), (forces_smooth, "smooth")]:
        fig, axs = plt.subplots(3,1, figsize=(8,10), sharex=True)
        fig.suptitle(f"Run {run_id} – {label} vs GT  (RMSE={rmse:.3f})")
        for i, ax in enumerate(axs):
            ax.plot(t[:len(smooth)], smooth[:,i], label=label)
            ax.plot(t[:len(smooth)], forces_gt[:len(smooth),i], label="GT")
            ax.set_ylabel(f"F{axes[i]} [N]"); ax.legend()
        axs[-1].set_xlabel("Time (frames)")
        fig.tight_layout(rect=[0,0,1,0.96])
        fig.savefig(out_dir/f"{label}_run{run_id}.{ext}")
        plt.close(fig)

# ------------------  main eval routine  ----------------------------- #
def main():
    args = get_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    # ------- instantiate model -------
    model = ForceEstimator(
        architecture=args.architecture,
        recurrency=args.recurrency,
        pretrained=False,
        att_type=args.att_type,
        state_size=args.state_size,
        seq_length=args.seq_length,
    ).to(args.device)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    # ------- dataset -------
    if args.type == "s":
        feats, forces, _, _ = load_dataset(
            args.data_root, [args.run], [], sequential=True, crop_runs=False)
        ds = SequentialDataset(feats, forces,
                               seq_length=args.seq_length,
                               normalize_targets=False)
        collate = None
    else:
        vt = get_val_transforms()
        ds = VisionStateDataset(
            mode="custom-single",
            transform=vt,
            recurrency_size=args.seq_length,
            dataset=args.dataset,
            occlude_param=none_or_str("None"),
            force_policy_runs=[args.run]
        )
        collate = custom_collate_fn

    dl = DataLoader(ds, batch_size=64, shuffle=False,
                    collate_fn=collate, num_workers=4)

    # ------- inference -------
    pred, gt = infer(dl, model, args)
    rmse = np.sqrt(mean_squared_error(gt, pred))

    # ------- post‑process & plots -------
    smooth = moving_average(pred, k=5)
    ts = int(time.time())
    out_dir = Path("eval_outputs")/f"run{args.run}_{ts}"
    save_txt(out_dir, "pred", pred); save_txt(out_dir, "smooth", smooth); save_txt(out_dir, "gt", gt)
    plot(pred, smooth, gt, rmse, args.run, out_dir, pdf=args.pdf)
    print(f"[✓] Outputs saved under {out_dir}")

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    main()
