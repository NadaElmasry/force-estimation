#!/usr/bin/env python3
# ---------------------------------------------------------------------
#  right_camera_force_estimator.py ‑ 2025‑05 revision
#  Trains a force estimator from right‑camera frames (+/‑ state)
# ---------------------------------------------------------------------
import argparse
import csv
import gc
import os
import shutil
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Subset
import wandb

from datasets import augmentations
from logger import AverageMeter, TermLogger
from models.force_estimator import ForceEstimator
from utils import (
    create_saving_dir,
    load_dataset,
    none_or_str,
    save_checkpoint,
)

# ------------------------------------------------------------------ #
# Globals
# ------------------------------------------------------------------ #
device: torch.device                     # filled in main()
best_error = float("inf")
n_iter = 0                               # global step counter

# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
parser = argparse.ArgumentParser(
    description="Force‑estimation from right camera (+/‑ state)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Splits
parser.add_argument("--dataset", choices=["dafoes", "dvrk", "mixed"], default="mixed")
parser.add_argument("--train-force-runs", nargs="+", type=int, required=True)
parser.add_argument("--test-force-runs", nargs="+", type=int, required=True)
parser.add_argument("--kfold", type=int, default=1)

# Model
parser.add_argument("--architecture", choices=["cnn", "vit", "fc"], required=True)
parser.add_argument("--type", choices=["v", "vs", "s"], default="vs",
                    help="v=image, s=state, vs=vision+state")
parser.add_argument("--recurrency", action="store_true")
parser.add_argument("--seq-length", type=int, default=1)
parser.add_argument("--state-size", type=int, default=58)
parser.add_argument("--att-type", default=None)

# Optim
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch-size", "-b", type=int, default=128)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--beta", type=float, default=0.999)
parser.add_argument("--weight-decay", type=float, default=0)

# Loss weights
parser.add_argument("--rmse-loss-weight", "-r", type=float, default=5.0)
parser.add_argument("--l1-weight", type=float, default=1e-3)

# Misc / logging
parser.add_argument("--name", required=True)
parser.add_argument("--pretrained", type=str, default=None)
parser.add_argument("--print-freq", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log-output", action="store_true")
parser.add_argument("--occlude-param",
                    choices=["force_sensor", "robot_p", "robot_o", "robot_v",
                             "robot_w", "robot_q", "robot_vq", "robot_tq",
                             "robot_qd", "robot_tqd", "None"],
                    default="None")
parser.add_argument("--wandb-project", default="force_estimation_right")

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
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
    imgs = torch.stack([b["img_right"] for b in batch])

    forces_lst = [b["target"] for b in batch]          # (T,3) or (3,)
    if forces_lst[0].dim() == 2:                       # sequence → pick last
        forces = torch.stack([f[-1] for f in forces_lst])
    else:
        forces = torch.stack(forces_lst)

    robot_state = torch.stack([b["features"] for b in batch])
    return {"img_right": imgs, "forces": forces, "robot_state": robot_state}


@torch.no_grad()
def validate_epoch(args, loader, model, logger) -> float:
    """Returns validation RMSE."""
    rmse_meter, batch_t = AverageMeter(i=1, precision=4), AverageMeter()
    crit = nn.MSELoss()
    model.eval()

    end = time.time(); logger.valid_bar.update(0)

    for i, batch in enumerate(loader):
        targets, preds = forward_pass(args, batch, model)
        rmse = torch.sqrt(crit(preds, targets))
        rmse_meter.update([rmse.item()])

        batch_t.update(time.time() - end); end = time.time()
        logger.valid_bar.update(i + 1)
        if i % args.print_freq == 0:
            logger.valid_writer.write(f"Valid: Time {batch_t} RMSE {rmse_meter}")

    logger.valid_bar.update(len(loader))
    return rmse_meter.avg[0]


def forward_pass(args, batch, model) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatches to the correct model signature."""
    if args.type == "s":
        st = batch["robot_state"].to(device).float()
        targets = batch["forces"].to(device).float()
        preds = model(st)
    elif args.type == "v":
        img = batch["img_right"].to(device)
        targets = batch["forces"].to(device).float()
        preds = model(img, None)
    else:  # vs
        img = batch["img_right"].to(device)
        st = batch["robot_state"].to(device).float()
        targets = batch["forces"].to(device).float()
        preds = model(img, st)
    return targets, preds


def train_epoch(args, loader, model, optim, logger, tb_writer) -> float:
    """Runs one training epoch and logs every print‑freq minibatches."""
    global n_iter
    time_data, time_batch = AverageMeter(), AverageMeter()
    loss_meter = AverageMeter(i=2, precision=4)
    crit = nn.MSELoss()

    model.train(); end = time.time(); logger.train_bar.update(0)

    for i, batch in enumerate(loader):
        time_data.update(time.time() - end)
        targets, preds = forward_pass(args, batch, model)

        mse = crit(preds, targets)
        l1 = args.l1_weight * sum(p.abs().sum() for p in model.parameters())
        loss = args.rmse_loss_weight * mse + l1

        optim.zero_grad(); loss.backward(); optim.step()

        loss_meter.update([loss.item(), mse.item()], args.batch_size)
        time_batch.update(time.time() - end); end = time.time()

        if i % args.print_freq == 0:
            logger.train_writer.write(f"Train: Time {time_batch} "
                                      f"Data {time_data} Loss {loss_meter}")
            tb_writer.add_scalar("Loss/train", loss.item(), n_iter)
            tb_writer.add_scalar("MSE/train", mse.item(), n_iter)
        logger.train_bar.update(i + 1); n_iter += 1

    torch.cuda.empty_cache(); gc.collect()
    return loss_meter.avg[1]   # return mean MSE


def get_transforms():
    """Vision augmentations ⟶ (train_tf, val_tf)."""
    norm = augmentations.Normalize(mean=[0.45]*3, std=[0.225]*3)
    common = [augmentations.CentreCrop(), augmentations.SquareResize(),
              augmentations.ArrayToTensor(), norm]
    bright = augmentations.BrightnessContrast(contrast=2.0, brightness=12.0)

    train_tf = augmentations.Compose([
        augmentations.RandomHorizontalFlip(),
        augmentations.RandomVerticalFlip(),
        augmentations.RandomRotation(),
        *common, bright,
    ])
    val_tf = augmentations.Compose(common)
    return train_tf, val_tf

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    global device
    args = parser.parse_args()

    # --- determinism & device ---
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.deterministic = True; cudnn.benchmark = False

    # --- output folders ---
    root = Path(__file__).resolve().parent
    ckpt_root = root / "checkpoints"; ckpt_root.mkdir(exist_ok=True)
    occ = none_or_str(args.occlude_param)
    save_dir = create_saving_dir(
        ckpt_root, Path(args.name), args.architecture,
        args.dataset, args.recurrency, args.att_type, occ
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- logging ---
    tb_root = SummaryWriter(save_dir)

    # We create a **new** wandb run for each fold – no global run needed
    # (avoids nested runs clutter).

    train_tf, val_tf = get_transforms()

    # --- dataset loading ---
    run_map = {"train": (args.train_force_runs, []),
               "test":  (args.test_force_runs,  [])}

    if args.type == "s":                              # state only
        from datasets.dataset import SequentialDataset
        tr_data = load_dataset("data", *run_map["train"], sequential=True)
        te_data = load_dataset("data", *run_map["test"],  sequential=True)
        train_ds = SequentialDataset(*tr_data[:2], seq_length=args.seq_length,
                                     normalize_targets=False)
        val_ds   = SequentialDataset(*te_data[:2], seq_length=args.seq_length,
                                     normalize_targets=False)
    else:                                             # vision (+state)
        from datasets.vision_state_dataset import VisionStateDataset
        train_ds = VisionStateDataset(
            mode="train", transform=train_tf,
            recurrency_size=args.seq_length,
            dataset=args.dataset, occlude_param=occ
        )
        val_ds = VisionStateDataset(
            mode="val", transform=val_tf,
            recurrency_size=args.seq_length,
            dataset=args.dataset, occlude_param=occ
        )

    # --- K‑fold split indices ---
    splits = (KFold(args.kfold, shuffle=True, random_state=args.seed)
              .split(range(len(train_ds))) if args.kfold > 1
              else [(np.arange(len(train_ds)), None)])

    # ===========================  TRAIN / VAL LOOP  ===========================
    for fold, (tr_idx, _) in enumerate(splits, start=1):
        fold_name = f"{args.name}_fold{fold}" if args.kfold > 1 else args.name
        wandb_run = wandb.init(project=args.wandb_project,
                               name=fold_name, config=vars(args), reinit=True)

        tb_writer = SummaryWriter(save_dir / f"tb_fold{fold}")
        tr_subset = Subset(train_ds, tr_idx) if args.kfold > 1 else train_ds

        dl_train = DataLoader(tr_subset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True,
                              collate_fn=(custom_collate_fn
                                          if args.type != "s" else None))
        dl_val = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True,
                            collate_fn=(custom_collate_fn
                                        if args.type != "s" else None))

        # ---- model ----
        model = ForceEstimator(
            architecture=args.architecture,
            recurrency=args.recurrency,
            pretrained=False,
            att_type=args.att_type,
            state_size=args.state_size,
            seq_length=args.seq_length,
        ).to(device)

        if args.pretrained:
            ck = torch.load(args.pretrained, map_location=device)
            model.load_state_dict(ck["state_dict"], strict=False)

        optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

        # ---- progress logger ----
        logger = TermLogger(args.epochs, len(dl_train), len(dl_val))
        logger.epoch_bar.start(); best_rmse = float("inf")

        for epoch in range(args.epochs):
            logger.epoch_bar.update(epoch)
            logger.reset_train_bar()
            tr_mse = train_epoch(args, dl_train, model, optim, logger, tb_writer)

            logger.reset_valid_bar()
            val_rmse = validate_epoch(args, dl_val, model, logger)
            tb_writer.add_scalar("RMSE/val", val_rmse, epoch)
            wandb_run.log({"epoch": epoch, "train_mse": tr_mse,
                           "val_rmse": val_rmse})

            is_best = val_rmse < best_rmse
            best_rmse = min(best_rmse, val_rmse)
            save_checkpoint(save_dir,
                            {"epoch": epoch + 1,
                             "state_dict": model.state_dict()},
                            is_best,
                            filename=f"fold{fold}_epoch{epoch+1}.pth")

        logger.epoch_bar.finish()
        tb_writer.close(); wandb_run.finish()

    # --- promote best checkpoint ---
    best_ckpts = sorted(save_dir.glob("*best*.pth"), key=os.path.getmtime)
    if best_ckpts:
        shutil.copy(best_ckpts[-1], save_dir / "best_overall_model.pth")
        print(f"[✓] Best overall model saved to "
              f"{save_dir/'best_overall_model.pth'}")

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    main()
