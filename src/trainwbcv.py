#!/usr/bin/env python3

import argparse, itertools, time, shutil, os
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.backends.cudnn as cudnn
from sklearn.decomposition import PCA
from tensorboardX import SummaryWriter
import wandb
from typing import Dict, Optional, Tuple

from datasets.vision_robot_dataset import (
    VisionRobotDataset, custom_collate_fn, build_cv_loaders
)
# from datasets.cv_loader_builder import build_cv_loaders
from datasets import augmentations
from logger import AverageMeter, TermLogger
from models.force_estimator import ForceEstimator
from utils import load_dataset, create_saving_dir, save_checkpoint

# ───────────────────────────── CLI ────────────────────────────────
p = argparse.ArgumentParser(description='Cross-validation training for force estimation')
p.add_argument("--name", required=True, help="Experiment name")
p.add_argument("--mode", choices=["vision", "force", "vision+force"], default="vision+force",
               help="Training mode: vision-only, force-only, or combined")
p.add_argument("--dataset", default="mixed", help="Dataset type")
p.add_argument("--data-root", default=os.getenv("DAFOES_DATA", "data"),
               help="Path to data directory (can be set via DAFOES_DATA env var)")
p.add_argument("--train-force-runs", nargs="+", type=int, required=True,
               help="Run IDs for training")
p.add_argument("--test-force-runs",  nargs="+", type=int, required=True,
               help="Run IDs for testing")
p.add_argument("--kfold", type=int, default=5, help="Number of CV folds")
p.add_argument("--architecture", choices=["cnn", "vit"], default="vit",
               help="Vision backbone architecture")
p.add_argument("--seq-length", type=int, default=5, help="Sequence length for temporal modeling")
p.add_argument("--state-size", type=int, default=58, help="Robot state vector size")
p.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
p.add_argument("--batch-size", type=int, default=32, help="Training batch size")
p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
p.add_argument("--wandb-project", default="force_estimation_right",
               help="W&B project name")
args = p.parse_args()

def log_shapes(name: str, tensors: Dict[str, torch.Tensor]) -> None:
    """Log tensor shapes for debugging."""
    shapes = {k: tuple(v.shape) for k, v in tensors.items()}
    print(f"\n{name} shapes:", shapes)

# ───────────────────── reproducibility & device ───────────────────
print("\nSetting up training...")
torch.manual_seed(0); np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
cudnn.deterministic = True; cudnn.benchmark = False

# ────────────────────── folders & W&B run ─────────────────────────
root = Path(__file__).resolve().parent
ckpt_root = root/"checkpoints"; ckpt_root.mkdir(exist_ok=True)
save_dir = create_saving_dir(
    root=ckpt_root,
    experiment_name=Path(args.name),
    architecture=args.architecture,
    dataset=args.dataset,
    recurrency=True,
    att_type=None,
    occ_param=None
)
save_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving outputs to: {save_dir}")

# Initialize W&B
wandb.login()
print(f"Initialized W&B project: {args.wandb_project}")

# ─────────────────────── transforms --------------------------------
def get_tf(is_train=True):
    norm = augmentations.Normalize([0.45]*3, [0.225]*3)
    base = [augmentations.CentreCrop(), augmentations.SquareResize(),
            augmentations.ArrayToTensor(), norm]
    if not is_train:
        return augmentations.Compose(base)
    return augmentations.Compose([
        augmentations.RandomHorizontalFlip(),
        augmentations.RandomVerticalFlip(),
        augmentations.RandomRotation(),
        *base,
        augmentations.BrightnessContrast(2., 12.),
    ])

train_tf, val_tf = get_tf(True), get_tf(False)

# ─────────────────────── data loading -----------------------------
print("\nLoading datasets...")
def build_ds(force_runs, tf):
    """Build dataset with proper logging of data loading process."""
    print(f"\nLoading dataset for runs: {force_runs}")
    feats, forces, imgL, imgR = load_dataset(
        path=args.data_root,
        force_policy_runs=force_runs,
        no_force_policy_runs=[],
        sequential=False,
        crop_runs=False
    )
    print(f"Data loaded successfully:")
    print(f"- Features shape: {feats.shape}")
    print(f"- Forces shape: {forces.shape}")
    print(f"- Number of left images: {len(imgL)}")
    print(f"- Number of right images: {len(imgR)}")
    
    # Validate data shapes
    assert len(feats) == len(forces) == len(imgL) == len(imgR), \
        "Mismatched data lengths"
    assert forces.shape[1] == 3, \
        f"Forces should have 3 components (x,y,z), got {forces.shape[1]}"
    
    ds = VisionRobotDataset(
        robot_features=feats,
        force_targets=forces,
        img_left_paths=imgL,
        img_right_paths=imgR,
        path=args.data_root,
        img_transforms=tf,
        seq_length=args.seq_length,
    )
    print(f"Dataset created with {len(ds)} samples")
    return ds

train_ds = build_ds(args.train_force_runs, train_tf)
val_ds = build_ds(args.test_force_runs, val_tf)
print(f"Dataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}")

# build CV loaders
print("\nBuilding cross-validation folds...")
fold_loaders = build_cv_loaders(
    dataset=train_ds,
    k_folds=args.kfold,
    batch_size=args.batch_size,
    collate_fn=custom_collate_fn,
    pin_memory=True
)
print(f"Created {len(fold_loaders)} CV folds")

# helper visuals
def spec_and_pca(model: ForceEstimator, img: torch.Tensor) -> Tuple[wandb.Image, wandb.Image]:
    """Generate spectrogram and PCA visualizations of image features."""
    if img.dim() == 5: img = img.view(-1, *img.shape[2:])
    print(f"Extracting features from image shape: {img.shape}")
    feats = model.get_image_latent(img.to(device)).cpu().numpy()
    print(f"Extracted features shape: {feats.shape}")
    
    # Spectrogram
    fig1, ax = plt.subplots(figsize=(3,2))
    ax.imshow(feats.T, aspect="auto")
    ax.set_title("Feature Spectrogram")
    ax.axis("off")
    spec = wandb.Image(fig1)
    plt.close(fig1)
    
    # PCA
    XY = PCA(2).fit_transform(feats)
    fig2, ax = plt.subplots(figsize=(3,2))
    ax.scatter(XY[:,0], XY[:,1], s=8)
    ax.set_title("PCA of Features")
    ax.axis("off")
    pca = wandb.Image(fig2)
    plt.close(fig2)
    
    return spec, pca

def force_curves(gt: np.ndarray, pr: np.ndarray) -> list:
    """Generate force prediction vs ground truth plots."""
    outs = []
    axes = "xyz"
    for i in range(3):
        fig, ax = plt.subplots(figsize=(3,2))
        ax.plot(gt[:,i], label="Ground Truth", color='blue')
        ax.plot(pr[:,i], label="Prediction", color='red', linestyle='--')
        ax.set_title(f"Force {axes[i].upper()}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Force (N)")
        ax.legend()
        outs.append(wandb.Image(fig))
        plt.close(fig)
    return outs

def validate_batch(batch: Dict[str, torch.Tensor], mode: str) -> None:
    """Validate batch contents based on training mode."""
    expected_keys = {
        "vision": ["img_right"],
        "force": ["robot_state"],
        "vision+force": ["img_right", "robot_state"]
    }
    for k in expected_keys[mode]:
        assert k in batch, f"Missing {k} in batch for {mode} mode"

# Loss function
crit = torch.nn.MSELoss()

# ────────────────────── training function ─────────────────────────
def train_epoch(model, loader, optimizer, epoch, logger, mode="vision+force"):
    """Train for one epoch with comprehensive logging."""
    logger.epoch_bar.update(epoch)
    logger.reset_train_bar()
    model.train()
    mse_meter = AverageMeter(i=1)
    
    for batch_idx, batch in enumerate(loader):
        # Validate batch contents based on mode
        if mode in ["vision", "vision+force"]:
            assert "img_right" in batch, f"Missing images in {mode} mode"
            assert batch["img_right"].dim() in [4, 5], \
                f"Wrong image dimensions: {batch['img_right'].shape}"
        if mode in ["force", "vision+force"]:
            assert "robot_state" in batch, f"Missing robot state in {mode} mode"
        
        # Log shapes periodically
        if batch_idx % 100 == 0:
            shapes = {k: tuple(v.shape) for k, v in batch.items()}
            print(f"\nBatch {batch_idx} shapes:", shapes)
        
        # Prepare inputs based on mode
        inputs = {}
        if mode in ["vision", "vision+force"]:
            inputs["img"] = batch["img_right"].to(device)
        if mode in ["force", "vision+force"]:
            inputs["state"] = batch["robot_state"].to(device).float()
        
        tgt = batch["forces"].to(device).float()
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(**inputs)
        mse = crit(pred, tgt)
        
        # Backward pass
        mse.backward()
        optimizer.step()
        
        # Update metrics
        mse_meter.update([mse.item()], n=tgt.size(0))
        logger.train_bar.update()
        
        # Log to W&B
        if batch_idx % 10 == 0:
            wandb.log({
                "train/batch_mse": mse.item(),
                "train/batch_rmse": np.sqrt(mse.item()),
                "epoch": epoch,
                "batch": batch_idx
            })
    
    return mse_meter.avg[0]

def validate(model, loader, epoch, logger, mode="vision+force"):
    """Validate with comprehensive logging."""
    logger.reset_valid_bar()
    model.eval()
    val_meter = AverageMeter(i=1)
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Validate batch contents based on mode
            if mode in ["vision", "vision+force"]:
                assert "img_right" in batch, f"Missing images in {mode} mode"
            if mode in ["force", "vision+force"]:
                assert "robot_state" in batch, f"Missing robot state in {mode} mode"
            
            # Log shapes periodically
            if batch_idx % 50 == 0:
                shapes = {k: tuple(v.shape) for k, v in batch.items()}
                print(f"\nValidation batch {batch_idx} shapes:", shapes)
            
            # Prepare inputs based on mode
            inputs = {}
            if mode in ["vision", "vision+force"]:
                inputs["img"] = batch["img_right"].to(device)
            if mode in ["force", "vision+force"]:
                inputs["state"] = batch["robot_state"].to(device).float()
            
            tgt = batch["forces"].to(device).float()
            pred = model(**inputs)
            
            rmse = torch.sqrt(crit(pred, tgt))
            val_meter.update([rmse.item()], n=tgt.size(0))
            logger.valid_bar.update()
            
            # Store predictions for visualization
            all_preds.append(pred.cpu())
            all_targets.append(tgt.cpu())
    
    # Compute final metrics
    val_rmse = val_meter.avg[0]
    val_mse = val_rmse ** 2
    
    # Concatenate all predictions for visualization
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    return val_rmse, val_mse, all_preds, all_targets

# ───────────────────────── training over folds ────────────────────
print("\nStarting cross-validation training...")
print(f"Training mode: {args.mode}")
fold_rmses = []

for fold_id, (dl_train, dl_val) in enumerate(fold_loaders, start=1):
    print(f"\nTraining Fold {fold_id}/{len(fold_loaders)}")
    
    # Initialize W&B run for this fold
    run = wandb.init(
        project=args.wandb_project,
        name=f"{args.name}_fold{fold_id}",
        config={
            **vars(args),
            "fold": fold_id,
            "train_samples": len(dl_train.dataset),
            "val_samples": len(dl_val.dataset)
        },
        reinit=True
    )
    
    # Initialize model and training components
    tb = SummaryWriter(save_dir/f"tb_fold{fold_id}")
    model = ForceEstimator(
        architecture=args.architecture,
        recurrency=True,
        seq_length=args.seq_length,
        state_size=args.state_size
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger = TermLogger(args.epochs, len(dl_train), len(dl_val))
    best_rmse = float('inf')
    logger.epoch_bar.start()

    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_mse = train_epoch(model, dl_train, opt, epoch, logger, args.mode)
        train_rmse = np.sqrt(train_mse)
        
        # Validate
        val_rmse, val_mse, all_preds, all_targets = validate(
            model, dl_val, epoch, logger, args.mode
        )
        
        # Log metrics
        metrics = {
            "epoch": epoch,
            "fold": fold_id,
            "train/mse": train_mse,
            "train/rmse": train_rmse,
            "val/mse": val_mse,
            "val/rmse": val_rmse,
        }
        wandb.log(metrics)
        
        # Log to tensorboard
        tb.add_scalar("RMSE/train", train_rmse, epoch)
        tb.add_scalar("RMSE/val", val_rmse, epoch)
        
        print(f"Epoch {epoch+1} - Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")

        # Every 5 epochs: save checkpoint and generate visualizations
        if (epoch + 1) % 5 == 0:
            print(f"\nGenerating visualizations for epoch {epoch+1}")
            
            # Save checkpoint
            ck = save_dir/f"fold{fold_id}_ep{epoch+1}.pth"
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "mode": args.mode
            }, ck)
            
            # Log checkpoint to W&B
            art = wandb.Artifact(
                ck.name,
                type="checkpoint",
                description=f"Model checkpoint for fold {fold_id}, epoch {epoch+1}"
            )
            art.add_file(ck)
            wandb.log_artifact(art)
            
            # Generate visualizations
            batch0 = next(iter(dl_val))
            if args.mode in ["vision", "vision+force"]:
                spec, pca = spec_and_pca(model, batch0["img_right"])
            else:
                spec, pca = None, None
            
            # Generate predictions for visualization
            inputs = {}
            if args.mode in ["vision", "vision+force"]:
                inputs["img"] = batch0["img_right"].to(device)
            if args.mode in ["force", "vision+force"]:
                inputs["state"] = batch0["robot_state"].to(device).float()
            
            pred0 = model(**inputs).cpu().numpy()
            curves = force_curves(batch0["forces"].numpy(), pred0)
            
            # Create and log table
            tbl = wandb.Table(
                columns=["epoch", "fold", "train_loss", "val_loss",
                        "spectrogram", "pca", "Fx", "Fy", "Fz"]
            )
            tbl.add_data(
                epoch + 1, fold_id,
                float(train_mse), float(val_rmse),
                spec, pca, *curves
            )
            wandb.log({"epoch5_table": tbl})

        # Save best model
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            save_checkpoint(
                save_dir,
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_rmse": best_rmse,
                    "mode": args.mode
                },
                is_best=True,
                filename=f"best_fold{fold_id}.pth"
            )
            print(f"New best model saved with RMSE: {best_rmse:.4f}")

    fold_rmses.append(best_rmse)
    tb.close()
    run.finish()

# ───────────────────────── final summary ────────────────────────
print("\nTraining completed!")
print("Fold RMSEs:", fold_rmses)
print(f"Mean RMSE: {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}")

# Copy best model across folds
best = sorted(save_dir.glob("best_fold*.pth"), key=lambda p: p.stat().st_mtime)[-1]
shutil.copy(best, save_dir/"best_overall_model.pth")
print(f"\n[✓] Best model saved → {save_dir/'best_overall_model.pth'}")
