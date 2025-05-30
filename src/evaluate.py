#!/usr/bin/env python3

from __future__ import annotations
import argparse, time, os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from sklearn.metrics import mean_squared_error
import wandb
from tqdm import tqdm
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd

from datasets.vision_robot_dataset import VisionRobotDataset, custom_collate_fn
from datasets import augmentations
from models.force_estimator import ForceEstimator
from utils import load_dataset

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained ForceEstimator")
    p.add_argument("--checkpoint", "-c", required=True,
                   help="Path to trained model checkpoint (*.pth)")
    p.add_argument("--mode", choices=["vision", "force", "vision+force"],
                   default="vision+force", help="Evaluation mode")
    p.add_argument("--runs", "-r", nargs="+", type=int, required=True,
                   help="Run IDs to evaluate")
    p.add_argument("--architecture", required=True, choices=["cnn", "vit"],
                   help="Model architecture type")
    p.add_argument("--seq-length", type=int, default=1,
                   help="Sequence length for temporal modeling")
    p.add_argument("--state-size", type=int, default=58,
                   help="Robot state vector size")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for evaluation")
    p.add_argument("--data-root", default="data",
                   help="Path to data directory")
    p.add_argument("--output-dir", default="evaluation_results",
                   help="Directory to save evaluation results")
    p.add_argument("--wandb-project", default="force_estimation_eval",
                   help="W&B project for logging results")
    p.add_argument("--feature-analysis", action="store_true",
                   help="Perform detailed feature analysis")
    p.add_argument("--save-predictions", action="store_true",
                   help="Save raw force predictions to CSV")
    p.add_argument("--save-smooth-predictions", action="store_true",
                   help="Save smoothed force predictions to CSV")
    p.add_argument("--save-ground-truth", action="store_true",
                   help="Save raw ground truth forces to CSV")
    p.add_argument("--save-smooth-ground-truth", action="store_true",
                   help="Save smoothed ground truth forces to CSV")
    p.add_argument("--save-run-info", action="store_true",
                   help="Save run information to CSV")
    return p.parse_args()

def get_transforms():
    """Get evaluation transforms."""
    transforms = augmentations.Compose([
        augmentations.CentreCrop(),
        augmentations.SquareResize(),
        augmentations.ArrayToTensor(),
        augmentations.Normalize([0.45]*3, [0.225]*3)
    ])
    print("Using evaluation transforms:", transforms)
    return transforms

def plot_force_comparison(gt: np.ndarray, pred: np.ndarray,
                         save_path: Path, run_id: int) -> List[wandb.Image]:
    """Plot ground truth vs predicted forces with enhanced visualization."""
    images = []
    axes = "xyz"
    
    # Create directory if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Plot individual components
    for i in range(3):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(gt[:, i], label="Ground Truth", color='blue', alpha=0.7)
        ax.plot(pred[:, i], label="Prediction", color='red', linestyle='--', alpha=0.7)
        
        # Add error region
        error = np.abs(gt[:, i] - pred[:, i])
        ax.fill_between(range(len(gt)), gt[:, i] - error, gt[:, i] + error,
                       color='gray', alpha=0.2, label='Error Region')
        
        ax.set_title(f"Force {axes[i].upper()} - Run {run_id}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Force (N)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add metrics to plot
        mse = mean_squared_error(gt[:, i], pred[:, i])
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(gt[:, i] - pred[:, i]))
        ax.text(0.02, 0.98, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save locally
        plt.savefig(save_path/f"force_{axes[i]}_run{run_id}.pdf", dpi=300, bbox_inches='tight')
        images.append(wandb.Image(fig))
        plt.close(fig)
    
    # Plot 3D trajectory
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], label='Ground Truth', color='blue')
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], label='Prediction', color='red', linestyle='--')
    ax.set_xlabel('Force X')
    ax.set_ylabel('Force Y')
    ax.set_zlabel('Force Z')
    ax.legend()
    plt.savefig(save_path/f"force_3d_run{run_id}.pdf", dpi=300, bbox_inches='tight')
    images.append(wandb.Image(fig))
    plt.close(fig)
    
    return images

def analyze_features(model: nn.Module, batch: Dict[str, torch.Tensor],
                    device: torch.device, mode: str) -> Dict[str, wandb.Image]:
    """Analyze and visualize model features."""
    model.eval()
    with torch.no_grad():
        # Get image features if applicable
        if mode in ["vision", "vision+force"]:
            img_features = model.get_image_latent(batch["img_right"].to(device))
            img_features = img_features.cpu().numpy()
            
            # PCA visualization
            pca = PCA(n_components=2)
            img_features_2d = pca.fit_transform(img_features)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            scatter = ax.scatter(img_features_2d[:, 0], img_features_2d[:, 1],
                               c=np.arange(len(img_features_2d)), cmap='viridis')
            ax.set_title("PCA of Image Features")
            plt.colorbar(scatter, label="Sample Index")
            
            # Feature correlation heatmap
            fig_corr, ax_corr = plt.subplots(figsize=(10, 10))
            sns.heatmap(np.corrcoef(img_features.T), ax=ax_corr,
                       cmap='coolwarm', center=0)
            ax_corr.set_title("Feature Correlation Matrix")
            
            return {
                "pca_viz": wandb.Image(fig),
                "feature_corr": wandb.Image(fig_corr)
            }
    return {}

def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics."""
    metrics = {}
    
    # Global metrics
    mse = mean_squared_error(gt, pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(gt - pred))
    mape = np.mean(np.abs((gt - pred) / (gt + 1e-8))) * 100
    
    metrics.update({
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape
    })
    
    # Per-axis metrics
    for i, axis in enumerate("xyz"):
        axis_gt = gt[:, i]
        axis_pred = pred[:, i]
        
        axis_mse = mean_squared_error(axis_gt, axis_pred)
        axis_rmse = np.sqrt(axis_mse)
        axis_mae = np.mean(np.abs(axis_gt - axis_pred))
        axis_mape = np.mean(np.abs((axis_gt - axis_pred) / (axis_gt + 1e-8))) * 100
        
        # Correlation coefficient
        axis_corr = np.corrcoef(axis_gt, axis_pred)[0, 1]
        
        metrics.update({
            f"mse_f{axis}": axis_mse,
            f"rmse_f{axis}": axis_rmse,
            f"mae_f{axis}": axis_mae,
            f"mape_f{axis}": axis_mape,
            f"corr_f{axis}": axis_corr
        })
    
    return metrics

def smooth_forces(forces: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply moving average smoothing to force data."""
    return pd.DataFrame(forces).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values

def save_forces_to_csv(predictions: np.ndarray, 
                      ground_truth: np.ndarray,
                      run_id: int, 
                      save_path: Path,
                      smooth_pred: Optional[np.ndarray] = None,
                      smooth_gt: Optional[np.ndarray] = None):
    """Save all force data to CSV with one row per run_id/force_type/axis combination."""
    # Create directory if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize list to store rows
    rows = []
    
    # Helper function to add a row of data
    def add_row(data: np.ndarray, force_type: str):
        for i, axis in enumerate(['x', 'y', 'z']):
            # Convert the force values to a string, with values separated by semicolons
            values_str = ';'.join(map(str, data[:, i]))
            rows.append({
                'run_id': run_id,
                'force_type': force_type,
                'axis': axis,
                'values': values_str
            })
    
    # Add all available force data
    add_row(predictions, 'predicted')
    add_row(ground_truth, 'ground_truth')
    if smooth_pred is not None:
        add_row(smooth_pred, 'smooth_predicted')
    if smooth_gt is not None:
        add_row(smooth_gt, 'smooth_ground_truth')
    
    # Create and save DataFrame
    df = pd.DataFrame(rows)
    csv_path = save_path/f"forces_run{run_id}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved all force data to {csv_path}")

def evaluate_run(model: nn.Module,
                dataloader: torch.utils.data.DataLoader,
                device: torch.device,
                mode: str,
                run_id: int,
                output_dir: Path,
                feature_analysis: bool = False,
                save_predictions: bool = False,
                save_smooth_predictions: bool = False,
                save_ground_truth: bool = False,
                save_smooth_ground_truth: bool = False,
                save_run_info: bool = False) -> Tuple[Dict[str, float], List[wandb.Image]]:
    """Evaluate model on a single run with enhanced monitoring."""
    model.eval()
    all_preds = []
    all_targets = []
    feature_viz = {}
    
    print(f"\nEvaluating run {run_id}...")
    print(f"Mode: {mode}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Run {run_id}")):
            # Validate batch contents
            if mode in ["vision", "vision+force"]:
                assert "img_right" in batch, f"Missing images in {mode} mode"
                assert batch["img_right"].dim() in [4, 5], \
                    f"Wrong image dimensions: {batch['img_right'].shape}"
            if mode in ["force", "vision+force"]:
                assert "robot_state" in batch, f"Missing robot state in {mode} mode"
            
            # Log shapes periodically
            if batch_idx == 0:
                print("\nBatch shapes:")
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        print(f"- {k}: {tuple(v.shape)}")
            
            # Prepare inputs based on mode
            inputs = {}
            if mode in ["vision", "vision+force"]:
                inputs["img"] = batch["img_right"].to(device)
            if mode in ["force", "vision+force"]:
                inputs["state"] = batch["robot_state"].to(device).float()
            
            # Forward pass
            pred = model(**inputs)
            
            # Store predictions
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch["forces"].numpy())
            
            # Feature analysis on first batch
            if feature_analysis and batch_idx == 0:
                feature_viz = analyze_features(model, batch, device, mode)
    
    # Concatenate predictions
    predictions = np.concatenate(all_preds)
    ground_truth = np.concatenate(all_targets)
    
    # Compute smoothed data if requested
    smooth_pred = smooth_forces(predictions) if save_smooth_predictions else None
    smooth_gt = smooth_forces(ground_truth) if save_smooth_ground_truth else None
    
    # Save all force data if any saving is requested
    if any([save_predictions, save_ground_truth, save_smooth_predictions, save_smooth_ground_truth]):
        save_forces_to_csv(
            predictions=predictions,
            ground_truth=ground_truth,
            run_id=run_id,
            save_path=output_dir/f"run_{run_id}",
            smooth_pred=smooth_pred,
            smooth_gt=smooth_gt
        )
    
    # Save run information if requested
    if save_run_info:
        run_info = {
            'run_id': run_id,
            'mode': mode,
            'sequence_length': dataloader.dataset.seq_length,
            'total_samples': len(predictions),
            'sampling_rate': 30  # Hz
        }
        pd.DataFrame([run_info]).to_csv(output_dir/f"run_{run_id}/run_info.csv", index=False)
    
    print(f"\nPrediction statistics:")
    print(f"- Shape: {predictions.shape}")
    print(f"- Range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"- Mean: {predictions.mean():.3f}")
    print(f"- Std: {predictions.std():.3f}")
    
    # Compute metrics
    metrics = compute_metrics(ground_truth, predictions)
    
    # Generate plots
    force_plots = plot_force_comparison(
        ground_truth, predictions,
        output_dir/f"run_{run_id}",
        run_id
    )
    
    # Add feature visualization to plots
    if feature_viz:
        force_plots.extend(list(feature_viz.values()))
    
    return metrics, force_plots

def main():
    args = parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B
    run = wandb.init(
        project=args.wandb_project,
        name=f"eval_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}",
        config=vars(args)
    )
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Validate checkpoint mode matches evaluation mode
    ckpt_mode = checkpoint.get("mode", "unknown")
    if ckpt_mode != "unknown" and ckpt_mode != args.mode:
        print(f"Warning: Checkpoint mode ({ckpt_mode}) differs from evaluation mode ({args.mode})")
    
    model = ForceEstimator(
        architecture=args.architecture,
        recurrency=True,
        seq_length=args.seq_length,
        state_size=args.state_size
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best RMSE from training: {checkpoint.get('best_rmse', 'unknown')}")
    
    # Prepare transforms
    transforms = get_transforms()
    
    # Evaluate each run
    all_metrics = []
    for run_id in args.runs:
        # Load data
        print(f"\nLoading data for run {run_id}...")
        feats, forces, imgL, imgR = load_dataset(
            path=args.data_root,
            force_policy_runs=[run_id],
            no_force_policy_runs=[],
            sequential=False,
            crop_runs=False
        )
        
        print(f"Data loaded successfully:")
        print(f"- Features shape: {feats.shape}")
        print(f"- Forces shape: {forces.shape}")
        print(f"- Number of images: {len(imgL)}")
        
        # Create dataset
        dataset = VisionRobotDataset(
            robot_features=feats,
            force_targets=forces,
            img_left_paths=imgL,
            img_right_paths=imgR,
            path=args.data_root,
            img_transforms=transforms,
            seq_length=args.seq_length
        )
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )
        
        # Evaluate
        run_metrics, force_plots = evaluate_run(
            model, dataloader, device,
            args.mode, run_id, output_dir,
            feature_analysis=args.feature_analysis,
            save_predictions=args.save_predictions,
            save_smooth_predictions=args.save_smooth_predictions,
            save_ground_truth=args.save_ground_truth,
            save_smooth_ground_truth=args.save_smooth_ground_truth,
            save_run_info=args.save_run_info
        )
        
        # Log to W&B
        metrics = {f"run_{run_id}/{k}": v for k, v in run_metrics.items()}
        wandb.log(metrics)
        
        # Log force plots
        for i, (axis, plot) in enumerate(zip("xyz", force_plots[:3])):  # First 3 are force plots
            wandb.log({f"run_{run_id}/force_{axis}": plot})
        
        if len(force_plots) > 3:  # Feature visualization plots
            wandb.log({
                f"run_{run_id}/force_3d": force_plots[3],
                f"run_{run_id}/features": force_plots[4:]
            })
        
        all_metrics.append(run_metrics)
        
        # Print detailed results
        print(f"\nRun {run_id} Results:")
        print("=" * 40)
        for k, v in run_metrics.items():
            print(f"{k:>15}: {v:.4f}")
    
    # Compute and log aggregate metrics
    print("\nAggregate Results:")
    print("=" * 40)
    
    # Create results table for W&B
    table = wandb.Table(columns=["Metric", "Mean", "Std"])
    
    for metric in sorted(all_metrics[0].keys()):
        values = [m[metric] for m in all_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric:>15}: {mean_val:.4f} ± {std_val:.4f}")
        
        wandb.log({
            f"aggregate/{metric}_mean": mean_val,
            f"aggregate/{metric}_std": std_val
        })
        
        table.add_data(metric, f"{mean_val:.4f}", f"{std_val:.4f}")
    
    wandb.log({"metrics_summary": table})
    
    # Save detailed results
    results_file = output_dir/"evaluation_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Evaluation Results\n")
        f.write(f"=================\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Architecture: {args.architecture}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Sequence length: {args.seq_length}\n")
        f.write(f"Runs evaluated: {args.runs}\n\n")
        
        for run_id, metrics in zip(args.runs, all_metrics):
            f.write(f"Run {run_id}:\n")
            f.write("-" * 40 + "\n")
            for k, v in sorted(metrics.items()):
                f.write(f"{k:>15}: {v:.4f}\n")
            f.write("\n")
        
        f.write("\nAggregate Results:\n")
        f.write("-" * 40 + "\n")
        for metric in sorted(all_metrics[0].keys()):
            values = [m[metric] for m in all_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)
            f.write(f"{metric:>15}: {mean_val:.4f} ± {std_val:.4f}\n")
    
    print(f"\nResults saved to {results_file}")
    run.finish()

if __name__ == "__main__":
    main()
