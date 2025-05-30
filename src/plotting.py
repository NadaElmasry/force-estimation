from pathlib import Path
import numpy as np, matplotlib.pyplot as plt, wandb
from typing import Tuple
import matplotlib.pyplot as plt, wandb, numpy as np
from sklearn.decomposition import PCA


# ─────────────────────── plotting helpers  ───────────────────────────────────
def moving_avg(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1: return x
    pad = np.pad(x, ((k-1,0),(0,0)), mode="constant")
    csum = np.cumsum(pad, axis=0)
    return (csum[k:] - csum[:-k]) / k

def curve_images(gt: np.ndarray, pred: np.ndarray, run_id: int) -> List[wandb.Image]:
    axes="xyz"; imgs=[]
    for i in range(3):
        fig, ax = plt.subplots(figsize=(4,2))
        ax.plot(pred[:,i], label="pred"); ax.plot(gt[:,i], label="gt")
        ax.set_title(f"Run {run_id} • F{axes[i]}"); ax.axis("off"); ax.legend()
        imgs.append(wandb.Image(fig))
        plt.close(fig)
    return imgs

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


def spectrogram(feats: np.ndarray) -> wandb.Image:
    fig, ax = plt.subplots(figsize=(3,2))
    ax.imshow(feats.T, aspect="auto", origin="lower"); ax.axis("off")
    out = wandb.Image(fig); plt.close(fig); return out

def pca_scatter(feats: np.ndarray) -> wandb.Image:
    xy = PCA(2).fit_transform(feats)
    fig, ax = plt.subplots(figsize=(3,2))
    ax.scatter(xy[:,0], xy[:,1], s=8); ax.axis("off")
    out = wandb.Image(fig); plt.close(fig); return out