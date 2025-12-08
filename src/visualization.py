import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
from typing import List, Tuple,Optional

def plot_f1_curve(
    thresholds: np.ndarray, 
    f1_scores: np.ndarray, 
    precisions: np.ndarray, 
    recalls: np.ndarray,
    best_threshold: float,
    save_path: Path
):
    """
    F1スコア、Precision、Recallの曲線を描画し保存する。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label="F1 Score", linewidth=2)
    plt.plot(thresholds, precisions[:-1], label="Precision", linestyle="--", alpha=0.5)
    plt.plot(thresholds, recalls[:-1], label="Recall", linestyle="--", alpha=0.5)
    
    plt.axvline(x=best_threshold, color="r", linestyle=":", label=f"Best Thresh ({best_threshold:.4f})")
    
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Tuning (F1 Score Maximization)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path)
    plt.close()

def plot_anomaly_map(
    original: np.ndarray, 
    reconstruction: np.ndarray, 
    anomaly_map: np.ndarray, 
    title: str,
    save_path: Path
):
    """
    入力画像、再構成画像、異常マップ（ヒートマップ）を並べて保存する。
    Args:
        original: (H, W) or (H, W, C), range [0, 1]
        reconstruction: (H, W) or (H, W, C), range [0, 1]
        anomaly_map: (H, W), raw score values
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    if original.ndim == 2:
        axes[0].imshow(original, cmap='gray')
    else:
        axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Reconstruction
    if reconstruction.ndim == 2:
        axes[1].imshow(reconstruction, cmap='gray')
    else:
        axes[1].imshow(reconstruction)
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')
    
    # Anomaly Map
    im = axes[2].imshow(anomaly_map, cmap='jet')
    axes[2].set_title(title)
    axes[2].axis('off')
    
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
