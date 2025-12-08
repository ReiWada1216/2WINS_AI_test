import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, precision_recall_curve, roc_auc_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import math

import src.config as config
from src.visualization import plot_f1_curve, plot_anomaly_map
from src.dataset import MVTecImageDataset

def gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """
    2D Gaussian Kernel creation (for blurring anomaly maps).
    """
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma**2.
    
    gaussian_kernel = (1./(2.*math.pi*variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) / \
                          (2*variance)
                      )
    
    return gaussian_kernel / torch.sum(gaussian_kernel)

def apply_gaussian_blur(img: torch.Tensor, kernel_size: int = 15, sigma: float = 4.0) -> torch.Tensor:
    """
    Apply Gaussian Blur to anomaly map tensor (1, 1, H, W).
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.to(img.device)
    
    padding = kernel_size // 2
    img_blurred = F.conv2d(img, kernel, padding=padding)
    return img_blurred

def stitch_patches(
    patches: torch.Tensor, 
    coords: List[Tuple[int, int]], 
    img_h: int, 
    img_w: int, 
    patch_size: int = config.PATCH_SIZE
) -> torch.Tensor:
    """
    Stitch patches back to full image with averaging overlap.
    """
    device = patches.device
    reconstructed = torch.zeros((1, img_h, img_w), device=device)
    count_map = torch.zeros((1, img_h, img_w), device=device)
    
    for patch, (x, y) in zip(patches, coords):
        reconstructed[:, y:y+patch_size, x:x+patch_size] += patch.squeeze(0)
        count_map[:, y:y+patch_size, x:x+patch_size] += 1.0
        
    # Avoid division by zero
    count_map[count_map == 0] = 1.0
    
    return reconstructed / count_map

def predict_single_image(
    model: nn.Module, 
    img_tensor: torch.Tensor, # (1, H, W)
    device: str,
    patch_size: int = config.PATCH_SIZE, 
    stride: int = config.STRIDE
) -> torch.Tensor:
    """
    1枚の画像に対してパッチ分割推論を行い、再構成画像を生成する。
    """
    model.eval()
    c, h, w = img_tensor.shape
    
    # Extract patches
    patches = []
    coords = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img_tensor[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((x, y))
            
    if not patches:
        return torch.zeros_like(img_tensor)

    # Batch Inference
    batch_tensors = torch.stack(patches).to(device) # (N, 1, 256, 256)
    
    reconstructed_patches = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(batch_tensors), batch_size):
            batch = batch_tensors[i:i+batch_size]
            # Mixed Precision inference
            device_type = 'cuda' if device == 'cuda' else 'mps' if device == 'mps' else 'cpu'
            dtype = torch.float16 if device_type != 'cpu' else torch.bfloat16
            
            with torch.amp.autocast(device_type=device_type, dtype=dtype):
                output = model(batch)
            
            reconstructed_patches.append(output.float()) # Back to float32 for stitching
            
    reconstructed_patches = torch.cat(reconstructed_patches, dim=0)
    
    # Stitching
    reconstructed_img = stitch_patches(reconstructed_patches, coords, h, w, patch_size)
    
    return reconstructed_img

def find_optimal_threshold(labels: np.ndarray, scores: np.ndarray, default_thresh: float = 0.5) -> Tuple[float, float, float, float]:
    """
    F1最大化による閾値決定。
    Returns: best_thresh, max_f1, max_prec, max_rec
    """
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    f1_scores = f1_scores[:-1] # Drop last
    
    if len(f1_scores) == 0:
        return default_thresh, 0.0, 0.0, 0.0
        
    max_idx = np.argmax(f1_scores)
    max_f1 = f1_scores[max_idx]
    best_thresh = thresholds[max_idx]
    best_prec = precisions[max_idx]
    best_rec = recalls[max_idx]
    
    # Plot
    save_dir = config.RESULTS_DIR / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_f1_curve(thresholds, f1_scores, precisions, recalls, best_thresh, save_dir / "threshold_tuning.png")
    
    return best_thresh, max_f1, best_prec, best_rec

def evaluate_test_set_full(
    model: nn.Module, 
    test_loader_full: DataLoader, # MVTecImageDatasetを用いたLoader
    device: str,
    threshold: float = 0.05
) -> Dict[str, float]:
    """
    Phase 2 Final Evaluation Pipeline (Full Image Stitching).
    """
    model.eval()
    print("Starting Full Image Evaluation (Stitching & Anomaly Map)...")
    
    true_labels = []
    anomaly_scores = []
    
    save_dir = config.RESULTS_DIR / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for i, (img, label, path) in enumerate(tqdm(test_loader_full, desc="Inference")):
            img = img.squeeze(0) # (1, H, W)
            label = label.item()
            true_labels.append(label)
            
            # 1. Stitching Inference
            recon = predict_single_image(model, img, device) # (1, H, W)
            
            # 2. Anomaly Map (Diff)
            diff = (img.to(device) - recon) ** 2
            
            # 3. Gaussian Blur
            anomaly_map = apply_gaussian_blur(diff.unsqueeze(0), sigma=4.0).squeeze(0).squeeze(0) # (H, W)
            
            # 4. Score (Max Value)
            score = anomaly_map.max().item()
            anomaly_scores.append(score)
            
            # Visualization (Sample)
            if i < 5 or (label == 1 and i < 15): # Save a few examples
                plot_anomaly_map(
                    original=img.squeeze(0).cpu().numpy(),
                    reconstruction=recon.squeeze(0).cpu().numpy(),
                    anomaly_map=anomaly_map.cpu().numpy(),
                    title=f"Label:{label} Score:{score:.4f}",
                    save_path=save_dir / f"eval_{i}_label{label}.png"
                )
                
    # Threshold Tuning
    y_true = np.array(true_labels)
    y_scores = np.array(anomaly_scores)
    
    # AUROC
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except:
        auroc = 0.5
        
    best_thresh, max_f1, best_prec, best_rec = find_optimal_threshold(y_true, y_scores, default_thresh=threshold)
    
    print("\n[Final Evaluation Results]")
    print(f"AUROC:     {auroc:.4f}")
    print(f"Max F1:    {max_f1:.4f}")
    print(f"Precision: {best_prec:.4f}")
    print(f"Recall:    {best_rec:.4f}")
    print(f"Threshold: {best_thresh:.4f}")
    
    return {
        "test_auroc": auroc,
        "test_f1_max": max_f1,
        "test_precision": best_prec,
        "test_recall": best_rec,
        "test_threshold": best_thresh
    }

if __name__ == "__main__":
    pass

def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    threshold: float = 0.5,
    phase2: bool = True
) -> Dict[str, float]:
    """
    Wrapper for main.py compatibility.
    """
    if phase2:
        return evaluate_test_set_full(model, test_loader, device, threshold=threshold)
    else:
        # Phase 1: Classification Evaluation
        model.eval()
        print("Starting Phase 1 (Classification) Evaluation on Test Set...")
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Inference"):
                images = images.to(device)
                
                # Forward
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                
        # Metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        print("\n[Final Evaluation Results (Phase 1)]")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        return {
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        }
