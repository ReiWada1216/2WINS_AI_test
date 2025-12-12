
import torch
from pathlib import Path
import time
import numpy as np
from torchvision.transforms.v2 import Resize, Compose
from torchvision.transforms.functional import InterpolationMode
from anomalib.data import Folder, PredictDataset
from anomalib.models import EfficientAd
from anomalib.engine import Engine
from anomalib.pre_processing import PreProcessor
from torchmetrics import AUROC, F1Score, ConfusionMatrix
from torch.utils.data import DataLoader

# Fix for PyTorch 2.6+ checkpoint loading
torch.serialization.add_safe_globals([PreProcessor, Compose, Resize, InterpolationMode])

def run_evaluation():
    print("=== Evaluation Started ===")
    
    # Config
    ckpt_path = Path("results/efficient_ad/efficient_ad_v1_best/weights/efficient_v1_rocauc_0.983.ckpt")
    image_size = (384, 384)
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Switch to CPU strictly for stability during full dataset evaluation
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 2. Model
    print(f"Loading model from {ckpt_path}...")
    transform = Compose([Resize(image_size, antialias=True)])
    pre_processor = PreProcessor(transform=transform)
    
    model = EfficientAd.load_from_checkpoint(
        str(ckpt_path),
        pre_processor=pre_processor,
        map_location=device,
        weights_only=False
    )
    model.eval()
    
    # Check device (Expecting MPS)
    print(f"Model device (first param): {next(model.parameters()).device}")

    
    # 3. Data (Use PredictDataset to match visualize_inference.py success)
    dataset_root = Path("./data/processed/test")
    dataset = PredictDataset(
        path=dataset_root,
        transform=transform,
        image_size=image_size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )

    # 4. Engine for Prediction
    engine = Engine(
        accelerator="cpu",
        devices=1,
        default_root_dir="results/evaluation"
    )
    
    print("Running prediction (CPU)...")
    # PASS ckpt_path=None explicitly
    predictions = engine.predict(model=model, dataloaders=dataloader)
    
    # 5. Collect Results
    all_pred_scores = []
    all_pred_labels = []
    all_gt_labels = []
    
    for pred in predictions:
        # pred is a dictionary for the batch (batch_size=1)
        # keys: image_path, pred_score, pred_label, anomaly_map ...
        
        # Derived GT Label from path
        img_path = str(pred["image_path"][0])
        gt_label = 1 if "bad" in img_path else 0
        
        all_pred_scores.append(pred["pred_score"])
        all_pred_labels.append(pred["pred_label"])
        all_gt_labels.append(torch.tensor([gt_label]))
        
    pred_scores = torch.cat(all_pred_scores).cpu()
    pred_labels = torch.cat(all_pred_labels).cpu()
    gt_labels = torch.cat(all_gt_labels).cpu()
    
    # 6. Calculate Metrics
    print("\n--- Metrics ---")
    
    # AUROC
    auroc = AUROC(task="binary")
    auroc_score = auroc(pred_scores, gt_labels)
    print(f"Image ROC AUC: {auroc_score:.4f}")
    
    # F1 Score
    f1 = F1Score(task="binary")
    f1_score = f1(pred_labels, gt_labels)
    print(f"Image F1 Score: {f1_score:.4f}")
    
    # Confusion Matrix
    confmat = ConfusionMatrix(task="binary", num_classes=2)
    cm = confmat(pred_labels, gt_labels)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"TN: {tn} | FP: {fp}")
    print(f"FN: {fn} | TP: {tp}")
    print(cm)
    
    # 7. Latency Measurement (Pure Inference)
    print("\n--- Latency Measurement ---")
    dummy_input = torch.randn(1, 3, *image_size).to(device)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(20):
            model(dummy_input)
            
    # Timing
    print("Measuring latency (100 iterations)...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(dummy_input)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = (total_time / 100) * 1000 # ms
    throughput = 1000 / avg_latency # fps
    
    print(f"Average Inference Latency: {avg_latency:.2f} ms")
    print(f"Throughput: {throughput:.2f} FPS")

if __name__ == "__main__":
    run_evaluation()
