import logging
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, precision_recall_curve, roc_auc_score
import wandb
import numpy as np

import src.config as config
from src.utils import SSIMLoss

# ロガーの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    """
    モデルの学習と検証を管理するクラス。
    Phase 1 (CNN) では、単純な損失関数 (CrossEntropyLoss) と通常の学習プロセスを適用する。
    Phase 2 (CAE) では、複合損失 (MSE + SSIM) と 混合精度学習 (Mixed Precision) を適用する。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module, # MSELoss (Main)
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        phase2: bool = False,
        threshold: float = 0.5
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion # Phase 1: CELoss, Phase 2: MSELoss
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.scheduler = scheduler
        self.phase2 = phase2
        self.threshold = threshold
        
        # モデルをデバイスへ転送
        self.model.to(self.device)
        
        # 損失関数の設定
        if self.phase2:
            self.ssim_criterion = SSIMLoss()
            self.alpha = 1.0 # MSE Weight
            self.beta = 0.0  # SSIM Weight (Disabled)
        
        # Mixed Precision Setup
        # MPSデバイスの場合は scaler は通常不要だが、float16の安定性のためには GradScaler が有効な場合がある。
        # PyTorchのバージョンによって挙動が異なるが、ここでは汎用的な torch.cuda.amp.GradScaler 
        # もしくは torch.amp.GradScaler (PyTorch 2.3+ for generic) を想定。
        # MPSでのGradScalerサポートはまだexperimentalな部分も多いため、
        # 今回はシンプルに autocast のみを使用し、もし勾配消失等があればScalerを検討する方針とする。
        # しかし、一般的なAMP実装としてScalerも準備しておく。
        self.use_amp = True
        self.scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else torch.amp.GradScaler('mps') if hasattr(torch.amp, 'GradScaler') and device == 'mps' else None
        
        # 最良スコア
        self.best_recall = 0.0
        self.best_f1 = 0.0
        self.best_auroc = 0.0 # Phase 2 (Max AUROC)
        self.best_loss = float('inf') 

    def train_one_epoch(self, epoch_index: int) -> float:
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_index+1} [Train]", leave=False)
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed Precision Context
            # MPSの場合は device_type='mps'
            device_type = 'cuda' if self.device == 'cuda' else 'mps' if self.device == 'mps' else 'cpu'
            dtype = torch.float16 if device_type != 'cpu' else torch.bfloat16
            
            try:
                # CPU以外はautocast有効
                with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=self.use_amp):
                    outputs = self.model(images)
                    
                    if self.phase2:
                        # Composite Loss: MSE + (1 - SSIM)
                        loss_mse = self.criterion(outputs, images)
                        loss_ssim = self.ssim_criterion(outputs, images)
                        loss = self.alpha * loss_mse + self.beta * loss_ssim
                    else:
                        loss = self.criterion(outputs, labels)
                
                # Backward & Step
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                    
            except Exception as e:
                # MPSなどでAutocastがコケた場合のフォールバック（デバッグ用）
                print(f"AMP Error: {e}, falling back to FP32")
                outputs = self.model(images)
                loss = self.criterion(outputs, images if self.phase2 else labels)
                loss.backward()
                self.optimizer.step()
            
            loss_val = loss.item()
            running_loss += loss_val
            
            pbar.set_postfix({'loss': f"{loss_val:.4f}"})

        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss

    def validate(self) -> Dict[str, any]:
        self.model.eval()
        running_loss = 0.0
        
        # Phase 2 Metrics
        running_loss_good = 0.0
        count_good = 0
        running_loss_bad = 0.0
        count_bad = 0
        
        all_preds = []
        all_labels = []
        all_scores = [] # For AUROC
        
        # 可視化要画像リスト
        viz_good_samples = []
        viz_bad_samples = []
        viz_limit = 4
        
        device_type = 'cuda' if self.device == 'cuda' else 'mps' if self.device == 'mps' else 'cpu'
        dtype = torch.float16 if device_type != 'cpu' else torch.bfloat16

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="[Validate]", leave=False)
            
            for i, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Validationは推論のみなのでAutocastは必須ではないが、速度向上のため有効化
                with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=self.use_amp):
                    outputs = self.model(images)
                    
                    if self.phase2:
                        # Loss Calculation (Composite)
                        loss_mse = self.mse_criterion(outputs, images)
                        loss_ssim = self.ssim_criterion(outputs, images)
                        loss = self.alpha * loss_mse + self.beta * loss_ssim
                        batch_loss = loss.item()
                        
                        # Anomaly Score Calculation (Pixel-wise MSE for now)
                        # NOTE: SSIM map could also be used for anomaly score, but using MSE is standard.
                        # Let's use MSE matching the config threshold logic.
                        loss_per_sample = nn.MSELoss(reduction='none')(outputs, images)
                        mse_per_image = loss_per_sample.view(images.size(0), -1).mean(dim=1)
                        
                        all_scores.extend(mse_per_image.float().cpu().numpy())
                        
                        # Good/Bad Split Logging
                        for idx, label in enumerate(labels):
                            mse_val = mse_per_image[idx].item()
                            if label.item() == 1: # Bad
                                running_loss_bad += mse_val
                                count_bad += 1
                                if len(viz_bad_samples) < viz_limit:
                                    viz_bad_samples.append((images[idx].float().cpu(), outputs[idx].float().cpu()))
                            else: # Good
                                running_loss_good += mse_val
                                count_good += 1
                                if len(viz_good_samples) < viz_limit:
                                    viz_good_samples.append((images[idx].float().cpu(), outputs[idx].float().cpu()))
                                    viz_good_samples.append((images[idx].float().cpu(), outputs[idx].float().cpu()))
                    else:
                        # Phase 1: Classification
                        loss = self.criterion(outputs, labels)
                        batch_loss = loss.item()
                        _, preds = torch.max(outputs, 1) # Logits -> Preds
                        all_preds.extend(preds.cpu().numpy())

                running_loss += batch_loss * images.size(0)
                all_labels.extend(labels.cpu().numpy())
        
        num_samples = len(self.val_loader.dataset)
        avg_loss = running_loss / num_samples
        
        metrics = { "val_loss": avg_loss }
        
        if self.phase2:
            metrics["val_loss_good"] = running_loss_good / count_good if count_good > 0 else 0.0
            metrics["val_loss_bad"] = running_loss_bad / count_bad if count_bad > 0 else 0.0
            
            y_true = np.array(all_labels)
            y_scores = np.array(all_scores)
            
            # --- AUROC ---
            # Good=0, Bad=1. Anomaly Score should be higher for Bad.
            try:
                auroc = roc_auc_score(y_true, y_scores)
            except ValueError:
                # If only one class is present in y_true
                auroc = 0.5
            metrics["val_auroc"] = auroc

            # --- Dynamic Thresholding (F1) ---
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            f1_scores = f1_scores[:-1] 
            
            if len(f1_scores) > 0:
                max_idx = np.argmax(f1_scores)
                max_f1 = f1_scores[max_idx]
                best_thresh = thresholds[max_idx]
            else:
                max_f1 = 0.0
                best_thresh = self.threshold
            
            metrics["val_f1_max"] = max_f1
            metrics["val_best_threshold"] = best_thresh
            
            # Visualization
            val_reconstruction_images = []
            for name, samples in [("Good", viz_good_samples), ("Bad", viz_bad_samples)]:
                for idx, (inp, out) in enumerate(samples):
                    combined = np.concatenate((inp.permute(1,2,0).numpy(), out.permute(1,2,0).numpy()), axis=1)
                    val_reconstruction_images.append(wandb.Image(combined, caption=f"{name}_{idx}"))
            metrics["val_reconstruction_images"] = val_reconstruction_images
            
            metrics["val_reconstruction_images"] = val_reconstruction_images
            
        else:
            # Phase 1 Metrics
            metrics["val_accuracy"] = (np.array(all_preds) == np.array(all_labels)).mean()
            metrics["val_recall"] = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
            metrics["val_f1"] = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
        
        return metrics

    def fit(self, num_epochs: int):
        logger.info(f"Start training for {num_epochs} epochs on {self.device} (Phase 2: {self.phase2})...")
        if self.scaler:
            logger.info(f"Mixed Precision Enabled (Scaler: {type(self.scaler).__name__})")
        else:
            logger.info(f"Mixed Precision Enabled (Autocast Only)")

        for epoch in range(num_epochs):
            start_time = time.time()
            
            train_loss = self.train_one_epoch(epoch)
            val_metrics = self.validate()
            
            if self.scheduler:
                self.scheduler.step()
            
            end_time = time.time()
            
            # Logging
            if self.phase2:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Val AUROC: {val_metrics['val_auroc']:.4f} "
                    f"| Val F1(Max): {val_metrics['val_f1_max']:.4f}"
                )
            else:
                logger.info(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val F1: {val_metrics['val_f1']:.4f}")
            
            # W&B
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                **{f"val/{k}": v for k, v in val_metrics.items() if not isinstance(v, list)}
            }
            if val_metrics.get("val_reconstruction_images"):
                log_dict["val/reconstruction"] = val_metrics["val_reconstruction_images"]
            
            wandb.log(log_dict)
            
            # Save Model Logic
            save_path = config.WEIGHTS_DIR / "best_model.pth"
            if self.phase2:
                # Max F1 (Dynamic Threshold)
                current_f1 = val_metrics.get("val_f1_max", 0.0)
                if current_f1 > self.best_f1:
                    self.best_f1 = current_f1
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"New Best F1 (Max): {self.best_f1:.4f}. Model saved.")
                
                # Also log AUROC but don't save based on it for now (or save as separate file if needed)
                current_auroc = val_metrics.get("val_auroc", 0.0)
                if current_auroc > self.best_auroc:
                    self.best_auroc = current_auroc
                    # Optional: save best_auroc model

            else:
                # Max F1
                current_f1 = val_metrics.get("val_f1", 0.0)
                if current_f1 > self.best_f1:
                    self.best_f1 = current_f1
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"New Best F1: {self.best_f1:.4f}. Model saved.")

        logger.info("Training completed.")

