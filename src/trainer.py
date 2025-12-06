import logging
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score
import wandb

import src.config as config

# ロガーの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    """
    モデルの学習と検証を管理するクラス。
    W&Bへのログ出力機能、KPI計算、およびベストモデル保存機能を持つ。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Trainerの初期化。

        Args:
            model (nn.Module): 学習対象のモデル
            optimizer (optim.Optimizer): オプティマイザ
            criterion (nn.Module): 損失関数
            train_loader (DataLoader): 学習データのローダー
            val_loader (DataLoader): 検証データのローダー
            device (str): 計算デバイス ('cpu', 'cuda', 'mps' など)
            scheduler (Optional[optim.lr_scheduler._LRScheduler]): 学習率スケジューラ (オプション)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.scheduler = scheduler
        
        # モデルをデバイスへ転送
        self.model.to(self.device)
        
        # 最良スコアの記録用 (Recallを重視)
        self.best_recall = 0.0

    def train_one_epoch(self, epoch_index: int) -> float:
        """
        1エポック分の学習を実行する。

        Args:
            epoch_index (int): 現在のエポック番号 (表示用)

        Returns:
            float: エポック全体の平均損失
        """
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_index+1} [Train]", leave=False)
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 勾配の初期化
            self.optimizer.zero_grad()
            
            # 順伝播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 逆伝播と更新
            loss.backward()
            self.optimizer.step()
            
            loss_val = loss.item()
            running_loss += loss_val
            
            # tqdmの表示更新
            pbar.set_postfix({'loss': f"{loss_val:.4f}"})
            
            # ステップごとのログ (任意だが細かい変動を見たい場合に有効)
            # wandb.log({"train_step_loss": loss_val})

        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss

    def validate(self) -> Dict[str, float]:
        """
        検証データセットでモデルを評価する。

        Returns:
            Dict[str, float]: 評価指標の辞書 (Loss, Recall, Precision, F1)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="[Validate]", leave=False)
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                # 予測クラスの取得 (logits -> softmax -> argmax, または単純にargmax)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 指標計算 (Badクラス(1)に対するスコアを重視)
        avg_loss = running_loss / len(self.val_loader)
        
        # zero_division=0: 予測が一つもない場合の警告回避
        recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
        precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
        f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
        
        return {
            "val_loss": avg_loss,
            "val_recall": recall,
            "val_precision": precision,
            "val_f1": f1
        }

    def fit(self, num_epochs: int):
        """
        指定されたエポック数で学習ループを実行する。

        Args:
            num_epochs (int): 学習する総エポック数
        """
        logger.info(f"Start training for {num_epochs} epochs on {self.device}...")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 学習
            train_loss = self.train_one_epoch(epoch)
            
            # 検証
            val_metrics = self.validate()
            
            # スケジューラの更新
            if self.scheduler:
                self.scheduler.step()
            
            end_time = time.time()
            epoch_duration = end_time - start_time
            
            # ログメッセージ作成
            log_msg = (
                f"Epoch {epoch+1}/{num_epochs} "
                f"| Train Loss: {train_loss:.4f} "
                f"| Val Loss: {val_metrics['val_loss']:.4f} "
                f"| Recall: {val_metrics['val_recall']:.4f} "
                f"| Precision: {val_metrics['val_precision']:.4f} "
                f"| F1: {val_metrics['val_f1']:.4f} "
                f"| Time: {epoch_duration:.2f}s"
            )
            logger.info(log_msg)
            
            # W&Bへのログ送信
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_metrics['val_loss'],
                "val/recall": val_metrics['val_recall'],
                "val/precision": val_metrics['val_precision'],
                "val/f1": val_metrics['val_f1']
            })
            
            # ベストモデルの保存 (Recallが向上した場合)
            current_recall = val_metrics['val_recall']
            if current_recall > self.best_recall:
                logger.info(f"Recall Improved ({self.best_recall:.4f} -> {current_recall:.4f}). Saving model...")
                self.best_recall = current_recall
                
                # 保存パス
                save_path = config.WEIGHTS_DIR / f"best_model_recall_{current_recall:.4f}.pth"
                # 最新のベストモデルを上書き保存する場合
                best_model_path = config.WEIGHTS_DIR / "best_model.pth"
                
                torch.save(self.model.state_dict(), save_path)
                torch.save(self.model.state_dict(), best_model_path)
        
        logger.info("Training completed.")
        logger.info(f"Best Validation Recall: {self.best_recall:.4f}")
