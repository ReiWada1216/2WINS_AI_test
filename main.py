import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

import src.config as config
from src.dataset import get_train_val_loaders
from src.models import get_model
from src.trainer import Trainer

def set_seed(seed: int = 42):
    """
    再現性確保のために乱数シードを固定する。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device() -> str:
    """
    利用可能な最適なデバイスを取得する。
    Mac (MPS) > CUDA > CPU の優先順位。
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def main():
    """
    学習プロセスのメインエントリーポイント。
    """
    
    # -----------------------------------------------------
    # 1. 初期設定
    # -----------------------------------------------------
    set_seed(config.SEED)
    device = get_device()
    print(f"Using device: {device}")
    
    # W&Bの初期化
    wandb.init(
        project="defect-detection-patch-64",
        config={
            "seed": config.SEED,
            "patch_size": config.PATCH_SIZE,
            "original_size": config.ORIGINAL_SIZE,
            "batch_size": config.BATCH_SIZE,
            "class_weights": config.CLASS_WEIGHTS,
            "device": device,
            "learning_rate": 1e-3, # ここで定義
            "epochs": 20           # 仮定値
        }
    )
    
    # W&B configからパラメータを取得（上書き可能にするため）
    cfg = wandb.config
    
    # -----------------------------------------------------
    # 2. データ準備
    # -----------------------------------------------------
    print("Preparing data loaders...")
    train_loader, val_loader, test_loader = get_train_val_loaders(
        data_dir=str(config.PROCESSED_DATA_DIR),
        batch_size=cfg.batch_size,
        val_size=0.1,  # 10%
        test_size=0.1, # 10%
        num_workers=config.NUM_WORKERS
    )
    
    if train_loader is None or val_loader is None:
        print("Error: Failed to create data loaders. Please check data directory.")
        return

    # -----------------------------------------------------
    # 3. モデル・損失関数・最適化手法の準備
    # -----------------------------------------------------
    print("Initializing model...")
    model = get_model(device=device)
    
    # クラス不均衡対策の重み
    class_weights = torch.tensor(cfg.class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    # -----------------------------------------------------
    # 4. 学習実行
    # -----------------------------------------------------
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    trainer.fit(num_epochs=cfg.epochs)
    
    # W&B終了
    wandb.finish()

if __name__ == "__main__":
    main()
