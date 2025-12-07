import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

import src.config as config
import src.utils as utils
from src.dataset import get_train_val_loaders
from src.models import get_model
from src.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Defect Detection Training & Evaluation")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and run only evaluation on Test Set")
    parser.add_argument("--unzip", action="store_true", help="Unzip data before running")
    args = parser.parse_args()
    
    """
    学習プロセスのメインエントリーポイント。
    """
    
    # -----------------------------------------------------
    # 1. 初期設定
    # -----------------------------------------------------
    utils.seed_everything(config.SEED)
    device = utils.get_device()
    print(f"Using device: {device}")

    # 解凍処理(オプション)
    if args.unzip:
        # 想定されるzipパス
        zip_path = config.DATA_DIR / "dataset.zip" 
        utils.unzip_data(zip_path, config.PROCESSED_DATA_DIR)
    
    # W&Bの初期化
    wandb.init(
        project="defect-detection-patch-64", # プロジェクト名は要件に合わせて変更可
        config={
            "seed": config.SEED,
            "patch_size": config.PATCH_SIZE,
            "original_size": config.ORIGINAL_SIZE,
            "batch_size": config.BATCH_SIZE,
            "class_weights": config.CLASS_WEIGHTS,
            "device": device,
            "learning_rate": config.LEARNING_RATE, 
            "epochs": config.EPOCHS,
            "num_classes": config.NUM_CLASSES,
            "threshold": config.THRESHOLD
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
    
    if not args.eval_only:
        trainer.fit(num_epochs=cfg.epochs)
    
    # -----------------------------------------------------
    # 5. 最終評価
    # -----------------------------------------------------
    print("\nStarting Final Evaluation on Test Set...")
    
    # ベストモデルのロード
    best_model_path = config.WEIGHTS_DIR / "best_model.pth"
    if best_model_path.exists():
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model weights not found. Using current model weights.")
        
    # Test Set評価の実行
    from src.evaluator import evaluate_test_set
    
    # test_loaderが作成されているか確認
    if test_loader is not None:
        test_metrics = evaluate_test_set(
            model=model,
            test_loader=test_loader,
            device=device,
            threshold=cfg.threshold # config値またはwandb設定値を使用
        )
        
        # W&Bに記録
        wandb.log(test_metrics)
    else:
        print("Test loader is None. Skipping evaluation.")
        
    # W&B終了
    wandb.finish()

if __name__ == "__main__":
    main()
