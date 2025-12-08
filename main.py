import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

import src.config as config
import src.utils as utils
from src.dataset import get_dataloaders
from src.models import get_model
from src.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Defect Detection Training & Evaluation (Phase 2: CAE)")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and run only evaluation on Test Set")
    parser.add_argument("--unzip", action="store_true", help="Unzip data before running")
    parser.add_argument("--phase1", action="store_true", help="Run in Phase 1 (Classification) mode instead of Phase 2 (AE)")
    args = parser.parse_args()
    
    # Phase判定
    is_phase2 = not args.phase1
    phase_name = "Phase 2 (AE)" if is_phase2 else "Phase 1 (Classification)"
    print(f"Starting execution in {phase_name} mode.")
    
    # プロジェクト設定
    project_name = "defect-detection-ae" if is_phase2 else "defect-detection-patch-64"
    
    # Config選択
    phase_config = config.PHASE2_CONFIG if is_phase2 else config.PHASE1_CONFIG
    print(f"Loaded config for {phase_config['phase_name']}")
    
    # -----------------------------------------------------
    # 1. 初期設定
    # -----------------------------------------------------
    utils.seed_everything(config.SEED)
    device = utils.get_device()
    print(f"Using device: {device}")

    # 解凍処理(オプション)
    if args.unzip:
        zip_path = config.DATA_DIR / "dataset.zip" 
        utils.unzip_data(zip_path, config.PROCESSED_DATA_DIR)
    
    # W&Bの初期化
    # W&Bの初期化
    # 共通設定とPhase設定をマージ
    wandb_config = {
        "seed": config.SEED,
        "patch_size": config.PATCH_SIZE,
        "original_size": config.ORIGINAL_SIZE,
        "num_classes": config.NUM_CLASSES,
        "mode": phase_name,
        **phase_config
    }
    
    wandb.init(
        project=project_name,
        config=wandb_config
    )
    
    cfg = wandb.config
    
    # -----------------------------------------------------
    # 2. データ準備
    # -----------------------------------------------------
    print("Preparing data loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=str(config.PROCESSED_DATA_DIR),
        batch_size=cfg.batch_size,
        val_ratio=0.1,
        test_ratio=0.1,
        num_workers=config.NUM_WORKERS,
        phase2=is_phase2,
        return_test_full_image=is_phase2 # Test時はFull Image (Phase 2)
    )
    
    if train_loader is None or val_loader is None:
        print("Error: Failed to create data loaders. Please check data directory.")
        return

    # -----------------------------------------------------
    # 3. モデル・損失関数・最適化手法の準備
    # -----------------------------------------------------
    print("Initializing model...")
    model = get_model(device=device, phase2=is_phase2)
    
    if is_phase2:
        # Phase 2: MSE Loss (Reconstruction)
        criterion = nn.MSELoss()
    else:
        # Phase 1: Cross Entropy Loss (Classification)
        # クラス不均衡対策の重み
        class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32).to(device)
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
        device=device,
        phase2=is_phase2,
        threshold=cfg.threshold
    )
    
    if not args.eval_only:
        trainer.fit(num_epochs=cfg.epochs)
    
    # -----------------------------------------------------
    # 5. 最終評価
    # -----------------------------------------------------
    print("\nStarting Final Evaluation on Test Set...")
    
    best_model_path = config.WEIGHTS_DIR / "best_model.pth"
    if best_model_path.exists():
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model weights not found. Using current model weights.")
        
    from src.evaluator import evaluate_test_set
    
    if test_loader is not None:
        test_metrics = evaluate_test_set(
            model=model,
            test_loader=test_loader,
            device=device,
            threshold=cfg.threshold,
            phase2=is_phase2
        )
        wandb.log(test_metrics)
    else:
        print("Test loader is None. Skipping evaluation.")
        
    wandb.finish()

if __name__ == "__main__":
    main()
