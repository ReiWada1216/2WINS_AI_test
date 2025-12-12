
import logging
import warnings
from pathlib import Path
import torch

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchvision.transforms.v2 import Resize, Compose
from anomalib.pre_processing import PreProcessor

# 警告をフィルタリング
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=== Anomalib EfficientAD Training (Python API) ===")
    print("Device: MPS (Mac M1) | Precision: Float32")

    # 1. 設定
    dataset_root = Path("./data/phase2_dataset")
    normal_dir = "train/good"
    abnormal_dir = "val/bad"
    
    # メモリ最適化のためのバッチサイズ=1 (EfficientADの制約)
    batch_size = 1 
    num_workers = 0 

    # 2. DataModule
    print(f"Setting up Folder DataModule from {dataset_root}...")
    datamodule = Folder(
        name="toyota_emblem",
        root=dataset_root,
        normal_dir=normal_dir,
        abnormal_dir=abnormal_dir,
        normal_test_dir="test/good",
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        # test_split_mode="from_dir", 
        val_split_mode="from_test",
        val_split_ratio=0.5,
        seed=42
    )

    # 3. モデル
    # 解像度設定: 384x384 (M1でのパフォーマンスと精度のバランス)
    image_size = (384, 384)
    print(f"Initializing EfficientAD (Small) with {image_size} resolution...")
    
    # プリプロセッサを作成
    transform = Compose([Resize(image_size, antialias=True)])
    pre_processor = PreProcessor(transform=transform)
    
    model = EfficientAd(
        model_size="small",
        pre_processor=pre_processor, # デフォルト(256)を上書き
    )

    # 4. コールバック & ロガー
    # Early Stopping
    from lightning.pytorch.callbacks import EarlyStopping
    early_stopping = EarlyStopping(
        monitor="train_loss_epoch",
        patience=10,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="./weights/efficient_ad",
        filename="best_model",
        monitor="train_loss_epoch", # Training lossを監視
        mode="min",
        save_last=True,
        save_top_k=1
    )

    wandb_logger = WandbLogger(
        project="2wins_anomalib",
        name="efficientad_mps_optimized",
        log_model=False
    )

    # 5. エンジン (Trainer)
    print("Initializing Engine with MPS accelerator...")
    engine = Engine(
        accelerator="mps",
        devices=1,
        precision="32-true", # MPSでのFloat32強制
        max_epochs=100, 
        callbacks=[checkpoint_callback, early_stopping],
        logger=wandb_logger,
        default_root_dir="./results/efficient_ad",
        check_val_every_n_epoch=1,
        log_every_n_steps=10
    )

    # 6. 学習
    print("Starting Training...")
    try:
        engine.fit(model=model, datamodule=datamodule)
    except Exception as e:
        logger.error(f"学習に失敗しました: {e}")
        # スタックトレースを表示してデバッグしやすくする
        import traceback
        traceback.print_exc()
        raise e

    # 7. テスト
    print("Starting Testing...")
    try:
        engine.test(model=model, datamodule=datamodule)
    except Exception as e:
        logger.error(f"テストに失敗しました: {e}")
    
    print("=== Finished ===")

if __name__ == "__main__":
    main()
