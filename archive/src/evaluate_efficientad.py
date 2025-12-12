
import logging
import warnings
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from anomalib.data import Folder
from anomalib.models import EfficientAd
from anomalib.engine import Engine
from anomalib.deploy import ExportType
from torchmetrics import ROC, F1Score


# 警告をフィルタリング
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=== EfficientAD Evaluation ===")
    
    # 1. 設定
    dataset_root = Path("./data/phase2_dataset")
    normal_dir = "train/good"
    abnormal_dir = "test/bad"
    
    # チェックポイントのパス (学習後に生成されるパスを指定)
    ckpt_path = Path("/Users/wadarei/2WINS_AI_test/efficientAD_rocauc_0.983.ckpt")
    
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    # 2. DataModule (学習時と同じ設定)
    from torchvision.transforms.v2 import Compose, Resize
    from anomalib.pre_processing import PreProcessor
    import torch

    # 学習時と同じPreProcessor設定を使用 (384x384 Ref: train_efficientad.py)
    transform = Compose([
        Resize((384, 384), antialias=True),
    ])
    pre_processor = PreProcessor(transform=transform)

    # 2. DataModule
    datamodule = Folder(
        name="toyota_emblem",
        root=dataset_root,
        normal_dir=normal_dir,
        abnormal_dir=abnormal_dir,
        normal_test_dir="test/good",
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
        val_split_mode="from_test",
        val_split_ratio=0.5,
        # test_augmentations=transform, # Model側のPreProcessorに任せる
        seed=42
    )
    datamodule.setup(stage="test")
    test_dataloader = datamodule.test_dataloader()

    # 3. モデルのロード
    print(f"Loading model state dict from {ckpt_path}...")
    
    # PreProcessorとpad_mapsを明示的に指定して初期化
    model = EfficientAd(
        model_size="small", 
        pad_maps=True,
        pre_processor=pre_processor
    )
    
    # チェックポイント読み込み
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    
    # MPSトラブル回避のためCPU強制
    device = torch.device("cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # 4. エンジンの初期化 (テスト用)
    engine = Engine(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        default_root_dir="./results/efficient_ad_eval",
        logger=False,
    )

    # 5. データとラベルの確認 (デバッグ)
    print("\n=== Checking Data & Labels ===")
    labels_list = []
    
    # 最初の数バッチだけ確認
    count = 0
    with torch.no_grad():
        for batch in test_dataloader:
            if count < 5:
                # 画像パスとラベルを表示
                # print(f"Sample {count}: Label={batch['label'].item()}, Path={batch['image_path'][0]}")
                pass
            labels_list.append(batch.gt_label.item())
            count += 1
            
    unique_labels = set(labels_list)
    print(f"Total test images: {len(labels_list)}")
    print(f"Unique labels found: {unique_labels}")
    print(f"Label distribution: 0(Good)={labels_list.count(0)}, 1(Bad)={labels_list.count(1)}")
    
    if len(unique_labels) < 2:
        print("WARNING: Only one class found in test set! ROC AUC will be undefined (0.5).")

    # 6. テスト実行 (Anomalibの標準テスト)
    print("\n=== Running standard Anomalib test ===")
    engine.test(model=model, dataloaders=test_dataloader)
    
    # 6. 推論レイテンシの計測
    print("\n=== Measuring Inference Latency ===")
    latencies = []
    
    # ウォームアップ
    dummy_input = torch.randn(1, 3, 384, 384).to(device)
    for _ in range(10):
        with torch.no_grad():
            model(dummy_input)
            
    # 計測
    count = 0
    max_count = 100 # 計測する枚数
    
    with torch.no_grad():
        for batch in test_dataloader:
            if count >= max_count:
                break
            
            image = batch["image"].to(device)
            
            start_time = time.time()
            model(image)
            end_time = time.time()
            
            latencies.append((end_time - start_time) * 1000) # ms
            count += 1
            
    avg_latency = sum(latencies) / len(latencies)
    print(f"Average Inference Latency: {avg_latency:.2f} ms")
    print(f"Processed {len(latencies)} images.")

if __name__ == "__main__":
    main()
