import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize, Compose
from torchvision.transforms.functional import InterpolationMode
from anomalib.data import Folder, PredictDataset
from anomalib.models import EfficientAd
from anomalib.engine import Engine
from anomalib.pre_processing import PreProcessor
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import cv2

# Fix for PyTorch 2.6+ checkpoint loading
torch.serialization.add_safe_globals([PreProcessor, Compose, Resize, InterpolationMode])

def visualize_results():
    print("=== Visualization Started ===")
    
    # 1. 設定
    ckpt_path = Path("results/efficient_ad/efficient_ad_v1_best/weights/efficient_v1_rocauc_0.983.ckpt")
    image_size = (384, 384) 
    output_dir = Path("results/visualization_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. モデルのロード
    print(f"Loading model from {ckpt_path}...")
    # 学習時と同じPreProcessor設定
    transform = Compose([Resize(image_size, antialias=True)])
    pre_processor = PreProcessor(transform=transform)
    
    model = EfficientAd.load_from_checkpoint(
        str(ckpt_path),
        map_location=device,
        pre_processor=pre_processor,
        weights_only=False
    )
    model.eval()
    
    # 3. データセット (テスト用)
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

    # 4. Engineで推論
    engine = Engine(
        accelerator="mps",
        devices=1,
        default_root_dir=output_dir,
    )
    
    print("Running prediction on test set...")
    # 最初の1バッチ(全データ)を取得するため predict を実行
    predictions = engine.predict(model=model, dataloaders=dataloader)
    
    # 5. 可視化と保存 (手動実装)
    print("Saving visualizations...")
    
    count = 0
    # predictionsはバッチごとのリスト
    # predictionsはバッチごとのリスト
    for batch_idx, pred in enumerate(predictions):
        # pred は dict-like なオブジェクト
        
        # バッチ内の各画像について処理
        batch_size_actual = len(pred["image_path"])
        
        for i in range(batch_size_actual):
            # 画像パス
            img_path = Path(pred["image_path"][i])
            label_str = "Anomaly" if "bad" in str(img_path) else "Normal"
            
            # オリジナル画像
            original_img = Image.open(img_path).convert("RGB").resize(image_size)
            original_np = np.array(original_img)
            
            # Anomaly Map (Heatmap)
            # torch tensor -> numpy
            # [H, W] の形状のはず
            anomaly_map = pred["anomaly_map"][i].cpu().numpy()
            
            # スコア
            pred_score = pred["pred_score"][i].item()
            
            # ヒートマップの作成 (Jet colormapで重ね合わせ)
            # 0-1に正規化 (可視化用)
            anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
            heatmap = cv2.applyColorMap(np.uint8(anomaly_map_norm * 255), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # 重ね合わせ (alpha blend)
            alpha = 0.4
            overlay = (original_np * (1 - alpha) + heatmap * alpha).astype(np.uint8)
            
            # Plot保存
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # 1. Original
            axes[0].imshow(original_np)
            axes[0].set_title(f"Original: {img_path.name}")
            axes[0].axis("off")
            
            # 2. Heatmap Only
            axes[1].imshow(anomaly_map, cmap="jet")
            axes[1].set_title("Anomaly Map")
            axes[1].axis("off")
            
            # 3. Overlay
            axes[2].imshow(overlay)
            axes[2].set_title(f"Overlay\nScore: {pred_score:.4f} ({label_str})")
            axes[2].axis("off")
            
            save_path = output_dir / f"{label_str}_{img_path.stem}_vis.png"
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)
            
            print(f"Saved: {save_path}")
            count += 1
            if count >= 20: # 全部やると多いので20枚で止める
                break
        if count >= 20:
            break

    print("=== Visualization Completed ===")

if __name__ == "__main__":
    visualize_results()
