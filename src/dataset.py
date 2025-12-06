import os
import glob
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# 設定のインポート
try:
    import src.config as config
except ImportError:
    # src.configが見つからない場合のスタンドアロンテスト用のフォールバックまたはモック

    class Config:
        IMAGE_SIZE = 256
        VALIDATION_SPLIT = 0.2
        BATCH_SIZE = 32
        SEED = 42
    config = Config()

# -----------------------------------------------------------------------------
# 1. Transforms（データ拡張）の定義
# -----------------------------------------------------------------------------
def get_train_transforms(size: int) -> transforms.Compose:
    """
    学習用のデータ拡張パイプラインを返します。
    ランダム回転、反転、色調補正を含みます。
    """

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])

def get_val_transforms(size: int) -> transforms.Compose:
    """
    検証用の最小限の前処理パイプラインを返します。
    リサイズ、ToTensor、正規化のみを行います。
    """

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])

# -----------------------------------------------------------------------------
# 2. カスタム Dataset クラス
# -----------------------------------------------------------------------------
class CustomDataset(Dataset):
    """
    パスから画像を読み込むためのカスタム Dataset。
    """
    def __init__(self, image_paths: List[str], labels: List[int], transform: Optional[transforms.Compose] = None):
        """
        Args:
            image_paths (List[str]): 画像へのファイルパスのリスト。
            labels (List[int]): 対応するラベルのリスト（0 または 1）。
            transform (callable, optional): サンプルに適用されるオプションの変換。
        """

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 画像を読み込み、3チャンネルを確保するためにRGBに変換
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Error loading image {img_path}: {e}")
            # 実際のシナリオでは、より適切に処理したい場合があります
            # 今のところは再送出するか、必要であればゼロテンソルを返しますが、再送出の方が安全です
            raise e


        if self.transform:
            image = self.transform(image)

        # 画像とラベルを返す
        # ラベルはBCEWithLogitsLoss用にfloat、またはCrossEntropyLoss用にlongに変換されます
        # 通常、出力ニューロンが1つの二値分類ではfloatが使用されます。

        return image, torch.tensor(label, dtype=torch.float32)

# -----------------------------------------------------------------------------
# 3. データローダー関数
# -----------------------------------------------------------------------------
def get_train_val_loaders(data_dir: str, config_module) -> Tuple[DataLoader, DataLoader]:
    """
    層化サンプリングを用いて学習用と検証用のDataLoaderを作成します。

    Args:
        data_dir (str): 'good' と 'bad' フォルダを含む 'processed' ディレクトリへのパス。
        config_module: 設定（IMAGE_SIZE, BATCH_SIZE など）を含むモジュールまたはオブジェクト。

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """

    # クラスディレクトリの定義
    good_dir = os.path.join(data_dir, "good")
    bad_dir = os.path.join(data_dir, "bad")

    # すべての画像パスを収集

    good_paths = glob.glob(os.path.join(good_dir, "*.png"))
    bad_paths = glob.glob(os.path.join(bad_dir, "*.png"))

    # ラベルの作成: goodは0, badは1
    good_labels = [0] * len(good_paths)
    bad_labels = [1] * len(bad_paths)

    all_paths = good_paths + bad_paths
    all_labels = good_labels + bad_labels

    print(f"Total images found: {len(all_paths)} (Good: {len(good_paths)}, Bad: {len(bad_paths)})")

    # 層化サンプリング（Stratified Split）
    # 学習セットと検証セットの両方でgood/badの比率を維持

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=config_module.VALIDATION_SPLIT,
        stratify=all_labels,
        random_state=config_module.SEED
    )

    print(f"Train set size: {len(train_paths)}")
    print(f"Validation set size: {len(val_paths)}")

    # Datasetの初期化

    train_dataset = CustomDataset(
        image_paths=train_paths,
        labels=train_labels,
        transform=get_train_transforms(config_module.IMAGE_SIZE)
    )

    val_dataset = CustomDataset(
        image_paths=val_paths,
        labels=val_labels,
        transform=get_val_transforms(config_module.IMAGE_SIZE)
    )

    # DataLoaderの初期化

    train_loader = DataLoader(
        train_dataset,
        batch_size=config_module.BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count() or 2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config_module.BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count() or 2,
        pin_memory=True
    )

    return train_loader, val_loader

if __name__ == "__main__":
    # コードが動作することを確認するための簡単なテストブロック
    print("Testing dataset.py...")

    try:
        # テスト目的でデータディレクトリが存在しない場合のモック

        test_data_dir = "data/processed"
        if not os.path.exists(test_data_dir):
            print(f"Data directory {test_data_dir} not found. Skipping runtime test.")
        else:
            train_loader, val_loader = get_train_val_loaders(test_data_dir, config)
            print("DataLoaders created successfully.")
            
            # 1バッチ分を反復処理

            for images, labels in train_loader:
                print(f"Batch shape: {images.shape}")
                print(f"Labels shape: {labels.shape}")
                break
    except Exception as e:
        print(f"An error occurred during testing: {e}")
