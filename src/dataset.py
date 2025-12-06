import os
from pathlib import Path
from typing import Optional, Tuple, List, Callable


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

import src.config as config

def get_train_transforms(size: int = config.PATCH_SIZE) -> transforms.Compose:
    """
    学習用データ変換を取得する関数。
    
    パッチベース学習のために、大きな元画像からランダムに切り出す処理を行う。
    
    Args:
        size (int): 切り出すパッチのサイズ (デフォルト: config.PATCH_SIZE)
        
    Returns:
        transforms.Compose: 適用する変換パイプライン
    """
    return transforms.Compose([
        # ここでランダムに64x64のパッチを切り出す
        transforms.RandomCrop(size),
        
        # データ拡張
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        
        # Tensor化と正規化
        transforms.ToTensor(),
        # ImageNetの平均・標準偏差を使用するのが一般的だが、独自のデータセットの場合は計算した方が良い場合もある
        # ここでは一般的な値を使用
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(size: int = config.PATCH_SIZE) -> transforms.Compose:
    """
    検証用データ変換を取得する関数。
    
    検証時はランダム性、特にAugmentationを排除するが、
    パッチ切り出しについては要件に従いCenterCropなどを使用する。
    
    Args:
        size (int): 切り出すパッチのサイズ (デフォルト: config.PATCH_SIZE)
        
    Returns:
        transforms.Compose: 適用する変換パイプライン
    """
    return transforms.Compose([
        # 検証用は中央を切り出す（要件準拠）
        transforms.CenterCrop(size),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

class CustomDataset(Dataset):
    """
    製品外観検査用カスタムデータセットクラス。
    画像パスのリストと対応するラベルを受け取り、データをロードして変換を適用する。
    """
    
    def __init__(self, 
                 image_paths: List[str], 
                 labels: List[int], 
                 transform: Optional[Callable] = None):
        """
        初期化メソッド。
        
        Args:
            image_paths (List[str]): 画像ファイルパスのリスト
            labels (List[int]): 各画像に対応するラベルのリスト (0: Good, 1: Bad)
            transform (Optional[Callable]): 画像に適用する変換 (torchvision.transforms)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """
        データセットのサイズを返す。
        
        Returns:
            int: データ数
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        指定されたインデックスのデータ（画像とラベル）を取得する。
        
        Args:
            idx (int): データのインデックス
            
        Returns:
            Tuple[torch.Tensor, int]: 変換後の画像テンソルとラベル
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 画像読み込み (PILを使用)
        # OpenCVで読み込んでPILに変換する方法もあるが、torchvisionはPILとの親和性が高い
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # エラー時は真っ黒な画像を返す等のハンドリングが考えられるが、
            # ここではエラーを再送出する
            raise e
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_train_val_loaders(
    data_dir: str = str(config.PROCESSED_DATA_DIR),
    val_size: float = 0.1,  # 全体の10%
    test_size: float = 0.1, # 全体の10%
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    学習・検証・テストデータのDataLoaderを作成して返す関数。
    sklearn.model_selection.train_test_split を2回使用して、
    Train(80%) / Val(10%) / Test(10%) の層化分割を行う。
    
    Args:
        data_dir (str): データディレクトリのパス
        val_size (float): 検証データの割合 (全体に対する割合, default: 0.1)
        test_size (float): テストデータの割合 (全体に対する割合, default: 0.1)
        batch_size (int): バッチサイズ
        num_workers (int): データ読み込みのワーカー数
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, Val, Test Loaders
    """
    
    # 画像パスとラベルのリストを作成
    image_paths = []
    labels = []
    
    # ディレクトリ構成を走査
    search_path = Path(data_dir)
    
    class_map = {
        "good": 0,
        "bad": 1
    }
    
    found_files = False
    
    if search_path.exists():
        for class_name, label_id in class_map.items():
            class_dir = search_path / class_name
            if class_dir.exists():
                files = list(class_dir.glob("*.png")) # PNG指定
                for f in files:
                    image_paths.append(str(f))
                    labels.append(label_id)
                if len(files) > 0:
                    found_files = True
    
    if not found_files:
        print(f"Warning: No images found in {data_dir} with 'good'/'bad' subdirectories.")
        print("Creating dummy data for validation purposes if needed...")
        # 実装動作確認用ダミーデータ生成ロジックが必要ならここに追加
    
    # データが見つかった場合のみ分割
    if len(image_paths) > 0:
        # Step 1: 全体を Train と Temp (Val + Test) に分割
        # Tempの割合 = val_size + test_size
        temp_size = val_size + test_size
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            image_paths, 
            labels, 
            test_size=temp_size, 
            random_state=config.SEED, 
            stratify=labels
        )
        
        # Step 2: Temp を Val と Test に分割
        # Temp内でのTestの割合 = test_size / temp_size
        # 例: 0.1 / 0.2 = 0.5 (50%)
        test_size_in_temp = test_size / temp_size
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, 
            y_temp, 
            test_size=test_size_in_temp, 
            random_state=config.SEED, 
            stratify=y_temp
        )
        
        print(f"Data Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Dataset作成
        train_dataset = CustomDataset(
            image_paths=X_train, 
            labels=y_train, 
            transform=get_train_transforms()
        )
        
        val_dataset = CustomDataset(
            image_paths=X_val, 
            labels=y_val, 
            transform=get_val_transforms()
        )

        test_dataset = CustomDataset(
            image_paths=X_test, 
            labels=y_test, 
            transform=get_val_transforms() # TestもValと同じTransform (Augmentationなし)
        )
        
        # DataLoader作成
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    else:
        return None, None, None

if __name__ == "__main__":
    # 簡易テスト
    print("Testing dataset module...")
    t = get_train_transforms()
    print(f"Train transforms: {t}")
    
    # ダミーデータローダー取得テスト（パスが存在しないとNoneが返る可能性が高いがコードとして通るか確認）
    tr_loader, val_loader, test_loader = get_train_val_loaders()
    if tr_loader:
        print(f"Train loader created with batch size {tr_loader.batch_size}")
    else:
        print("No dataloader created (data directory might be empty).")
