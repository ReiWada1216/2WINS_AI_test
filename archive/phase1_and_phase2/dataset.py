import os
from pathlib import Path
from typing import Optional, Tuple, List, Callable
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

import src.config as config

# ==========================================
# Transforms
# ==========================================

def get_transforms(mode: str = "train") -> transforms.Compose:
    """
    データ変換パイプラインを取得する関数。
    RGB -> Grayscale -> Tensor (0-1)
    
    Args:
        mode (str): 'train' or 'eval'
    """
    ops = []
    
    # 共通: グレースケール変換 (1ch) -> Tensor
    # NOTE: MVTecPatchDataset内でPILロード時にconvert('L')するため、
    # ここでは主にTensor化を行う。
    
    if mode == "train":
        # 学習時のみAugmentation (必要であれば)
        # Autoencoderの再構成タスクでは、極端な幾何学的変換は控えるが、
        # FlipやRotationは有効な場合がある。
        ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomRotation(degrees=15), 
        ])

    ops.append(transforms.ToTensor())
    # ToTensor() converts [0, 255] -> [0.0, 1.0]

    return transforms.Compose(ops)


# ==========================================
# Dataset
# ==========================================

class MVTecImageDataset(Dataset):
    """
    推論・評価用データセット。
    1024x1024の画像全体を返す。
    """
    
    def __init__(self, 
                 image_paths: List[str], 
                 labels: List[int],
                 transform: Optional[Callable] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        画像全体を取得する。
        Return:
            img_tensor: (1, 1024, 1024)
            label: int (0 or 1)
            path: str
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            with Image.open(img_path) as img:
                img = img.convert("L")
                if self.transform:
                    img = self.transform(img)
                return img, label, img_path
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Dummy return
            return torch.zeros((1, config.ORIGINAL_SIZE, config.ORIGINAL_SIZE)), label, img_path

class MVTecPatchDataset(Dataset):
    """
    M1 Mac (メモリ制約) 向け高効率パッチデータセット。
    1024x1024の画像をメモリに展開せず、__getitem__で必要な256x256パッチを
    オンザフライでロード・クロップする。
    """
    
    def __init__(self, 
                 image_paths: List[str], 
                 labels: List[int],
                 transform: Optional[Callable] = None,
                 patch_size: int = config.PATCH_SIZE,
                 stride: int = config.STRIDE,
                 original_size: int = config.ORIGINAL_SIZE):
        
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.original_size = original_size
        
        # パッチメタデータの生成
        # リスト形式: (image_index, top, left)
        # path文字列を繰り返すとメモリを食うため、indexで管理する。
        self.patches = []
        
        # パッチ座標の計算
        coords = []
        current = 0
        while current + patch_size <= original_size:
            coords.append(current)
            current += stride
            
        for idx, _ in enumerate(image_paths):
            # Optimization: Load image once per file
            img_path = self.image_paths[idx]
            try:
                with Image.open(img_path) as img:
                    img = img.convert("L")
                    
                    for top in coords:
                        for left in coords:
                            # Pre-check: Crop and calculate mean
                            patch = img.crop((left, top, left + self.patch_size, top + self.patch_size))
                            # Convert to numpy for fast mean calculation
                            # Only keep if mean intensity > threshold
                            if np.array(patch).mean() > config.BACKGROUND_THRESHOLD:
                                self.patches.append((idx, top, left))
            except Exception as e:
                print(f"Warning: Failed to load {img_path} during init: {e}")
                continue

                    
    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        パッチ画像を取得する。
        Return:
            img_tensor: (1, 256, 256)
            label: int (0 or 1)
        """
        img_idx, top, left = self.patches[idx]
        img_path = self.image_paths[img_idx]
        label = self.labels[img_idx]
        
        try:
            # グレースケールで開く ('L')
            with Image.open(img_path) as img:
                img = img.convert("L")
                
                # クロップ (left, top, right, bottom)
                patch = img.crop((left, top, left + self.patch_size, top + self.patch_size))
                
                if self.transform:
                    patch = self.transform(patch)
                    
                return patch, label
                
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((1, self.patch_size, self.patch_size)), label


# ==========================================
# Data Loading & Splitting
# ==========================================

def get_dataloaders(
    data_dir: str = str(config.PROCESSED_DATA_DIR),
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    bad_ratio: float = 0.35, # Goodに対するBadの比率 (Val/Test)
    batch_size: int = 32,
    num_workers: int = config.NUM_WORKERS,
    # 互換性のためのダミー引数 (main.pyからの呼び出しに対応する場合)
    phase2: bool = True,
    return_test_full_image: bool = False # Testセットをパッチではなくフル画像で返すか
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    データ分割とDataLoaderの作成を行う。
    """
    
    search_path = Path(data_dir)
    class_map = {"good": 0, "bad": 1}
    
    good_paths = []
    bad_paths = []
    
    # 画像収集
    if search_path.exists():
        for class_name, label_id in class_map.items():
            class_dir = search_path / class_name
            if class_dir.exists():
                files = sorted([str(f) for f in class_dir.glob("*.png")])
                if label_id == 0:
                    good_paths.extend(files)
                else:
                    bad_paths.extend(files)
    
    if not good_paths:
        print(f"Error: No 'good' images found in {data_dir}")
        return None, None, None
        
    # --- Split Logic ---
    
            
    # --- Split Logic ---
    
    # 1. Good Split (80/10/10)
    train_good, temp_good = train_test_split(
        good_paths, test_size=(val_ratio + test_ratio), random_state=config.SEED, shuffle=True
    )
    
    val_test_ratio = test_ratio / (val_ratio + test_ratio) # 0.5
    val_good, test_good = train_test_split(
        temp_good, test_size=val_test_ratio, random_state=config.SEED, shuffle=True
    )
    
    # 2. Bad Split
    # Phase 1 (Classification): Bad data required in Train
    # Phase 2 (Anomaly Detection): Bad data only in Val/Test
    
    train_bad = []
    val_bad = []
    test_bad = []
    
    if len(bad_paths) > 0:
        if not phase2:
            # Phase 1: Split Bad into Train/Val/Test (Same ratio as Good)
            train_bad_all, temp_bad = train_test_split(
                bad_paths, test_size=(val_ratio + test_ratio), random_state=config.SEED, shuffle=True
            )
            val_bad_all, test_bad_all = train_test_split(
                temp_bad, test_size=val_test_ratio, random_state=config.SEED, shuffle=True
            )
            train_bad = train_bad_all
            val_bad = val_bad_all
            test_bad = test_bad_all
        else:
            # Phase 2: Bad used only for Val/Test (Anomaly metrics)
            # Use 'bad_ratio' to control amount relative to Good samples in Val/Test
            n_val_bad = int(len(val_good) * bad_ratio)
            n_test_bad = int(len(test_good) * bad_ratio)
            total_needed_bad = n_val_bad + n_test_bad
            
            random.seed(config.SEED)
            random.shuffle(bad_paths)
            
            if len(bad_paths) < total_needed_bad:
                split_idx = len(bad_paths) // 2
                val_bad = bad_paths[:split_idx]
                test_bad = bad_paths[split_idx:]
            else:
                val_bad = bad_paths[:n_val_bad]
                test_bad = bad_paths[n_val_bad:n_val_bad+n_test_bad]
    else:
        print("Warning: No bad images found. Val/Test will only contain good images.")
            
    # パスリストとラベルリストの作成
    # Train
    train_paths = train_good + train_bad
    train_labels = [0] * len(train_good) + [1] * len(train_bad)
    
    # Val
    val_paths = val_good + val_bad
    val_labels = [0] * len(val_good) + [1] * len(val_bad)
    
    # Test
    test_paths = test_good + test_bad
    test_labels = [0] * len(test_good) + [1] * len(test_bad)
    
    print(f"Dataset Split Summary (Phase 2={phase2}):")
    print(f"  Train: {len(train_paths)} (Good: {len(train_good)}, Bad: {len(train_bad)})")
    print(f"  Val  : {len(val_paths)} (Good: {len(val_good)}, Bad: {len(val_bad)})")
    print(f"  Test : {len(test_paths)} (Good: {len(test_good)}, Bad: {len(test_bad)})")
    
    # Dataset作成
    train_dataset = MVTecPatchDataset(train_paths, train_labels, transform=get_transforms("train"))
    val_dataset = MVTecPatchDataset(val_paths, val_labels, transform=get_transforms("eval"))
    
    if return_test_full_image:
        test_dataset = MVTecImageDataset(test_paths, test_labels, transform=get_transforms("eval"))
    else:
        test_dataset = MVTecPatchDataset(test_paths, test_labels, transform=get_transforms("eval"))
    
    # DataLoader作成
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_batch_size = 1 if return_test_full_image else batch_size
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test
    # ダミーファイルを作らないとテスト落ちるので、ここではインポートのみ確認
    print("dataset.py module loaded.")
