import random
import numpy as np
import torch
import zipfile
import shutil
from pathlib import Path
import logging

def seed_everything(seed: int = 42):
    """
    再現性確保のために乱数シードを固定する。
    
    Args:
        seed (int): シード値
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
    
    Returns:
        str: デバイス名 ('mps', 'cuda', 'cpu')
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def unzip_data(src_path: Path, dest_dir: Path):
    """
    データセットのzipファイルを解凍する。
    
    Args:
        src_path (Path): 解凍元のzipファイルパス
        dest_dir (Path): 解凍先のディレクトリパス
    """
    if not src_path.exists():
        print(f"Zip file not found: {src_path}")
        return

    print(f"Unzipping {src_path} to {dest_dir}...")
    try:
        with zipfile.ZipFile(src_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print("Unzip completed.")
    except Exception as e:
        print(f"Error unzipping file: {e}")
