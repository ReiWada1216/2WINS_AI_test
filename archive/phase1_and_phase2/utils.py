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

# ==========================================
# SSIM Implementation (Differentiable)
# ==========================================

import torch.nn.functional as F
from math import exp

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """
    1D Gaussian kernel generation.
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size: int, channel: int) -> torch.Tensor:
    """
    Create 2D Gaussian window for Conv2d.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, window_size: int, channel: int, size_average: bool = True) -> torch.Tensor:
    """
    SSIM Calculation Core.
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(torch.nn.Module):
    """
    Structural Similarity Loss (1 - SSIM)
    Expects input range [0, 1].
    """
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            elif img1.device.type == 'mps':
                 window = window.to(img1.device)
            
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return 1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average)

