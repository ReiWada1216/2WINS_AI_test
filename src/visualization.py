import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from typing import Optional, Tuple, List
from pathlib import Path

import src.config as config

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) の手動実装クラス。
    
    CNNの最後の畳み込み層の勾配を使用して、予測に重要な領域を可視化する。
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        初期化メソッド。
        
        Args:
            model (nn.Module): 評価対象のモデル
            target_layer (nn.Module): 勾配を取得するターゲット層 (通常は最後のConv層)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        
        # フックの登録
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        """順伝播時の特徴マップを保存するフック関数"""
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        """逆伝播時の勾配を保存するフック関数"""
        # grad_output はタプルなので最初の要素を取得
        self.gradients = grad_output[0].detach()
        
    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Grad-CAMヒートマップを生成する。
        
        Args:
            x (torch.Tensor): 入力画像テンソル (Batch, C, H, W)
            class_idx (Optional[int]): ターゲットクラスのインデックス。Noneの場合は予測最大クラスを使用。
            
        Returns:
            np.ndarray: 生成されたヒートマップ (H, W), 0-1正規化済み
        """
        # モデルを評価モードに
        self.model.eval()
        
        # 勾配計算のため requires_grad を有効化して順伝播
        # 既存の勾配をクリア
        self.model.zero_grad()
        
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        # ターゲットクラスのスコアを取得
        target = output[0, class_idx]
        
        # 逆伝播を実行して勾配を計算
        target.backward()
        
        # GAP (Global Average Pooling) で重みを計算
        # gradients shape: (Batch, Channel, H, W) -> pooled_gradients shape: (Batch, Channel)
        # 平均をとって重みとする (alpha_k)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # 特徴マップに重みをかけて加重和をとる
        # activations shape: (Batch, Channel, H, W)
        activations = self.activations[0] # バッチ次元を削除
        
        # 重み付け: チャネル次元でfor文を回すと遅いので、ブロードキャストを利用
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        # チャネル方向の和 (ヒートマップの素)
        heatmap = torch.sum(activations, dim=0).cpu().numpy()
        
        # ReLUを適用 (正の影響のみ残す)
        heatmap = np.maximum(heatmap, 0)
        
        # 正規化 (0〜1)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        # 入力画像サイズにリサイズ
        # cv2.resize は (width, height)
        # heatmap (H, W) -> Image -> resize
        heatmap_img = Image.fromarray(heatmap)
        heatmap_img = heatmap_img.resize((x.shape[3], x.shape[2]), resample=Image.BICUBIC)
        heatmap = np.array(heatmap_img)
        
        return heatmap

def apply_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    元の画像にヒートマップを重ね合わせる。
    
    Args:
        image (np.ndarray): 元画像 (H, W, 3), 0-255 RGB
        heatmap (np.ndarray): ヒートマップ (H, W), 0-1
        alpha (float): ヒートマップの透明度
        
    Returns:
        np.ndarray: 重ね合わせ画像 (RGB)
    """
    # ヒートマップをカラーマップ(JET)に変換
    # matplotlibのcolormapを使用
    colormap = cm.get_cmap('jet')
    heatmap_color = colormap(heatmap) # Returns RGBA (0-1)
    
    # RGBAからRGBへ (0-255)
    heatmap_color = (heatmap_color[:, :, :3] * 255).astype(np.uint8)
    
    # 重ね合わせ
    overlay = heatmap_color * alpha + image * (1 - alpha)
    overlay = np.minimum(overlay, 255).astype(np.uint8)
    
    return overlay

def visualize_gradcam(
    model: nn.Module, 
    image_tensor: torch.Tensor, 
    original_image: np.ndarray,
    save_path: Path,
    device: str,
    target_class: int = 1 # Bad Class
):
    """
    指定された画像のGrad-CAMを生成して保存するラッパー関数。
    
    Args:
        model (nn.Module): 学習済みモデル
        image_tensor (torch.Tensor): モデル入力用テンソル (1, C, H, W)
        original_image (np.ndarray): 表示用元画像 (H, W, 3) RGB
        save_path (Path): 保存先パス
        device (str): デバイス
        target_class (int): 可視化対象のクラス
    """
    # ResNet18のlayer4 (最終Convブロック) をターゲットにする
    target_layer = model.model.layer4[-1]
    
    grad_cam = GradCAM(model, target_layer)
    
    image_tensor = image_tensor.to(device)
    
    # Grad-CAM生成
    heatmap = grad_cam(image_tensor, class_idx=target_class)
    
    # 画像サイズ合わせ (パッチサイズの場合と元画像サイズの場合があるため注意)
    # ここでは original_image のサイズに heatmap を合わせる
    heatmap_img = Image.fromarray(heatmap)
    heatmap_img = heatmap_img.resize((original_image.shape[1], original_image.shape[0]), resample=Image.BICUBIC)
    heatmap = np.array(heatmap_img)
    
    # 重ね合わせ
    result_image = apply_heatmap(original_image, heatmap)
    
    # 保存 (Matplotlibを使用)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("元画像")
    plt.imshow(original_image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Grad-CAM (クラス {target_class})")
    plt.imshow(result_image)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Grad-CAM可視化画像を保存しました: {save_path}")

if __name__ == "__main__":
    # 簡易テスト
    print("Testing visualization module...")
    from src.models import SimpleResNet
    
    model = SimpleResNet()
    # ダミー入力
    dummy_input = torch.randn(1, 3, 224, 224)
    dummy_original = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    save_path = config.RESULTS_DIR / "dummy_gradcam_test.png"
    
    try:
        visualize_gradcam(model, dummy_input, dummy_original, save_path, "cpu")
        print("Success.")
    except Exception as e:
        print(f"Failed: {e}")
