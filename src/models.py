import torch
import torch.nn as nn
from torchvision import models
from typing import Optional
import src.config as config

class SimpleResNet(nn.Module):
    """
    製品外観検査用ResNetモデル。
    
    標準のResNet18を使用し、224x224の入力サイズに対応する。
    転移学習の恩恵を最大化するため、アーキテクチャは変更せずそのまま利用する。
    """
    
    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        """
        初期化メソッド。
        
        Args:
            num_classes (int): 出力クラス数 (デフォルト: 2)
            pretrained (bool): ImageNetの事前学習済み重みを使用するかどうか (デフォルト: True)
        """
        super(SimpleResNet, self).__init__()
        
        # ResNet18モデルのロード
        if pretrained:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet18(weights=None)
            
        # 最終の全結合層 (fc) をクラス数に合わせて置換
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理。
        
        Args:
            x (torch.Tensor): 入力画像テンソル (Batch, 3, 224, 224)
            
        Returns:
            torch.Tensor: ロジット出力 (Batch, num_classes)
        """
        return self.model(x)

def get_model(device: str = "cpu") -> nn.Module:
    """
    モデルインスタンスを生成して指定デバイスに転送するヘルパー関数。
    
    Args:
        device (str): 利用するデバイス ('cpu', 'cuda', 'mps' 等)
            
    Returns:
        nn.Module: 初期化されたモデル
    """
    model = SimpleResNet(num_classes=config.NUM_CLASSES, pretrained=True)
    model.to(device)
    return model

if __name__ == "__main__":
    # モデルの簡易テスト
    print("Testing model module...")
    
    # ダミー入力 (Batch=2, Channels=3, H=224, W=224)
    dummy_input = torch.randn(2, 3, 224, 224)
    
    model = SimpleResNet()
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # 期待値: [2, 2]
    
    expected_shape = (2, 2)
    assert output.shape == expected_shape, f"Output shape mismatch. Expected {expected_shape}, got {output.shape}"
    print("Model test passed successfully.")
