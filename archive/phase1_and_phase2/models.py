import torch
import torch.nn as nn
from typing import Optional
import src.config as config

class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2, padding: int = 1, output_padding: int = 1):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv_t(x)))

class ConvolutionalAutoEncoder(nn.Module):
    """
    軽量CAE (Convolutional AutoEncoder)
    M1 Mac上での学習効率を考慮し、パラメータ数を抑えつつ特徴抽出能力を維持する設計。
    Input: (B, 1, 256, 256) -> Output: (B, 1, 256, 256)
    """
    
    def __init__(self):
        super(ConvolutionalAutoEncoder, self).__init__()
        
        # =======================
        # Encoder
        # =======================
        # Input: (B, 1, 256, 256)
        self.enc1 = EncoderBlock(1, 32)       # -> (32, 128, 128)
        self.enc2 = EncoderBlock(32, 64)      # -> (64, 64, 64)
        self.enc3 = EncoderBlock(64, 128)     # -> (128, 32, 32)
        self.enc4 = EncoderBlock(128, 256)    # -> (256, 16, 16)
        
        # Bottleneck: 空間解像度維持 (16x16), チャンネル圧縮 (256->64)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        ) # -> (64, 16, 16)
        
        # =======================
        # Decoder
        # =======================
        # Reverse Bottleneck: (64 -> 256)
        self.dec_bottleneck = nn.Sequential(
            nn.ConvTranspose2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ) # -> (256, 16, 16)
        
        # Upsampling
        # NOTE: output_padding=1 ensures doubling when k=3, s=2, p=1
        self.dec4 = DecoderBlock(256, 128)    # -> (128, 32, 32)
        self.dec3 = DecoderBlock(128, 64)     # -> (64, 64, 64)
        self.dec2 = DecoderBlock(64, 32)      # -> (32, 128, 128)
        
        # Final Output Layer
        self.final_conv = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        # -> (1, 256, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.bottleneck(x)
        
        # Decoder
        x = self.dec_bottleneck(x)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)
        
        return x

class SimpleCNN(nn.Module):
    """
    Phase 1 (分類) 用の単純なCNNモデル。
    パッチ画像 (256x256) を入力とし、良品(0)/不良品(1)の2クラス分類を行う。
    Input: (B, 1, 256, 256) -> Output: (B, 2)
    """
    def __init__(self, num_classes: int = 2):
        super(SimpleCNN, self).__init__()
        
        # Feature Extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # -> 128x128
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # -> 64x64
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # -> 32x32
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # -> 16x16
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1)) # -> (B, 256, 1, 1)
        )
        
        # Classifier
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x

def get_model(device: str = "cpu", phase2: bool = True) -> nn.Module:
    """
    モデルインスタンスを生成して指定デバイスに転送するヘルパー関数。
    Args:
        device (str): 'cpu', 'cuda', 'mps'
        phase2 (bool): TrueならAE (Phase 2), FalseならClassifier (Phase 1)
    """
    if phase2:
        model = ConvolutionalAutoEncoder()
    else:
        model = SimpleCNN(num_classes=2)
        
    model.to(device)
    return model

if __name__ == "__main__":
    # Test shape
    print("Testing ConvolutionalAutoEncoder...")
    model = ConvolutionalAutoEncoder()
    dummy_input = torch.randn(2, 1, 256, 256)
    output = model(dummy_input)
    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == dummy_input.shape
    print("Shape mismatch test passed!")
