import torch
import torch.nn as nn


class GateGenerator(nn.Module):
    """
    Gate Generator: Generates gate scores based on geometric features.
    
    The gate scores control the weight of semantic tokens during cross-attention.
    Regions with stronger geometric features (e.g., pointer, scale marks) will have
    higher gate scores, giving them more influence in the cross-attention module.
    
    Data Flow:
    Geometric Features [B, 256, 112, 112]
    → AvgPool2d (4×4) → [B, 256, 28, 28]
    → Conv2d (1×1) + Sigmoid → [B, 1, 28, 28] (gate map)
    → Flatten → [B, 784] (gate scores)
    
    The gate scores are in [0, 1] range due to the sigmoid activation.
    """
    
    def __init__(self, in_channels=256):
        """
        Initialize the gate generator.
        
        Args:
            in_channels: Number of input channels in geometric features
        """
        super().__init__()
        
        # Average pooling to downsample geometric features
        # 4×4 kernel with stride 4 reduces 112×112 to 28×28
        # Shape change: [B, 256, 112, 112] → [B, 256, 28, 28]
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        
        # Convolution to produce gate map
        # 1×1 conv reduces channels from 256 to 1
        # Sigmoid ensures gate values are in [0, 1]
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, geometric_features):
        """
        Forward pass through the gate generator.
        
        Args:
            geometric_features: Geometric features from DeepLabV3+
                               [B, 256, 112, 112]
        
        Returns:
            gate_scores: Gate scores for each spatial location
                        [B, 784] (784 = 28×28)
        """
        # Downsample geometric features
        # Shape change: [B, 256, 112, 112] → [B, 256, 28, 28]
        x = self.pool(geometric_features)
        
        # Generate gate map
        # Shape change: [B, 256, 28, 28] → [B, 1, 28, 28]
        gate_map = self.conv(x)
        
        # Flatten gate map to get gate scores
        B, C, H, W = gate_map.shape
        # Shape change: [B, 1, 28, 28] → [B, 784]
        gate_scores = gate_map.flatten(1)
        
        return gate_scores
