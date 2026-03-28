import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.
    
    ASPP applies atrous convolution at multiple rates to capture
    multi-scale contextual information. This is a key component of DeepLabV3+.
    
    Data Flow:
    Input Features [B, in_channels, H, W]
    → Parallel branches:
      - 1x1 Conv
      - 3x3 Conv (rate=6)
      - 3x3 Conv (rate=12)
      - 3x3 Conv (rate=18)
      - Image Pooling + 1x1 Conv + Upsample
    → Concatenate all branches
    → 1x1 Conv projection + Dropout
    → Output [B, out_channels, H, W]
    """
    
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        """
        Initialize the ASPP module.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            atrous_rates: List of atrous (dilation) rates for 3x3 convolutions
        """
        super().__init__()
        
        # 1x1 convolution branch
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolutions with different atrous rates
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[0], dilation=atrous_rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[1], dilation=atrous_rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3x3_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[2], dilation=atrous_rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Image pooling branch: captures global context
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling to [B, C, 1, 1]
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final projection: concatenate all 5 branches and project to out_channels
        # Input channels: out_channels * 5
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        """
        Forward pass through ASPP.
        
        Args:
            x: Input features [B, in_channels, H, W]
        
        Returns:
            out: ASPP output features [B, out_channels, H, W]
        """
        B, C, H, W = x.shape
        
        # Apply all parallel branches
        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3_1(x)
        feat3 = self.conv3x3_2(x)
        feat4 = self.conv3x3_3(x)
        
        # Apply image pooling and upsample back to original size
        feat5 = F.interpolate(self.pool(x), size=(H, W), mode='bilinear', align_corners=False)
        
        # Concatenate all branches along channel dimension
        # Shape: [B, out_channels * 5, H, W]
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        
        # Project to final output dimension
        out = self.project(out)
        
        return out


class Decoder(nn.Module):
    """
    DeepLabV3+ Decoder module.
    
    Fuses high-level semantic features (from ASPP) with low-level
    spatial features (from early backbone layers) for better localization.
    
    Data Flow:
    Low-level Features [B, low_level_channels, H_low, W_low]
    → 1x1 Conv projection
    High-level Features [B, out_channels, H_high, W_high]
    → Upsample to [B, out_channels, H_low, W_low]
    → Concatenate with low-level features
    → Two 3x3 Convs
    → Output [B, out_channels, H_low, W_low]
    """
    
    def __init__(self, low_level_channels=24, out_channels=256):
        """
        Initialize the decoder.
        
        Args:
            low_level_channels: Number of channels in low-level features
            out_channels: Number of output channels
        """
        super().__init__()
        
        # Project low-level features to reduce channel dimension
        # Shape change: [B, low_level_channels, H, W] → [B, 48, H, W]
        self.low_level_project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder convolutions: fuse high-level and low-level features
        # Input channels: out_channels (high-level) + 48 (low-level)
        self.decoder = nn.Sequential(
            nn.Conv2d(out_channels + 48, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, low_level_feat):
        """
        Forward pass through decoder.
        
        Args:
            x: High-level features from ASPP [B, out_channels, H_high, W_high]
            low_level_feat: Low-level features from backbone [B, low_level_channels, H_low, W_low]
        
        Returns:
            out: Decoder output features [B, out_channels, H_low, W_low]
        """
        # Project low-level features
        low_level_feat = self.low_level_project(low_level_feat)
        
        # Upsample high-level features to match low-level feature size
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate along channel dimension
        # Shape: [B, out_channels + 48, H_low, W_low]
        x = torch.cat([x, low_level_feat], dim=1)
        
        # Apply decoder convolutions
        x = self.decoder(x)
        
        return x


class GeometricBranch(nn.Module):
    """
    Geometric Branch: DeepLabV3+ based geometric feature extraction.
    
    This branch extracts geometric features using DeepLabV3+ architecture,
    and optionally predicts keypoint heatmaps for Stage 1 training.
    
    Data Flow (Stage 1):
    Input Image [B, 3, 448, 448]
    → MobileNetV2 Backbone
      → Low-level features [B, 24, 112, 112]
      → High-level features [B, 160, 28, 28]
    → ASPP
    → ASPP Output [B, 256, 28, 28]
    → Decoder (fuses with low-level features)
    → Decoder Output [B, 256, 112, 112]
    → (Optional) Keypoint Head (1x1 Conv)
    → Heatmaps [B, num_keypoints, 112, 112]
    """
    
    def __init__(self, backbone="mobilenet_v2", num_keypoints=6):
        """
        Initialize the geometric branch.
        
        Args:
            backbone: Backbone network ('mobilenet_v2', 'resnet50')
            num_keypoints: Number of keypoints to detect (for Stage 1)
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.num_keypoints = num_keypoints
        
        # Load pre-trained DeepLabV3+ with MobileNetV2 backbone
        if backbone == "mobilenet_v2":
            model = deeplabv3_mobilenet_v3_large(pretrained=True)
            self.backbone = model.backbone
            self.low_level_channels = 24  # Channel dimension of low-level features
            self.high_level_channels = 160  # Channel dimension of high-level features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # ASPP module for multi-scale context
        self.aspp = ASPP(self.high_level_channels, out_channels=256)
        
        # Decoder module for feature fusion
        self.decoder = Decoder(low_level_channels=self.low_level_channels, out_channels=256)
        
        # Keypoint head: predicts heatmaps for each keypoint (used in Stage 1)
        # 1x1 convolution to map decoder features to keypoint heatmaps
        self.keypoint_head = nn.Conv2d(256, num_keypoints, kernel_size=1)
    
    def get_geometric_features(self, x):
        """
        Extract geometric features (without keypoint head).
        
        Args:
            x: Input image [B, 3, 448, 448]
        
        Returns:
            decoder_out: Geometric features [B, 256, 112, 112]
        """
        # Extract backbone features
        # The backbone returns a dictionary with 'low_level' and 'out' features
        features = self.backbone(x)
        
        low_level_feat = features['low_level']  # [B, 24, 112, 112]
        high_level_feat = features['out']       # [B, 160, 28, 28]
        
        # Apply ASPP to high-level features
        aspp_out = self.aspp(high_level_feat)   # [B, 256, 28, 28]
        
        # Apply decoder to fuse high-level and low-level features
        decoder_out = self.decoder(aspp_out, low_level_feat)  # [B, 256, 112, 112]
        
        return decoder_out
    
    def forward(self, x, return_heatmap=False):
        """
        Forward pass through the geometric branch.
        
        Args:
            x: Input image [B, 3, 448, 448]
            return_heatmap: Whether to return keypoint heatmaps (for Stage 1)
        
        Returns:
            If return_heatmap=True:
                (decoder_out, heatmaps): Tuple of features and heatmaps
            Otherwise:
                decoder_out: Geometric features only
        """
        # Get geometric features
        decoder_out = self.get_geometric_features(x)
        
        # Optionally predict keypoint heatmaps
        if return_heatmap:
            # Apply keypoint head: [B, 256, 112, 112] → [B, num_keypoints, 112, 112]
            heatmaps = self.keypoint_head(decoder_out)
            return decoder_out, heatmaps
        
        return decoder_out
