import torch
import torch.nn as nn
import torch.nn.functional as F
from .deeplabv3plus import deeplabv3plus_resnet50, load_pretrained_weights


class GeometricBranch(nn.Module):
    """
    Geometric Branch: DeepLabV3+ (ResNet50) based geometric feature extraction.
    
    This branch extracts geometric features using DeepLabV3+ architecture
    with ResNet50 backbone, and optionally predicts keypoint heatmaps
    for Stage 1 training.
    
    Key Features:
    - Uses ResNet50 backbone with DeepLabV3+ decoder
    - Loads pre-trained weights from checkpoints folder
    - Supports easy head replacement via replace_classification_head()
    - Can extract [B, 256, 112, 112] features for the geometric branch
    
    Data Flow (Stage 1):
    Input Image [B, 3, 448, 448]
    → DeepLabV3+ (ResNet50) Backbone
      → Low-level features (layer1) [B, 256, 112, 112]
      → High-level features (layer4) [B, 2048, 28, 28]
    → ASPP
    → ASPP Output [B, 256, 28, 28]
    → Decoder (fuses with low-level features)
    → Decoder Output [B, 256, 112, 112] ← This is the geometric feature output
    → (Optional) Keypoint Head (1x1 Conv)
    → Heatmaps [B, num_keypoints, 112, 112]
    """
    
    def __init__(self, backbone="resnet50", num_keypoints=2, pretrained_path=None):
        """
        Initialize the geometric branch.
        
        Args:
            backbone: Backbone network (only 'resnet50' supported)
            num_keypoints: Number of keypoints to detect (for Stage 1)
            pretrained_path: Path to pre-trained DeepLabV3+ weights (optional)
                           If provided, will load weights and replace head for num_keypoints
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.num_keypoints = num_keypoints
        
        if backbone == "resnet50":
            # Create DeepLabV3+ with ResNet50 backbone
            # Initially create with 21 classes (VOC), we'll replace the head if needed
            self.deeplab = deeplabv3plus_resnet50(num_classes=21, output_stride=16)
            
            # Load pre-trained weights if provided
            if pretrained_path is not None:
                print(f"Loading pre-trained weights from: {pretrained_path}")
                self.deeplab = load_pretrained_weights(self.deeplab, pretrained_path)
                
                # Replace classification head for our keypoint detection task
                # num_keypoints is the number of heatmap channels we need
                self.deeplab.replace_classification_head(num_classes=num_keypoints)
                print(f"Replaced classification head for {num_keypoints} keypoints")
            else:
                # If no pretrained weights, just create with correct number of classes
                self.deeplab.replace_classification_head(num_classes=num_keypoints)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Only 'resnet50' is supported.")
    
    def get_geometric_features(self, x):
        """
        Extract geometric features (without keypoint head).
        
        This returns the [B, 256, 112, 112] feature map from the
        DeepLabV3+ decoder (before the final classification layer).
        
        Args:
            x: Input image [B, 3, 448, 448]
        
        Returns:
            decoder_out: Geometric features [B, 256, 112, 112]
        """
        # Use extract_features method from DeepLabV3+
        # This returns [B, 256, H/4, W/4] = [B, 256, 112, 112] for 448x448 input
        geometric_features = self.deeplab.extract_features(x)
        
        return geometric_features
    
    def forward(self, x, return_heatmap=False):
        """
        Forward pass through the geometric branch.
        
        Args:
            x: Input image [B, 3, 448, 448]
            return_heatmap: Whether to return keypoint heatmaps (for Stage 1)
        
        Returns:
            If return_heatmap=True:
                (decoder_out, heatmaps): Tuple of features and heatmaps
                - decoder_out: [B, 256, 112, 112]
                - heatmaps: [B, num_keypoints, 448, 448] (upsampled to input size)
            Otherwise:
                decoder_out: Geometric features only [B, 256, 112, 112]
        """
        if return_heatmap:
            # Get both logits (heatmaps) and geometric features
            heatmaps, decoder_out = self.deeplab(x, return_features=True)
            # heatmaps: [B, num_keypoints, 448, 448] (already upsampled)
            # decoder_out: [B, 256, 112, 112]
            return decoder_out, heatmaps
        else:
            # Just get geometric features
            decoder_out = self.get_geometric_features(x)
            return decoder_out
