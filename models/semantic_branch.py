import torch
import torch.nn as nn
from segment_anything import sam_model_registry, SamPredictor


class SemanticBranch(nn.Module):
    """
    Semantic Branch: SAM-based feature extraction.
    
    This branch uses Meta's Segment Anything Model (SAM) to extract
    rich semantic features from the input image. The SAM features are
    then projected to match the visual dimension required by downstream modules.
    
    Data Flow:
    Input Image [B, 3, 448, 448]
    → SAM Image Encoder (frozen)
    → SAM Features [B, 768, 28, 28] (for sam-base)
    → 1x1 Conv Projection
    → Projected Features [B, visual_dim, 28, 28]
    → Flatten & Transpose
    → Semantic Tokens [B, 784, visual_dim] (784 = 28×28)
    """
    
    def __init__(self, sam_model_name="sam-base", visual_dim=896):
        """
        Initialize the semantic branch.
        
        Args:
            sam_model_name: SAM model name ('sam-base', 'sam-large', 'sam-huge')
            visual_dim: Output feature dimension for downstream modules
        """
        super().__init__()
        
        # Mapping from model name to checkpoint file
        sam_checkpoint_map = {
            'sam-base': 'sam_vit_b_01ec64.pth',
            'sam-large': 'sam_vit_l_0b3195.pth',
            'sam-huge': 'sam_vit_h_4b8939.pth'
        }
        
        # Load pre-trained SAM model
        self.sam = sam_model_registry[sam_model_name](checkpoint=sam_checkpoint_map[sam_model_name])
        
        # Determine SAM's output dimension based on model size
        self.sam_dim = 768 if sam_model_name == 'sam-base' else (1024 if sam_model_name == 'sam-large' else 1280)
        
        # 1x1 convolution to project SAM features to the desired visual dimension
        # Shape change: [B, sam_dim, 28, 28] → [B, visual_dim, 28, 28]
        self.projection = nn.Conv2d(self.sam_dim, visual_dim, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        """
        Forward pass through the semantic branch.
        
        Args:
            x: Input image tensor [B, 3, 448, 448]
        
        Returns:
            semantic_tokens: Semantic tokens [B, 784, visual_dim]
                            where 784 = 28×28 (spatial grid)
        """
        # Extract SAM features without gradient computation
        # SAM is kept frozen during training
        with torch.no_grad():
            # SAM image encoder output shape: [B, sam_dim, 28, 28]
            features = self.sam.image_encoder(x)
        
        # Project SAM features to the desired visual dimension
        # Shape change: [B, sam_dim, 28, 28] → [B, visual_dim, 28, 28]
        features = self.projection(features)
        
        # Reshape to sequence format for transformer processing
        # Shape changes:
        # [B, visual_dim, 28, 28] → [B, visual_dim, 784] (flatten spatial dimensions)
        # → [B, 784, visual_dim] (transpose to put sequence dimension first)
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)
        
        return features
