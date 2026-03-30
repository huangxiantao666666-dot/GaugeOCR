import torch
import torch.nn as nn
from segment_anything import sam_model_registry


class SemanticBranch(nn.Module):
    """
    Semantic Branch: SAM-based feature extraction.
    
    This branch uses Meta's Segment Anything Model (SAM) to extract
    rich semantic features from the input image.
    
    Important:
    - This branch only contains the SAM image encoder (frozen)
    - The projection layer is in the Adapter module, not here
    - This separation allows easy parameter freezing in Stage 2 training
    
    Data Flow:
    Input Image [B, 3, 448, 448]
    → SAM Image Encoder (frozen)
    → SAM Features [B, sam_dim, 28, 28] (sam_dim: 768/1024/1280)
    
    Note: The SAM features are then projected to visual_dim (896) by the
    Adapter module's SAMProjection layer, not by this branch.
    """
    
    def __init__(self, sam_model_name="sam-base", input_size=448, 
                 train_sam=False, sam_checkpoint_path=None):
        """
        Initialize the semantic branch.
        
        Args:
            sam_model_name: SAM model name ('sam-base', 'sam-large', 'sam-huge')
            input_size: Input image size (for padding)
            train_sam: Whether to train the SAM model (default: False)
            sam_checkpoint_path: Path to the SAM model checkpoint (default: None, uses default checkpoints)
        """
        super().__init__()
        
        # Mapping from model name to checkpoint file
        sam_checkpoint_map = {
            'sam-base': 'sam_vit_b_01ec64.pth',
            'sam-large': 'sam_vit_l_0b3195.pth',
            'sam-huge': 'sam_vit_h_4b8939.pth'
        }
        
        # Load pre-trained SAM model
        self.sam = sam_model_registry[sam_model_name](checkpoint=sam_checkpoint_path)

        self.sam.image_encoder.set_image_size(input_size)
        
        # Determine SAM's output dimension based on model size
        self.sam_dim = 768 if sam_model_name == 'sam-base' else (1024 if sam_model_name == 'sam-large' else 1280)
        
        # Freeze SAM parameters by default
        for param in self.sam.parameters():
            param.requires_grad = train_sam
        if not train_sam:
            self.sam.eval()
    
    def forward(self, x):
        """
        Forward pass through the semantic branch.
        
        Args:
            x: Input image tensor [B, 3, 448, 448]
        
        Returns:
            sam_features: SAM encoder output [B, sam_dim, 28, 28]
                         Note: This is NOT projected to visual_dim yet
        """
        # Extract SAM features without gradient computation
        # SAM is kept frozen during training
        with torch.no_grad():
            # SAM image encoder output shape: [B, sam_dim, 28, 28]
            features = self.sam.image_encoder(x)
        
        # Return raw SAM features (no projection)
        # The projection to visual_dim is done by the Adapter module
        return features
