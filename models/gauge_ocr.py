import torch
import torch.nn as nn
from .semantic_branch import SemanticBranch
from .geometric_branch import GeometricBranch
from .adapter import Adapter
from .causal_encoder import CausalEncoder
from .decoder import LLMDecoder


class CausalGauge(nn.Module):
    """
    Complete Gauge OCR Model: CausalGauge.
    
    This is the main model that integrates all components for three-stage training:
    - Stage 1: Only geometric branch (keypoint detection)
    - Stage 2: Semantic + geometric fusion with Adapter (no causal queries)
    - Stage 3: Full model with causal flow reordering (with causal queries)
    
    Data Flow Overview:
    Input Image [B, 3, 448, 448]
    ├─→ Semantic Branch (SAM) → SAM Features [B, sam_dim, 28, 28]
    └─→ Geometric Branch (DeepLabV3+) → Geometric Features [B, 256, 112, 112]
    
    Adapter:
    ├─→ SAM Projection: [B, sam_dim, 28, 28] → [B, 784, visual_dim]
    ├─→ Gate Generator: [B, 256, 112, 112] → [B, 784]
    └─→ Cross-Attention: Fuses semantic + geometric → [B, 784, visual_dim]
    
    Causal Encoder:
    Fused Tokens [+ Causal Queries] → Encoded Tokens
    
    LLM Decoder:
    Encoded Tokens → Generated Text
    """
    
    def __init__(self, config):
        """
        Initialize the complete CausalGauge model.
        
        Args:
            config: Configuration dictionary containing model hyperparameters
        """
        super().__init__()
        
        self.config = config
        self.visual_dim = config['model']['visual_dim']
        self.num_queries = config['model']['num_queries']
        
        # Semantic Branch: SAM-based feature extraction
        # Extracts high-level semantic features from the image
        self.semantic_branch = SemanticBranch(
            sam_model_name=config['model']['sam_model_name'],
            input_size=448
        )
        
        # Geometric Branch: DeepLabV3+-based keypoint detection
        # Detects keypoints (pointer tip, center, scale marks)
        self.geometric_branch = GeometricBranch(
            backbone=config['model']['deeplab_backbone'],
            num_keypoints=config['model']['num_keypoints']
        )
        
        # Adapter: Combines SAM projection, gate generator, and cross-attention
        # This module fuses semantic and geometric features
        self.adapter = Adapter(
            sam_dim=self.semantic_branch.sam_dim,
            visual_dim=self.visual_dim,
            geometric_dim=256,
            num_heads=8
        )
        
        # Causal Encoder: Qwen2-based with spatial bias
        # Processes fused tokens with optional causal queries
        self.causal_encoder = CausalEncoder(
            qwen_model_name=config['model']['qwen_model_name'],
            visual_dim=self.visual_dim,
            num_queries=self.num_queries
        )
        
        # LLM Decoder: Generates text from visual tokens
        self.llm_decoder = LLMDecoder(
            qwen_model_name=config['model']['qwen_model_name'],
            visual_dim=self.visual_dim
        )
    
    def forward(self, images, text_inputs=None, labels=None, stage='inference', use_causal_queries=False):
        """
        Forward pass through the CausalGauge model.
        
        Args:
            images: Input images [B, 3, 448, 448]
            text_inputs: Text input IDs [B, text_len] (for training, teacher forcing)
            labels: Target labels [B, text_len] (for training, loss computation)
            stage: Training stage ('stage1', 'stage2', 'stage3', 'inference')
            use_causal_queries: Whether to use causal flow queries (Stage 3)
        
        Returns:
            outputs: Dictionary with different outputs depending on the stage:
                - Stage 1: {'heatmaps', 'geometric_features'}
                - Training (with text): {'loss', 'logits', 'gate_scores'}
                - Inference: {'visual_tokens', 'gate_scores'}
        """
        # Stage 1: Only geometric branch for keypoint detection
        if stage == 'stage1':
            # Get geometric features and heatmaps from geometric branch
            geometric_features, heatmaps = self.geometric_branch(images, return_heatmap=True)
            return {'heatmaps': heatmaps, 'geometric_features': geometric_features}
        
        # Stages 2, 3, or inference: Full model
        
        # Step 1: Extract SAM features
        # Shape: [B, sam_dim, 28, 28]
        sam_features = self.semantic_branch(images)
        
        # Step 2: Extract geometric features from DeepLabV3+
        # Shape: [B, 256, 112, 112]
        geometric_features = self.geometric_branch(images)
        
        # Step 3: Use Adapter to project SAM features and fuse with geometric features
        # Returns:
        #   - fused_tokens: [B, 784, visual_dim]
        #   - gate_scores: [B, 784]
        fused_tokens, gate_scores = self.adapter(sam_features, geometric_features)
        
        # Step 4: Encode with causal encoder
        # Shape: [B, 784, hidden_size] or [B, 1568, hidden_size]
        encoded_tokens = self.causal_encoder(fused_tokens, use_causal_queries=use_causal_queries)
        
        # Extract visual tokens (either all tokens or just causal queries)
        if use_causal_queries:
            # Take only the causal queries part
            # Shape: [B, num_queries, hidden_size]
            visual_tokens = encoded_tokens[:, -self.num_queries:, :]
        else:
            # Use all encoded tokens
            # Shape: [B, 784, hidden_size]
            visual_tokens = encoded_tokens
        
        # Training mode with text inputs
        if self.training and text_inputs is not None:
            # Pass through LLM decoder with teacher forcing
            outputs = self.llm_decoder(visual_tokens, text_inputs, labels)
            return {
                'loss': outputs.loss,
                'logits': outputs.logits,
                'gate_scores': gate_scores
            }
        else:
            # Inference mode or no text inputs
            return {
                'visual_tokens': visual_tokens,
                'gate_scores': gate_scores
            }
    
    def generate(self, images, max_length=50, use_causal_queries=True):
        """
        Generate reading text from images (inference mode).
        
        Args:
            images: Input images [B, 3, 448, 448]
            max_length: Maximum number of tokens to generate
            use_causal_queries: Whether to use causal flow queries
        
        Returns:
            readings: List of generated reading text strings
        """
        # Set model to evaluation mode
        self.eval()
        
        # Forward pass without gradient computation
        with torch.no_grad():
            outputs = self.forward(
                images,
                stage='inference',
                use_causal_queries=use_causal_queries
            )
            
            # Get visual tokens from forward pass
            visual_tokens = outputs['visual_tokens']
            
            # Generate text autoregressively
            readings = self.llm_decoder.generate(visual_tokens, max_length=max_length)
        
        return readings
