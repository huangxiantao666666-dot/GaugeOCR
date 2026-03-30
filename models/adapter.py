"""
Adapter Module for GaugeOCR
===========================
This module contains the adapter components that connect the vision branches
(SAM and DeepLabV3+) to the Qwen2-based encoder.

Overall Architecture Data Flow:
1. Input Image [B, 3, 448, 448]
   → Semantic Branch (SAM) → Visual Tokens [B, 784, visual_dim]
   → Geometric Branch (DeepLabV3+) → Geometric Features [B, 256, 112, 112]

2. Adapter (this module):
   → SAM Projection: [B, sam_dim, 28, 28] → [B, 784, visual_dim]
   → Gate Generator: [B, 256, 112, 112] → [B, 784] (gate scores)
   → Cross-Attention: Fuses visual tokens with geometric features
   → Output: Fused Tokens [B, 784, visual_dim]

3. Qwen2 Encoder (CausalEncoder):
   → Stage 2 (without causal queries): [B, 784, visual_dim] → [B, 784, hidden_size]
   → Stage 3 (with causal queries): [B, 784, visual_dim] + [B, 392, hidden_size] → [B, 392, hidden_size]

4. Qwen2 Decoder:
   → Generates text autoregressively

Adapter Components:
1. SAM Projection Layer: Projects SAM features (768/1024/1280) to Qwen2's visual dimension (896)
2. Gate Generator: Generates gate scores from geometric features to weight semantic tokens
3. Cross-Attention Module: Fuses semantic tokens (query) with geometric features (key/value)

Training Strategy:
- Stage 1: Train DeepLabV3+ only (keypoint detection)
  - SAM: Frozen (not used)
  - Adapter: Not used
  - DeepLabV3+: Trainable
  
- Stage 2: Train adapter only
  - SAM: Frozen (requires_grad=False)
  - DeepLabV3+: Frozen (requires_grad=False, classification head removed)
  - Adapter: Trainable (requires_grad=True)
  - Causal Encoder: Frozen (use_causal_queries=False)
  
- Stage 3: Train causal encoder and LLM decoder
  - SAM: Frozen
  - DeepLabV3+: Frozen
  - Adapter: Frozen
  - Causal Encoder: Trainable (use_causal_queries=True)
  - LLM Decoder: Trainable
"""

import torch
import torch.nn as nn


class SAMProjection(nn.Module):
    """
    SAM Feature Projection Layer.
    
    Projects SAM encoder features to the visual dimension required by Qwen2.
    This is a learnable 1x1 convolution that adapts SAM features for downstream tasks.
    
    Data Flow:
    SAM Features [B, sam_dim, 28, 28]
    → 1x1 Conv Projection
    → Projected Features [B, visual_dim, 28, 28]
    → Flatten & Transpose
    → Semantic Tokens [B, 784, visual_dim]
    
    Note:
    - SAM dimension varies by model:
      - sam-base: 768
      - sam-large: 1024
      - sam-huge: 1280
    - Qwen2 visual dimension: 896
    """
    
    def __init__(self, sam_dim=768, visual_dim=896):
        """
        Initialize the SAM projection layer.
        
        Args:
            sam_dim: SAM encoder output dimension
            visual_dim: Target dimension for Qwen2 (default: 896)
        """
        super().__init__()
        
        self.sam_dim = sam_dim
        self.visual_dim = visual_dim
        
        # 1x1 convolution to project SAM features to visual dimension
        # Shape change: [B, sam_dim, 28, 28] → [B, visual_dim, 28, 28]
        self.projection = nn.Conv2d(sam_dim, visual_dim, kernel_size=1, stride=1, padding=0)
    
    def forward(self, sam_features):
        """
        Forward pass through the projection layer.
        
        Args:
            sam_features: SAM encoder output [B, sam_dim, 28, 28]
        
        Returns:
            semantic_tokens: Projected features [B, 784, visual_dim]
        """
        B = sam_features.shape[0]
        
        # Project SAM features to visual dimension
        # Shape: [B, sam_dim, 28, 28] → [B, visual_dim, 28, 28]
        features = self.projection(sam_features)
        
        # Flatten and transpose to sequence format
        # Shape: [B, visual_dim, 28, 28] → [B, 784, visual_dim]
        features = features.flatten(2).transpose(1, 2)
        
        return features


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


class CrossAttentionModule(nn.Module):
    """
    Cross-Attention Module: Fuses semantic and geometric features.
    
    This module performs cross-attention where:
    - Query: Semantic tokens (weighted by gate scores)
    - Key/Value: Geometric tokens (projected from geometric features)
    
    The module also includes a residual connection and layer normalization.
    
    Data Flow:
    Geometric Features [B, 256, 112, 112]
    → Flatten → [B, 256, 12544]
    → Reshape & Permute → [B, 28, 28, 4, 4, 256]
    → Reshape → [B, 784, 4096] (784 patches × 16×256)
    → Mean pooling → [B, 784, 256]
    → Linear projection → [B, 784, dim] (geometric tokens)
    
    Semantic Tokens [B, 784, dim]
    → Gate weighting (if provided) → [B, 784, dim]
    
    Cross-Attention:
    Query = Weighted semantic tokens
    Key/Value = Geometric tokens
    → Multi-head attention
    → Residual connection: query + attention_output
    → LayerNorm
    → Fused Tokens [B, 784, dim]
    """
    
    def __init__(self, dim=896, geometric_dim=256, num_heads=8):
        """
        Initialize the cross-attention module.
        
        Args:
            dim: Feature dimension (should match Qwen2's visual dimension)
            geometric_dim: Input geometric feature dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # Linear projection for geometric features
        # Projects from geometric_dim to dim
        self.geometric_projection = nn.Linear(geometric_dim, dim)
        
        # Multi-head attention layer
        # batch_first=True means input shape is (batch, seq_len, feature)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer normalization for the residual connection
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, semantic_tokens, geometric_features, gate_scores=None):
        """
        Forward pass through the cross-attention module.
        
        Args:
            semantic_tokens: Semantic tokens from SAM [B, 784, dim]
            geometric_features: Geometric features from DeepLabV3+ [B, 256, 112, 112]
            gate_scores: Gate scores from gate generator [B, 784] (optional)
        
        Returns:
            fused_tokens: Fused tokens after cross-attention [B, 784, dim]
        """
        B, C, H, W = geometric_features.shape
        
        # Process geometric features to get geometric tokens
        # First, flatten spatial dimensions
        # Shape change: [B, 256, 112, 112] → [B, 256, 12544] → [B, 12544, 256]
        geo_tokens = geometric_features.flatten(2).transpose(1, 2)
        
        # Reshape to group 4x4 patches together
        # This is one way to downsample geometric features to match semantic tokens
        # Shape changes:
        # [B, 12544, 256] → [B, 4, 4, 28, 28, 256]
        # → [B, 28, 28, 4, 4, 256] (permute to group spatial locations)
        # → [B, 784, 4096] (flatten: 28×28=784, 4×4×256=4096)
        geo_tokens = geo_tokens.view(B, 4, 4, H // 4, W // 4, C)
        geo_tokens = geo_tokens.permute(0, 3, 4, 1, 2, 5).contiguous()
        geo_tokens = geo_tokens.view(B, (H // 4) * (W // 4), 16 * C)
        
        # Mean pool over the 16 elements per patch
        # Shape change: [B, 784, 4096] → [B, 784, 256]
        geo_tokens = geo_tokens.mean(dim=2, keepdim=False)
        
        # Project geometric features to match semantic dimension
        # Shape change: [B, 784, 256] → [B, 784, dim]
        geo_tokens = self.geometric_projection(geo_tokens)
        
        # Apply gate scores to semantic tokens if provided
        if gate_scores is not None:
            # Add channel dimension for broadcasting
            # Shape change: [B, 784] → [B, 784, 1]
            gate_scores = gate_scores.unsqueeze(-1)
            # Element-wise multiplication: weight semantic tokens by gate scores
            semantic_tokens = semantic_tokens * gate_scores
        
        # Perform cross-attention
        # Query: semantic tokens (what we want to update)
        # Key/Value: geometric tokens (what we attend to)
        attn_output, _ = self.multihead_attn(
            query=semantic_tokens,
            key=geo_tokens,
            value=geo_tokens
        )
        
        # Residual connection + layer normalization
        # Shape: [B, 784, dim]
        fused_tokens = self.norm(semantic_tokens + attn_output)
        
        return fused_tokens


class Adapter(nn.Module):
    """
    Complete Adapter Module for GaugeOCR.
    
    This module combines all adapter components:
    1. SAM Projection: Projects SAM features to visual dimension
    2. Gate Generator: Generates gate scores from geometric features
    3. Cross-Attention: Fuses semantic and geometric features
    
    Data Flow:
    SAM Features [B, sam_dim, 28, 28]
    → SAM Projection → [B, 784, visual_dim] (semantic tokens)
    
    Geometric Features [B, 256, 112, 112]
    → Gate Generator → [B, 784] (gate scores)
    → Cross-Attention (with semantic tokens) → [B, 784, visual_dim] (fused tokens)
    
    Training Strategy (Stage 2):
    - SAM encoder: Frozen (requires_grad=False)
    - DeepLabV3+: Frozen (requires_grad=False)
    - Adapter: Trainable (requires_grad=True)
    
    Usage Example:
        # Initialize adapter
        adapter = Adapter(sam_dim=768, visual_dim=896)
        
        # Load SAM and DeepLabV3+ (frozen)
        sam = sam_model_registry['sam-base'](checkpoint='sam_vit_b.pth')
        deeplab = deeplabv3plus_resnet50(num_classes=6)
        
        # Stage 2 training: only train adapter
        for param in sam.parameters():
            param.requires_grad = False
        for param in deeplab.parameters():
            param.requires_grad = False
        for param in adapter.parameters():
            param.requires_grad = True
    """
    
    def __init__(self, sam_dim=768, visual_dim=896, geometric_dim=256, num_heads=8):
        """
        Initialize the complete adapter module.
        
        Args:
            sam_dim: SAM encoder output dimension (default: 768 for sam-base)
            visual_dim: Target dimension for Qwen2 (default: 896)
            geometric_dim: Geometric feature dimension (default: 256)
            num_heads: Number of attention heads (default: 8)
        """
        super().__init__()
        
        self.sam_dim = sam_dim
        self.visual_dim = visual_dim
        self.geometric_dim = geometric_dim
        
        # SAM projection layer
        self.sam_projection = SAMProjection(sam_dim=sam_dim, visual_dim=visual_dim)
        
        # Gate generator
        self.gate_generator = GateGenerator(in_channels=geometric_dim)
        
        # Cross-attention module
        self.cross_attention = CrossAttentionModule(
            dim=visual_dim,
            geometric_dim=geometric_dim,
            num_heads=num_heads
        )
    
    def forward(self, sam_features, geometric_features):
        """
        Forward pass through the adapter.
        
        Args:
            sam_features: SAM encoder output [B, sam_dim, 28, 28]
            geometric_features: DeepLabV3+ features [B, 256, 112, 112]
        
        Returns:
            fused_tokens: Fused tokens [B, 784, visual_dim]
            gate_scores: Gate scores [B, 784] (for visualization/debugging)
        """
        # Project SAM features to visual dimension
        # Shape: [B, sam_dim, 28, 28] → [B, 784, visual_dim]
        semantic_tokens = self.sam_projection(sam_features)
        
        # Generate gate scores from geometric features
        # Shape: [B, 256, 112, 112] → [B, 784]
        gate_scores = self.gate_generator(geometric_features)
        
        # Fuse semantic and geometric features via cross-attention
        # Shape: [B, 784, visual_dim]
        fused_tokens = self.cross_attention(
            semantic_tokens=semantic_tokens,
            geometric_features=geometric_features,
            gate_scores=gate_scores
        )
        
        return fused_tokens, gate_scores
