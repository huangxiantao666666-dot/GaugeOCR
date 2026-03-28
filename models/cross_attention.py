import torch
import torch.nn as nn


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
