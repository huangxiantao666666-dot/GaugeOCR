import torch
import torch.nn as nn
from transformers import Qwen2Model


class CausalEncoder(nn.Module):
    """
    Causal Flow Encoder: Qwen2-based Transformer layer with spatial bias and causal mask.
    
    This encoder processes visual tokens with:
    1. Optional projection to match Qwen2's hidden dimension
    2. Optional causal queries for reordering (Stage 3)
    3. Causal attention mask for autoregressive behavior
    4. Spatial bias based on token positions
    
    Data Flow (without causal queries):
    Visual Tokens [B, 784, dim]
    → Projection (if needed) → [B, 784, hidden_size]
    → Qwen2 Model
    → Encoded Tokens [B, 784, hidden_size]
    
    Data Flow (with causal queries):
    Visual Tokens [B, 784, dim]
    → Projection (if needed) → [B, 784, hidden_size]
    → Concatenate with causal queries → [B, 1568, hidden_size]
    → Qwen2 Model (with causal mask)
    → Encoded Tokens [B, 1568, hidden_size]
    """
    
    def __init__(self, qwen_model_name="Qwen/Qwen2-0.5B", dim=896, num_queries=784):
        """
        Initialize the causal encoder.
        
        Args:
            qwen_model_name: Name of the pre-trained Qwen2 model
            dim: Input visual feature dimension
            num_queries: Number of causal queries (and spatial grid size)
        """
        super().__init__()
        
        self.dim = dim
        self.num_queries = num_queries
        
        # Load pre-trained Qwen2 model
        self.qwen = Qwen2Model.from_pretrained(qwen_model_name)
        
        # Get Qwen2's hidden dimension
        self.hidden_size = self.qwen.config.hidden_size
        
        # Linear projection if input dimension doesn't match Qwen2's hidden size
        if dim != self.hidden_size:
            self.projection = nn.Linear(dim, self.hidden_size)
        else:
            self.projection = None
        
        # Learnable causal queries for Stage 3
        # Shape: [1, num_queries, hidden_size] - will be broadcasted to batch size
        self.causal_queries = nn.Parameter(torch.randn(1, num_queries, self.hidden_size))
        
        # Initialize spatial bias coordinates
        self._init_spatial_bias(num_queries)
    
    def _init_spatial_bias(self, num_queries):
        """
        Initialize spatial coordinates for computing spatial bias.
        
        Creates a 2D grid of coordinates for the spatial tokens.
        Assumes num_queries is a perfect square (e.g., 784 = 28×28).
        
        Data Flow:
        num_queries (784)
        → size = sqrt(num_queries) = 28
        → Create meshgrid of coordinates [size, size, 2]
        → Flatten to [num_queries, 2]
        → Register as buffer (saved with model but not trained)
        
        Args:
            num_queries: Number of queries (should be perfect square)
        """
        # Compute grid size (assuming square grid)
        size = int(num_queries ** 0.5)
        
        # Create 2D coordinate grid
        # coords shape: [size, size, 2] - (y, x) for each position
        coords = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size)), dim=-1)
        
        # Flatten to [num_queries, 2]
        coords = coords.view(-1, 2).float()
        
        # Register as buffer (persists in model state but not trained)
        self.register_buffer('coords', coords)
    
    def compute_spatial_bias(self):
        """
        Compute spatial bias matrix using Gaussian kernel.
        
        The spatial bias encourages nearby tokens to attend to each other more.
        
        Data Flow:
        Coords [num_queries, 2]
        → Compute pairwise Euclidean distances [num_queries, num_queries]
        → Apply Gaussian kernel: exp(-dist² / (2σ²))
        → Spatial bias matrix [num_queries, num_queries]
        
        Returns:
            bias: Spatial bias matrix [num_queries, num_queries]
        """
        coords = self.coords
        
        # Compute pairwise Euclidean distances between all coordinates
        # Shape: [num_queries, num_queries]
        dist = torch.cdist(coords, coords)
        
        # Apply Gaussian kernel to convert distances to bias weights
        sigma = 10.0
        bias = torch.exp(-dist ** 2 / (2 * sigma ** 2))
        
        return bias
    
    def build_causal_mask(self, seq_len):
        """
        Build causal (autoregressive) attention mask.
        
        Prevents tokens from attending to future tokens.
        
        Data Flow:
        seq_len (int)
        → Create upper triangular matrix of ones [seq_len, seq_len]
        → Mask upper triangle with -inf
        → Causal mask [seq_len, seq_len]
        
        Args:
            seq_len: Length of the sequence
        
        Returns:
            mask: Causal attention mask [seq_len, seq_len]
                  with 0s for allowed positions and -inf for masked positions
        """
        # Create upper triangular matrix (diagonal=1 means exclude diagonal)
        # Shape: [seq_len, seq_len]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        
        # Replace 1s with -inf (these positions will be masked in softmax)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        
        return mask
    
    def forward(self, visual_tokens, use_causal_queries=False):
        """
        Forward pass through the causal encoder.
        
        Args:
            visual_tokens: Visual tokens [B, 784, dim]
            use_causal_queries: Whether to use causal flow queries (Stage 3)
        
        Returns:
            encoded_tokens: Encoded tokens from Qwen2
                          [B, 784, hidden_size] or [B, 1568, hidden_size]
        """
        # Project visual tokens to Qwen2's hidden dimension if needed
        # Shape change: [B, 784, dim] → [B, 784, hidden_size]
        if self.projection is not None:
            visual_tokens = self.projection(visual_tokens)
        
        # Concatenate causal queries if using them (Stage 3)
        if use_causal_queries:
            B = visual_tokens.shape[0]
            # Repeat causal queries for batch dimension
            # Shape change: [1, num_queries, hidden_size] → [B, num_queries, hidden_size]
            causal_queries = self.causal_queries.repeat(B, 1, 1)
            # Concatenate visual tokens and causal queries
            # Shape change: [B, 784, hidden_size] + [B, 784, hidden_size] → [B, 1568, hidden_size]
            input_tokens = torch.cat([visual_tokens, causal_queries], dim=1)
        else:
            input_tokens = visual_tokens
        
        # Build causal mask for the input sequence
        seq_len = input_tokens.shape[1]
        causal_mask = self.build_causal_mask(seq_len).to(input_tokens.device)
        
        # Compute spatial bias
        spatial_bias = self.compute_spatial_bias()
        
        # Pad spatial bias if using causal queries to match sequence length
        if use_causal_queries:
            # Pad bottom: [num_queries, num_queries] → [seq_len, num_queries]
            padding = torch.zeros(seq_len - self.num_queries, seq_len).to(input_tokens.device)
            spatial_bias = torch.cat([spatial_bias, padding], dim=0)
            # Pad right: [seq_len, num_queries] → [seq_len, seq_len]
            padding = torch.zeros(seq_len, seq_len - self.num_queries).to(input_tokens.device)
            spatial_bias = torch.cat([spatial_bias, padding], dim=1)
        
        # Pass through Qwen2 model
        # Note: In this simplified version, spatial bias isn't directly added to attention scores
        # A full implementation would modify Qwen2's attention layers
        outputs = self.qwen(
            inputs_embeds=input_tokens,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=True
        )
        
        # Get the last hidden state
        encoded_tokens = outputs.last_hidden_state
        
        return encoded_tokens
