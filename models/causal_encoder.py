import torch
import torch.nn as nn
from transformers import Qwen2Model


import torch
import torch.nn as nn
from transformers import Qwen2Model
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


class CausalEncoder(nn.Module):
    """
    Causal Flow Encoder with custom attention mask.
    
    Attention mask design:
    - Visual tokens: bidirectional attention among themselves
    - Visual tokens → Causal queries: queries can see all visual tokens
    - Causal queries → Visual tokens: visual tokens can see all queries
    - Causal queries among themselves: causal attention (autoregressive)
    
    Data Flow (Stage 2 - without causal queries):
    Visual Tokens [B, 784, dim] → Qwen2 (default causal mask) → [B, 784, hidden_size]
    
    Data Flow (Stage 3 - with causal queries):
    Visual Tokens [B, 784, dim] + Queries [B, 392, hidden_size]
    → Concatenate → [B, 1176, hidden_size]
    → Qwen2 with custom mask → [B, 1176, hidden_size]
    → Return only query part [B, 392, hidden_size]
    """
    
    def __init__(self, qwen_model_name="Qwen/Qwen2-0.5B", 
                 visual_dim=896, num_visual_tokens=784, num_queries=392):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.num_visual_tokens = num_visual_tokens
        self.num_queries = num_queries
        self.total_tokens = num_visual_tokens + num_queries
        
        # Load pre-trained Qwen2 model
        self.qwen = Qwen2Model.from_pretrained(qwen_model_name)
        self.hidden_size = self.qwen.config.hidden_size
        
        # Linear projection if needed
        if visual_dim != self.hidden_size:
            self.projection = nn.Linear(visual_dim, self.hidden_size)
        else:
            self.projection = None
        
        # Learnable causal queries for Stage 3
        self.causal_queries = nn.Parameter(
            torch.randn(1, num_queries, self.hidden_size) * 0.02
        )
    
    def _build_custom_attention_mask(self, device, dtype):
        """
        Build custom 2D attention mask for Stage 3.
        
        Returns:
            mask: [total_tokens, total_tokens] with 0 for allowed, -inf for masked
        """
        n_vis = self.num_visual_tokens
        n_qry = self.num_queries
        total = self.total_tokens
        
        # Initialize with -inf (everything masked)
        mask = torch.full((total, total), float('-inf'), device=device, dtype=dtype)

        # qwen2 mask is used by addition not multiplication
        # mask like
        # v,v,v,v,v,v,q,q,q
        # 0,0,0,0,0,0,-inf,-inf,-inf
        # 0,0,0,0,0,0,-inf,-inf,-inf
        # 0,0,0,0,0,0,-inf,-inf,-inf
        # 0,0,0,0,0,0,-inf,-inf,-inf
        # 0,0,0,0,0,0,-inf,-inf,-inf
        # 0,0,0,0,0,0,-inf,-inf,-inf
        # 0,0,0,0,0,0, 0,  -inf,-inf
        # 0,0,0,0,0,0, 0,    0, -inf
        # 0,0,0,0,0,0, 0,    0,   0
        
        # 1. Visual tokens can attend to all visual tokens (bidirectional)
        mask[:n_vis, :n_vis] = 0.0
        
        # # 2. Visual tokens can attend to all queries (visual sees queries)
        # mask[:n_vis, n_vis:] = 0.0
        
        # # 3. Queries can attend to all visual tokens (queries see visual)
        # mask[n_vis:, :n_vis] = 0.0
        
        # 4. Queries can attend to themselves with causal mask (only previous)
        for i in range(n_qry):
            mask[n_vis + i, n_vis:n_vis + i + 1] = 0.0
        
        return mask
    
    def _prepare_4d_mask(self, custom_2d_mask, batch_size, device, dtype):
        """
        Convert custom 2D mask to 4D format expected by Qwen2.
        
        Args:
            custom_2d_mask: [total_tokens, total_tokens]
            batch_size: Batch size
        
        Returns:
            attention_mask: [batch_size, 1, total_tokens, total_tokens]
        """
        # Expand to 4D: [1, 1, total, total]
        mask_4d = custom_2d_mask.unsqueeze(0).unsqueeze(0)
        # Expand to batch: [batch_size, 1, total, total]
        mask_4d = mask_4d.expand(batch_size, -1, -1, -1)
        return mask_4d
    
    def forward(self, visual_tokens, use_causal_queries=False):
        """
        Forward pass.
        
        Args:
            visual_tokens: [B, num_visual_tokens, visual_dim]
            use_causal_queries: If True, use custom mask with causal queries
        
        Returns:
            If use_causal_queries=False: [B, num_visual_tokens, hidden_size]
            If use_causal_queries=True:  [B, num_queries, hidden_size]
        """
        B = visual_tokens.shape[0]
        device = visual_tokens.device
        dtype = visual_tokens.dtype
        
        # Project visual tokens
        if self.projection is not None:
            visual_tokens = self.projection(visual_tokens)
        
        # Stage 2: no queries, use default causal mask
        if not use_causal_queries:
            outputs = self.qwen(
                inputs_embeds=visual_tokens,
                attention_mask=None,  # Qwen2 uses default causal mask
                output_attentions=False,
                output_hidden_states=False
            )
            return outputs.last_hidden_state
        
        # Stage 3: with queries and custom mask
        # Repeat queries for batch
        queries = self.causal_queries.repeat(B, 1, 1)
        
        # Concatenate: [visual_tokens, queries]
        input_tokens = torch.cat([visual_tokens, queries], dim=1)
        
        # Build custom 2D attention mask
        custom_mask_2d = self._build_custom_attention_mask(device, dtype)
        
        # Convert to 4D format
        attention_mask = self._prepare_4d_mask(custom_mask_2d, B, device, dtype)
        
        # Forward through Qwen2 with custom mask
        outputs = self.qwen(
            inputs_embeds=input_tokens,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False
        )
        
        encoded_tokens = outputs.last_hidden_state
        
        # Return only query part
        return encoded_tokens[:, -self.num_queries:, :]