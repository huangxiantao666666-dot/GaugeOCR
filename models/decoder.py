import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer


class LLMDecoder(nn.Module):
    """
    LLM Decoder: Qwen2-based autoregressive decoder.
    
    This decoder generates text autoregressively from visual tokens.
    It can be used in two modes:
    1. Training mode: With text inputs and labels for teacher forcing
    2. Inference mode: Generating text token by token
    
    Data Flow (Training):
    Visual Tokens [B, num_tokens, visual_dim]
    → Projection (if needed) → [B, num_tokens, llm_dim]
    → Concatenate with text embeddings → [B, num_tokens+text_len, llm_dim]
    → Qwen2 Model
    → Loss + Logits
    
    Data Flow (Inference):
    Visual Tokens [B, num_tokens, visual_dim]
    → Projection (if needed) → [B, num_tokens, llm_dim]
    → Qwen2 Generation
    → Generated Text
    """
    
    def __init__(self, qwen_model_name="Qwen/Qwen2-0.5B", visual_dim=896):
        """
        Initialize the LLM decoder.
        
        Args:
            qwen_model_name: Name of the pre-trained Qwen2 model
            visual_dim: Dimension of input visual tokens
        """
        super().__init__()
        
        # Load tokenizer and set pad token
        self.tokenizer = Qwen2Tokenizer.from_pretrained(qwen_model_name)
        # Qwen2 doesn't have a pad token by default, use eos_token instead
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load pre-trained Qwen2 causal LM model
        self.llm = Qwen2ForCausalLM.from_pretrained(qwen_model_name)
        
        # Get Qwen2's hidden dimension
        self.llm_dim = self.llm.config.hidden_size
        
        # Linear projection if visual dimension doesn't match LLM's hidden size
        if visual_dim != self.llm_dim:
            self.visual_projection = nn.Linear(visual_dim, self.llm_dim)
        else:
            self.visual_projection = None
    
    def forward(self, visual_tokens, text_inputs=None, labels=None):
        """
        Forward pass through the LLM decoder (for training).
        
        Args:
            visual_tokens: Visual tokens from encoder [B, num_tokens, visual_dim]
            text_inputs: Text input IDs [B, text_len] (for teacher forcing)
            labels: Target labels [B, text_len] (for loss computation)
        
        Returns:
            outputs: Model outputs containing loss and logits
        """
        # Project visual tokens to LLM's hidden dimension if needed
        # Shape change: [B, num_tokens, visual_dim] → [B, num_tokens, llm_dim]
        if self.visual_projection is not None:
            visual_tokens = self.visual_projection(visual_tokens)
        
        # If text inputs are provided (training mode)
        if text_inputs is not None:
            # Get text embeddings from the LLM's embedding layer
            # Shape change: [B, text_len] → [B, text_len, llm_dim]
            text_embeddings = self.llm.get_input_embeddings()(text_inputs)
            
            # Concatenate visual tokens and text embeddings
            # Visual tokens come first, then text tokens
            # Shape change: [B, num_tokens, llm_dim] + [B, text_len, llm_dim]
            # → [B, num_tokens+text_len, llm_dim]
            inputs_embeds = torch.cat([visual_tokens, text_embeddings], dim=1)
            
            # Create attention mask
            # Visual tokens: all 1s (always attend to them)
            visual_attention_mask = torch.ones(
                visual_tokens.shape[0], visual_tokens.shape[1]
            ).to(visual_tokens.device)
            # Text tokens: 1 for non-pad tokens, 0 for pad tokens
            text_attention_mask = (text_inputs != self.tokenizer.pad_token_id).float()
            # Concatenate masks
            # Shape change: [B, num_tokens] + [B, text_len] → [B, num_tokens+text_len]
            attention_mask = torch.cat([visual_attention_mask, text_attention_mask], dim=1)
            
            # Prepare labels if provided
            if labels is not None:
                # Create dummy labels for visual tokens (-100 means ignore in loss)
                visual_labels = torch.full(
                    (visual_tokens.shape[0], visual_tokens.shape[1]),
                    -100,
                    dtype=torch.long
                ).to(labels.device)
                # Concatenate visual labels (ignored) with text labels
                # Shape change: [B, num_tokens] + [B, text_len] → [B, num_tokens+text_len]
                labels = torch.cat([visual_labels, labels], dim=1)
            
            # Pass through Qwen2 model
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
        else:
            # No text inputs - just pass visual tokens through (inference mode)
            outputs = self.llm(inputs_embeds=visual_tokens, return_dict=True)
        
        return outputs
    
    def generate(self, visual_tokens, max_length=50, **kwargs):
        """
        Autoregressively generate text from visual tokens (for inference).
        
        Args:
            visual_tokens: Visual tokens from encoder [B, num_tokens, visual_dim]
            max_length: Maximum number of new tokens to generate
            **kwargs: Additional generation arguments (temperature, top_k, etc.)
        
        Returns:
            generated_text: List of generated text strings
        """
        # Project visual tokens to LLM's hidden dimension if needed
        # Shape change: [B, num_tokens, visual_dim] → [B, num_tokens, llm_dim]
        if self.visual_projection is not None:
            visual_tokens = self.visual_projection(visual_tokens)
        
        # Create attention mask for visual tokens (all 1s)
        visual_attention_mask = torch.ones(
            visual_tokens.shape[0], visual_tokens.shape[1]
        ).to(visual_tokens.device)
        
        # Generate text autoregressively
        generated_ids = self.llm.generate(
            inputs_embeds=visual_tokens,
            attention_mask=visual_attention_mask,
            max_new_tokens=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        # Decode generated IDs to text
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
