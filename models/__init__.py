# Models module for Gauge OCR
# This module contains all neural network components of the CausalGauge model

from .semantic_branch import SemanticBranch
from .geometric_branch import GeometricBranch
from .gate import GateGenerator
from .cross_attention import CrossAttentionModule
from .causal_encoder import CausalEncoder
from .decoder import LLMDecoder
from .gauge_ocr import CausalGauge

__all__ = [
    'SemanticBranch',       # SAM-based semantic feature extraction
    'GeometricBranch',      # DeepLabV3+ based geometric feature extraction
    'GateGenerator',        # Generates gate scores from geometric features
    'CrossAttentionModule', # Cross-attention between semantic and geometric features
    'CausalEncoder',        # Causal flow encoder with spatial bias
    'LLMDecoder',           # Qwen2-based LLM decoder
    'CausalGauge'           # End-to-end CausalGauge model
]
