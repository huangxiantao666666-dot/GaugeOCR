# Utility module for Gauge OCR
# This module contains evaluation metrics, helper functions, and visualization tools

from .metrics import compute_keypoint_error, compute_reading_error, DiceLoss
from .utils import load_config, set_seed, save_checkpoint, load_checkpoint
from .visualize import visualize_heatmaps, visualize_gate_scores

__all__ = [
    # Metrics
    'compute_keypoint_error',    # Compute keypoint detection error from heatmaps
    'compute_reading_error',     # Compute reading recognition error
    'DiceLoss',                  # Dice loss for segmentation
    
    # Utilities
    'load_config',               # Load YAML configuration file
    'set_seed',                  # Set random seed for reproducibility
    'save_checkpoint',           # Save model checkpoint
    'load_checkpoint',           # Load model checkpoint
    
    # Visualization
    'visualize_heatmaps',        # Visualize keypoint heatmaps
    'visualize_gate_scores'      # Visualize gate scores
]
