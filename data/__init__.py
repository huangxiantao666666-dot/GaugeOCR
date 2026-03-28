# Data module for Gauge OCR
# This module contains dataset classes and data transformation utilities

from .datasets import KeypointDataset, ReadingDataset
from .transforms import get_train_transforms, get_val_transforms

__all__ = [
    'KeypointDataset',      # Dataset for keypoint detection (Stage 1)
    'ReadingDataset',       # Dataset for reading recognition (Stages 2 & 3)
    'get_train_transforms', # Get training data augmentation transforms
    'get_val_transforms'    # Get validation data transforms
]
