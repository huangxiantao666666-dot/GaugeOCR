import json
import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch


def generate_gaussian_heatmap(shape, center, sigma=5):
    """
    Generate a Gaussian heatmap for a single keypoint.
    
    Data Flow:
    - Input: shape (h, w), center (x, y), sigma (float)
    - Creates a 2D grid of coordinates
    - Computes Gaussian distribution around center
    - Output: heatmap array [h, w] with values in [0, 1]
    
    Args:
        shape: Tuple (height, width) of the output heatmap
        center: Tuple (x, y) of the keypoint center (in pixel coordinates)
        sigma: Standard deviation of the Gaussian
    
    Returns:
        g: Gaussian heatmap array of shape (h, w)
    """
    h, w = shape
    # Create coordinate grids
    x = np.arange(0, w, 1, float)
    y = np.arange(0, h, 1, float)
    y, x = np.meshgrid(y, x)
    
    # Extract center coordinates
    x0, y0 = center
    
    # Compute Gaussian distribution
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return g


class KeypointDataset(Dataset):
    """
    Keypoint Dataset for Stage 1 training.
    Loads images and 6 keypoint coordinates, generates Gaussian heatmaps as supervision.
    
    Keypoint order (typically):
    0: Pointer tip
    1: Center point
    2-5: 4 main scale points
    """
    
    def __init__(self, ann_file, image_size=(448, 448), transforms=None):
        """
        Initialize the keypoint dataset.
        
        Args:
            ann_file: Path to annotation JSON file
            image_size: Target image size (height, width)
            transforms: Optional data transforms/augmentations
        """
        # Load annotations from JSON file
        with open(ann_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        self.image_size = image_size
        self.transforms = transforms
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Data Flow for each sample:
        1. Load image from disk [H, W, 3] (BGR format from OpenCV)
        2. Convert to RGB format
        3. Resize to target size [448, 448, 3]
        4. Load keypoints (normalized coordinates 0-1)
        5. Scale keypoints to resized image dimensions
        6. Generate Gaussian heatmaps for each keypoint [6, 448, 448]
        7. Apply transforms if provided
        8. Convert to PyTorch tensors
        
        Args:
            idx: Index of the sample to retrieve
        
        Returns:
            dict containing:
                - 'image': Image tensor [3, 448, 448]
                - 'heatmaps': Heatmap tensor [6, 448, 448]
                - 'keypoints': Keypoint coordinates tensor [6, 2]
        """
        # Get annotation for this index
        ann = self.annotations[idx]
        
        # Load and preprocess image
        img_path = ann['image_path']
        # Read image in BGR format
        image = cv2.imread(img_path)
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get original image dimensions
        orig_h, orig_w = image.shape[:2]
        # Resize image to target size
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        
        # Process keypoints: convert from normalized to pixel coordinates
        # Shape change: [6, 2] (normalized) -> [6, 2] (pixel coordinates)
        keypoints = np.array(ann['keypoints'])
        # Scale x coordinates (width)
        keypoints[:, 0] = keypoints[:, 0] * self.image_size[1]
        # Scale y coordinates (height)
        keypoints[:, 1] = keypoints[:, 1] * self.image_size[0]
        
        # Generate Gaussian heatmaps for each keypoint
        # Shape: [num_keypoints, height, width] = [6, 448, 448]
        heatmaps = np.zeros((len(keypoints), self.image_size[0], self.image_size[1]), dtype=np.float32)
        for i, (x, y) in enumerate(keypoints):
            # Only generate heatmap if keypoint is within image bounds
            if 0 <= x < self.image_size[1] and 0 <= y < self.image_size[0]:
                heatmaps[i] = generate_gaussian_heatmap(self.image_size, (x, y))
        
        # Apply transforms if provided
        if self.transforms:
            image = self.transforms(image)
        else:
            # Default: convert to tensor and normalize to [0, 1]
            # Shape change: [448, 448, 3] -> [3, 448, 448]
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Convert numpy arrays to PyTorch tensors
        heatmaps = torch.from_numpy(heatmaps)
        
        return {
            'image': image,
            'heatmaps': heatmaps,
            'keypoints': torch.from_numpy(keypoints)
        }


class ReadingDataset(Dataset):
    """
    Reading Dataset for Stage 2 and 3 training.
    Loads images and corresponding gauge reading text.
    """
    
    def __init__(self, ann_file, image_size=(448, 448), transforms=None):
        """
        Initialize the reading dataset.
        
        Args:
            ann_file: Path to annotation JSON file
            image_size: Target image size (height, width)
            transforms: Optional data transforms/augmentations
        """
        # Load annotations from JSON file
        with open(ann_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        self.image_size = image_size
        self.transforms = transforms
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Data Flow for each sample:
        1. Load image from disk [H, W, 3] (BGR format)
        2. Convert to RGB format
        3. Resize to target size [448, 448, 3]
        4. Load reading text
        5. Apply transforms if provided
        6. Convert image to PyTorch tensor
        
        Args:
            idx: Index of the sample to retrieve
        
        Returns:
            dict containing:
                - 'image': Image tensor [3, 448, 448]
                - 'reading': Reading string (e.g., "0.45 MPa")
        """
        # Get annotation for this index
        ann = self.annotations[idx]
        
        # Load and preprocess image
        img_path = ann['image_path']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to target size
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        
        # Get the reading text
        reading = ann['reading']
        
        # Apply transforms if provided
        if self.transforms:
            image = self.transforms(image)
        else:
            # Default: convert to tensor and normalize to [0, 1]
            # Shape change: [448, 448, 3] -> [3, 448, 448]
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image,
            'reading': reading
        }
