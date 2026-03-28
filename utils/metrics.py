import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re


def extract_numeric_value(text):
    """
    Extract numeric value from text using regular expressions.
    
    Data Flow:
    Input: text string (e.g., "0.45 MPa", "pressure: 1.23 bar")
    → Regex search for numeric pattern
    → Convert to float if found
    Output: float value or None
    
    Args:
        text: Input text string
    
    Returns:
        Extracted numeric value as float, or None if no number found
    """
    # Regex pattern to match numbers (integers, decimals, positive/negative)
    # Pattern explanation:
    # [-+]? - optional sign
    # \d* - optional integer part
    # \.? - optional decimal point
    # \d+ - required fractional part
    match = re.search(r'[-+]?\d*\.?\d+', text)
    if match:
        return float(match.group())
    return None


class DiceLoss(nn.Module):
    """
    Dice Loss function for segmentation tasks.
    
    Dice Loss measures the overlap between prediction and target.
    Range: [0, 1], where 0 is perfect overlap.
    
    Formula:
    Dice = 2 * (intersection + smooth) / (|pred| + |target| + smooth)
    Dice Loss = 1 - Dice
    
    Data Flow:
    Logits [B, C, H, W]
    → Sigmoid (convert to probabilities)
    → Compute intersection and union
    → Compute Dice coefficient
    → Loss = 1 - Dice
    Output: scalar loss value
    """
    
    def __init__(self, smooth=1.0):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """
        Compute Dice Loss.
        
        Args:
            logits: Prediction logits [B, C, H, W]
            targets: Ground truth targets [B, C, H, W] (values in [0, 1])
        
        Returns:
            dice_loss: Scalar Dice loss value
        """
        # Apply sigmoid to convert logits to probabilities [0, 1]
        logits = torch.sigmoid(logits)
        
        # Compute intersection: element-wise product summed over spatial dimensions
        # Shape changes: [B, C, H, W] → [B, C]
        intersection = (logits * targets).sum(dim=(2, 3))
        
        # Compute union: sum of predictions and targets
        # Shape changes: [B, C, H, W] → [B, C]
        union = logits.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        # Compute Dice coefficient per batch and channel, then average
        # Shape: [B, C] → scalar
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss


def compute_keypoint_error(pred_heatmaps, gt_heatmaps):
    """
    Compute keypoint detection error from heatmaps.
    
    Finds the peak of each heatmap (argmax) and computes Euclidean distance
    between predicted and ground truth keypoint locations.
    
    Data Flow:
    pred_heatmaps [B, C, H, W], gt_heatmaps [B, C, H, W]
    → For each heatmap, find (x, y) of maximum value
    → Compute Euclidean distance between pred and gt
    → Average all errors
    Output: mean error in pixels
    
    Args:
        pred_heatmaps: Predicted heatmaps [B, C, H, W]
        gt_heatmaps: Ground truth heatmaps [B, C, H, W]
    
    Returns:
        mean_error: Average keypoint localization error in pixels
    """
    B, C, H, W = pred_heatmaps.shape
    
    # Convert to numpy for easier processing
    pred_heatmaps = pred_heatmaps.detach().cpu().numpy()
    gt_heatmaps = gt_heatmaps.detach().cpu().numpy()
    
    errors = []
    
    # Iterate over batch and channels
    for b in range(B):
        for c in range(C):
            # Get single heatmap
            pred_hm = pred_heatmaps[b, c]
            gt_hm = gt_heatmaps[b, c]
            
            # Find coordinates of maximum value (keypoint location)
            # unravel_index converts flat index to 2D coordinates
            y_pred, x_pred = np.unravel_index(pred_hm.argmax(), pred_hm.shape)
            y_gt, x_gt = np.unravel_index(gt_hm.argmax(), gt_hm.shape)
            
            # Compute Euclidean distance
            error = np.sqrt((x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2)
            errors.append(error)
    
    # Return mean error across all keypoints and all samples
    return np.mean(errors)


def compute_reading_error(pred_readings, gt_readings):
    """
    Compute reading recognition error.
    
    Extracts numeric values from both predicted and ground truth readings,
    then computes absolute and relative errors.
    
    Data Flow:
    pred_readings list[str], gt_readings list[str]
    → Extract numeric values
    → Compute absolute error: |pred - gt|
    → Compute relative error: |pred - gt| / |gt|
    → Average all errors
    Output: (mean_abs_error, mean_rel_error)
    
    Args:
        pred_readings: List of predicted reading strings
        gt_readings: List of ground truth reading strings
    
    Returns:
        tuple: (mean_absolute_error, mean_relative_error)
    """
    absolute_errors = []
    relative_errors = []
    
    for pred, gt in zip(pred_readings, gt_readings):
        # Extract numeric values from text
        pred_val = extract_numeric_value(pred)
        gt_val = extract_numeric_value(gt)
        
        # Only compute error if both values are successfully extracted
        if pred_val is not None and gt_val is not None:
            # Absolute error
            abs_error = abs(pred_val - gt_val)
            absolute_errors.append(abs_error)
            
            # Relative error (avoid division by zero)
            if gt_val != 0:
                rel_error = abs_error / abs(gt_val)
                relative_errors.append(rel_error)
    
    # Compute mean errors (handle empty lists)
    mean_abs_error = np.mean(absolute_errors) if absolute_errors else float('inf')
    mean_rel_error = np.mean(relative_errors) if relative_errors else float('inf')
    
    return mean_abs_error, mean_rel_error


def compute_accuracy_epsilon(pred_readings, gt_readings, epsilon=0.05):
    """
    Compute Accuracy ε: proportion of predictions within [gt - ε, gt + ε].
    
    This metric measures how many predictions are "close enough" to the ground truth.
    
    Data Flow:
    pred_readings list[str], gt_readings list[str]
    → Extract numeric values
    → Check if |pred - gt| ≤ epsilon
    → Count correct predictions
    → Compute accuracy = correct / total
    Output: accuracy in [0, 1]
    
    Args:
        pred_readings: List of predicted reading strings
        gt_readings: List of ground truth reading strings
        epsilon: Error tolerance threshold
    
    Returns:
        accuracy: Proportion of correct predictions (within epsilon)
    """
    correct = 0
    total = 0
    
    for pred, gt in zip(pred_readings, gt_readings):
        # Extract numeric values from text
        pred_val = extract_numeric_value(pred)
        gt_val = extract_numeric_value(gt)
        
        # Only count if both values are successfully extracted
        if pred_val is not None and gt_val is not None:
            total += 1
            # Check if prediction is within epsilon tolerance
            if abs(pred_val - gt_val) <= epsilon:
                correct += 1
    
    # Return accuracy (handle division by zero)
    return correct / total if total > 0 else 0.0
