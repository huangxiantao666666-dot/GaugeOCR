"""
Keypoint Detection and Visualization for Gauge OCR
==================================================

This module provides utilities for:
1. Loading trained DeepLabV3+ model from checkpoints
2. Detecting keypoints (pointer and scale marks) from gauge images
3. Visualizing segmentation results
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from models.deeplabv3plus import create_deeplabv3plus


# Class colors for visualization
# 0: Background (black), 1: Pointer (red), 2: Scale (green)
CLASS_COLORS = np.array([
    [0, 0, 0],      # Background
    [255, 0, 0],    # Pointer
    [0, 255, 0]     # Scale
], dtype=np.uint8)


def load_trained_model(checkpoint_path, num_classes=3, device='cpu'):
    """
    Load a trained DeepLabV3+ model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pth)
        num_classes: Number of classes (default: 3 for background, pointer, scale)
        device: Device to load the model on
        
    Returns:
        model: Loaded DeepLabV3+ model with trained weights
    """
    print(f"[load_trained_model] Creating DeepLabV3+ model with {num_classes} classes")
    
    # Create model with custom num_classes
    model = create_deeplabv3plus(
        backbone='resnet50',
        num_classes=num_classes,
        output_stride=16,
        pretrained=False
    )
    
    # Load checkpoint
    print(f"[load_trained_model] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"[load_trained_model] Model loaded successfully!")
    if 'val_iou' in checkpoint:
        print(f"[load_trained_model] Checkpoint mIoU: {checkpoint['val_iou']:.4f}")
    
    return model


def preprocess_image(image_path, image_size=(448, 448)):
    """
    Preprocess an image for segmentation.
    
    Args:
        image_path: Path to the input image
        image_size: Target size for resizing
        
    Returns:
        image_tensor: Preprocessed image tensor [1, 3, H, W]
        original_image: Original PIL image
    """
    # Load image
    original_image = Image.open(image_path).convert('RGB')
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(original_image).unsqueeze(0)
    
    return image_tensor, original_image


def predict_segmentation(model, image_tensor, device='cpu'):
    """
    Predict segmentation mask from an image.
    
    Args:
        model: DeepLabV3+ model
        image_tensor: Input image tensor [1, 3, H, W]
        device: Device to run inference on
        
    Returns:
        pred_mask: Predicted segmentation mask [H, W] (class indices)
        pred_probs: Prediction probabilities [H, W, num_classes]
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        probs = F.softmax(output, dim=1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    return pred, probs


def visualize_segmentation(original_image, pred_mask, gt_mask=None, image_size=(448, 448)):
    """
    Visualize segmentation results.
    
    Args:
        original_image: Original PIL image
        pred_mask: Predicted segmentation mask [H, W]
        gt_mask: Ground truth mask [H, W] (optional)
        image_size: Size for display
        
    Returns:
        vis_image: Visualization image [H, W, 3]
    """
    # Resize original image to match prediction size
    original_resized = original_image.resize((pred_mask.shape[1], pred_mask.shape[0]))
    original_np = np.array(original_resized)
    
    # Create color mask for prediction
    pred_color = pred_mask_to_color(pred_mask)
    
    # Blend original image with prediction
    blend = (original_np.astype(np.float32) * 0.5 + pred_color.astype(np.float32) * 0.5).astype(np.uint8)
    
    # Create visualization
    if gt_mask is not None:
        gt_color = pred_mask_to_color(gt_mask)
        vis_image = np.hstack([original_np, pred_color, gt_color, blend])
    else:
        vis_image = np.hstack([original_np, pred_color, blend])
    
    return vis_image


def pred_mask_to_color(pred_mask):
    """
    Convert prediction mask to RGB color image.
    
    Args:
        pred_mask: Segmentation mask [H, W] with class indices
        
    Returns:
        color_mask: RGB color mask [H, W, 3]
    """
    h, w = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx, color in enumerate(CLASS_COLORS):
        color_mask[pred_mask == class_idx] = color
    
    return color_mask


def load_ground_truth(gt_path, image_size=(448, 448)):
    """
    Load ground truth annotation.
    
    Args:
        gt_path: Path to ground truth annotation file
        image_size: Target size for resizing
        
    Returns:
        gt_mask: Ground truth mask [H, W]
    """
    gt_image = Image.open(gt_path)
    gt_mask = np.array(gt_image)
    
    # Resize with NEAREST interpolation to preserve class indices
    gt_image_resized = gt_image.resize(image_size, Image.NEAREST)
    gt_mask = np.array(gt_image_resized)
    
    # Clip to valid range
    gt_mask = np.clip(gt_mask, 0, 2).astype(np.uint8)
    
    return gt_mask


def process_single_image(model, image_path, gt_path=None, device='cpu', save_path=None):
    """
    Process a single image and visualize results.
    
    Args:
        model: Trained DeepLabV3+ model
        image_path: Path to input image
        gt_path: Path to ground truth annotation (optional)
        device: Device to run inference on
        save_path: Path to save visualization (optional)
        
    Returns:
        vis_image: Visualization image
    """
    # Preprocess
    image_tensor, original_image = preprocess_image(image_path)
    
    # Predict
    pred_mask, pred_probs = predict_segmentation(model, image_tensor, device)
    
    # Load ground truth if provided
    gt_mask = None
    if gt_path and os.path.exists(gt_path):
        gt_mask = load_ground_truth(gt_path)
    
    # Visualize
    vis_image = visualize_segmentation(original_image, pred_mask, gt_mask)
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis_image_pil = Image.fromarray(vis_image)
        vis_image_pil.save(save_path)
        print(f"[process_single_image] Visualization saved to: {save_path}")
    
    return vis_image


def process_dataset(model, images_dir, annotations_dir, output_dir, device='cpu', limit=10):
    """
    Process multiple images from a dataset.
    
    Args:
        model: Trained DeepLabV3+ model
        images_dir: Directory containing input images
        annotations_dir: Directory containing ground truth annotations
        output_dir: Directory to save visualizations
        device: Device to run inference on
        limit: Maximum number of images to process (None for all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    if limit:
        image_files = image_files[:limit]
    
    print(f"[process_dataset] Processing {len(image_files)} images...")
    
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        gt_file = image_file.replace('.jpg', '.png').replace('.jpeg', '.png')
        gt_path = os.path.join(annotations_dir, gt_file)
        
        if not os.path.exists(gt_path):
            gt_path = None
        
        output_path = os.path.join(output_dir, f"vis_{image_file.replace('.jpg', '.png')}")
        
        print(f"[{idx+1}/{len(image_files)}] Processing: {image_file}")
        process_single_image(model, image_path, gt_path, device, output_path)
    
    print(f"[process_dataset] Done! Visualizations saved to: {output_dir}")


def calculate_iou(pred_mask, gt_mask, num_classes=3):
    """
    Calculate Intersection over Union (IoU) for each class.
    
    Args:
        pred_mask: Predicted mask [H, W]
        gt_mask: Ground truth mask [H, W]
        num_classes: Number of classes
        
    Returns:
        ious: IoU for each class
        miou: Mean IoU
    """
    ious = []
    
    for class_idx in range(num_classes):
        pred_binary = (pred_mask == class_idx)
        gt_binary = (gt_mask == class_idx)
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    miou = np.nanmean(ious)
    
    return ious, miou


def evaluate_model(model, images_dir, annotations_dir, device='cpu', limit=None):
    """
    Evaluate model performance on a dataset.
    
    Args:
        model: Trained DeepLabV3+ model
        images_dir: Directory containing input images
        annotations_dir: Directory containing ground truth annotations
        device: Device to run inference on
        limit: Maximum number of images to evaluate (None for all)
        
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    if limit:
        image_files = image_files[:limit]
    
    all_ious = {0: [], 1: [], 2: []}
    
    print(f"[evaluate_model] Evaluating on {len(image_files)} images...")
    
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        gt_file = image_file.replace('.jpg', '.png').replace('.jpeg', '.png')
        gt_path = os.path.join(annotations_dir, gt_file)
        
        if not os.path.exists(gt_path):
            continue
        
        # Preprocess and predict
        image_tensor, _ = preprocess_image(image_path)
        pred_mask, _ = predict_segmentation(model, image_tensor, device)
        
        # Load ground truth
        gt_mask = load_ground_truth(gt_path)
        
        # Calculate IoU
        ious, _ = calculate_iou(pred_mask, gt_mask)
        
        for class_idx, iou in enumerate(ious):
            if not np.isnan(iou):
                all_ious[class_idx].append(iou)
        
        if (idx + 1) % 10 == 0:
            print(f"[evaluate_model] Processed {idx+1}/{len(image_files)} images")
    
    # Calculate mean IoU for each class
    mean_ious = {}
    class_names = ['Background', 'Pointer', 'Scale']
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for class_idx, class_name in enumerate(class_names):
        if all_ious[class_idx]:
            mean_iou = np.mean(all_ious[class_idx])
            mean_ious[class_name] = mean_iou
            print(f"{class_name:15s}: IoU = {mean_iou:.4f}")
        else:
            print(f"{class_name:15s}: No samples")
    
    overall_miou = np.mean([v for v in mean_ious.values()])
    print("-"*60)
    print(f"Overall mIoU  : {overall_miou:.4f}")
    print("="*60)
    
    return {
        'class_ious': mean_ious,
        'mIoU': overall_miou
    }


if __name__ == '__main__':
    # Configuration
    CHECKPOINT_PATH = 'checkpoints/segmentation/final_model.pth'
    IMAGES_DIR = 'dataset/meter_seg/meter_seg/images/val'
    ANNOTATIONS_DIR = 'dataset/meter_seg/meter_seg/annotations/val'
    OUTPUT_DIR = 'outputs/visualization'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_trained_model(CHECKPOINT_PATH, device=device)
    
    # Process some images
    print("\n" + "="*60)
    print("VISUALIZING SEGMENTATION RESULTS")
    print("="*60)
    process_dataset(model, IMAGES_DIR, ANNOTATIONS_DIR, OUTPUT_DIR, device, limit=10)
    
    # Evaluate
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    evaluate_model(model, IMAGES_DIR, ANNOTATIONS_DIR, device)
