"""
Stage 1 Training Script: Geometric Branch Pre-training.

This script trains the DeepLabV3+ based geometric branch to detect keypoints:
- Pointer tip
- Gauge center
- 4 main scale points

The training uses a combination of MSE loss (for heatmap regression) and
Dice loss (for better segmentation quality).

Data Flow:
Input Image [B, 3, 448, 448]
→ Geometric Branch (DeepLabV3+)
→ Heatmaps [B, 6, 112, 112] (6 keypoints)
→ Loss = λ_mse * MSE_Loss + λ_dice * Dice_Loss
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import KeypointDataset, get_train_transforms, get_val_transforms
from models import GeometricBranch
from utils import load_config, set_seed, save_checkpoint, DiceLoss, compute_keypoint_error


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        args: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Train Stage 1: Geometric Branch')
    parser.add_argument('--config', type=str, default='configs/train_stage1.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, criterion_heatmap, criterion_dice, loss_weights, device, scaler=None):
    """
    Train the model for one epoch.
    
    Args:
        model: Geometric branch model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion_heatmap: MSE loss for heatmaps
        criterion_dice: Dice loss for segmentation
        loss_weights: Dictionary with 'heatmap' and 'dice' weights
        device: Device to use ('cuda' or 'cpu')
        scaler: Gradient scaler for mixed precision training
    
    Returns:
        metrics: Dictionary with average losses for the epoch
    """
    model.train()
    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_dice_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch in progress_bar:
        # Move data to device
        # Images shape: [B, 3, 448, 448]
        # Heatmaps shape: [B, 6, 112, 112] (6 keypoints)
        images = batch['image'].to(device)
        heatmaps_gt = batch['heatmaps'].to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if available
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # Get predictions from model
                outputs = model(images, return_heatmap=True)
                heatmaps_pred = outputs[1]
                
                # Compute losses
                # MSE loss between predicted and ground truth heatmaps
                loss_heatmap = criterion_heatmap(heatmaps_pred, heatmaps_gt)
                # Dice loss for better segmentation quality
                loss_dice = criterion_dice(heatmaps_pred, heatmaps_gt)
                # Weighted sum of losses
                loss = loss_weights['heatmap'] * loss_heatmap + loss_weights['dice'] * loss_dice
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass without mixed precision
            outputs = model(images, return_heatmap=True)
            heatmaps_pred = outputs[1]
            
            loss_heatmap = criterion_heatmap(heatmaps_pred, heatmaps_gt)
            loss_dice = criterion_dice(heatmaps_pred, heatmaps_gt)
            loss = loss_weights['heatmap'] * loss_heatmap + loss_weights['dice'] * loss_dice
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_heatmap_loss += loss_heatmap.item()
        total_dice_loss += loss_dice.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'heatmap': f'{loss_heatmap.item():.4f}',
            'dice': f'{loss_dice.item():.4f}'
        })
    
    # Compute average losses
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'heatmap_loss': total_heatmap_loss / num_batches,
        'dice_loss': total_dice_loss / num_batches
    }


def validate(model, dataloader, criterion_heatmap, criterion_dice, loss_weights, device):
    """
    Validate the model on the validation set.
    
    Args:
        model: Geometric branch model
        dataloader: Validation data loader
        criterion_heatmap: MSE loss for heatmaps
        criterion_dice: Dice loss for segmentation
        loss_weights: Dictionary with 'heatmap' and 'dice' weights
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        metrics: Dictionary with validation metrics including keypoint error
    """
    model.eval()
    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_dice_loss = 0.0
    total_keypoint_error = 0.0
    
    # Disable gradient computation for validation
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validating')
        
        for batch in progress_bar:
            # Move data to device
            images = batch['image'].to(device)
            heatmaps_gt = batch['heatmaps'].to(device)
            
            # Forward pass
            outputs = model(images, return_heatmap=True)
            heatmaps_pred = outputs[1]
            
            # Compute losses
            loss_heatmap = criterion_heatmap(heatmaps_pred, heatmaps_gt)
            loss_dice = criterion_dice(heatmaps_pred, heatmaps_gt)
            loss = loss_weights['heatmap'] * loss_heatmap + loss_weights['dice'] * loss_dice
            
            # Compute keypoint localization error
            # Finds peak in heatmap and computes distance to ground truth
            keypoint_error = compute_keypoint_error(heatmaps_pred, heatmaps_gt)
            
            # Accumulate metrics
            total_loss += loss.item()
            total_heatmap_loss += loss_heatmap.item()
            total_dice_loss += loss_dice.item()
            total_keypoint_error += keypoint_error
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'error': f'{keypoint_error:.2f}'
            })
    
    # Compute average metrics
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'heatmap_loss': total_heatmap_loss / num_batches,
        'dice_loss': total_dice_loss / num_batches,
        'keypoint_error': total_keypoint_error / num_batches
    }


def main():
    """
    Main training function for Stage 1.
    
    Steps:
    1. Parse arguments and load config
    2. Set random seed for reproducibility
    3. Create datasets and dataloaders
    4. Initialize model, losses, optimizer, scheduler
    5. Train and validate for multiple epochs
    6. Save best model and checkpoints
    """
    args = parse_args()
    config = load_config(args.config)
    set_seed(args.seed)
    
    device = args.device
    print(f'Using device: {device}')
    
    # Create checkpoint directory
    checkpoint_dir = config['train']['stage1']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load datasets
    print('Loading datasets...')
    train_dataset = KeypointDataset(
        ann_file=config['data']['train_ann_file'],
        image_size=config['data']['image_size'],
        transforms=get_train_transforms()
    )
    val_dataset = KeypointDataset(
        ann_file=config['data']['val_ann_file'],
        image_size=config['data']['image_size'],
        transforms=get_val_transforms()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
    
    # Initialize model
    print('Initializing model...')
    model = GeometricBranch(
        backbone=config['model']['deeplab_backbone'],
        num_keypoints=config['model']['num_keypoints']
    ).to(device)
    
    # Initialize losses
    criterion_heatmap = nn.MSELoss()
    criterion_dice = DiceLoss()
    loss_weights = config['train']['stage1']['loss_weights']
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['train']['stage1']['lr'],
        weight_decay=config['train']['stage1']['weight_decay']
    )
    
    # Initialize learning rate scheduler (cosine annealing)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['train']['stage1']['epochs'],
        eta_min=config['train']['stage1']['lr_decay']
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    best_val_error = float('inf')
    
    # Training loop
    print('Starting training...')
    for epoch in range(config['train']['stage1']['epochs']):
        print(f'\nEpoch {epoch + 1}/{config["train"]["stage1"]["epochs"]}')
        
        # Train one epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion_heatmap,
            criterion_dice, loss_weights, device, scaler
        )
        print(f'Train - Loss: {train_metrics["loss"]:.4f}, Heatmap: {train_metrics["heatmap_loss"]:.4f}, Dice: {train_metrics["dice_loss"]:.4f}')
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion_heatmap,
            criterion_dice, loss_weights, device
        )
        print(f'Val - Loss: {val_metrics["loss"]:.4f}, Error: {val_metrics["keypoint_error"]:.2f}')
        
        # Update learning rate
        scheduler.step()
        
        # Save best model based on keypoint error
        if val_metrics['keypoint_error'] < best_val_error:
            best_val_error = val_metrics['keypoint_error']
            save_path = os.path.join(checkpoint_dir, 'best_geometric.pth')
            torch.save(model.state_dict(), save_path)
            print(f'Saved best model with error: {best_val_error:.2f}')
            
            # Also save a version without the keypoint head for transfer to later stages
            model_without_head = GeometricBranch(
                backbone=config['model']['deeplab_backbone'],
                num_keypoints=config['model']['num_keypoints']
            )
            model_without_head.load_state_dict(model.state_dict())
            model_without_head.keypoint_head = None
            save_path_no_head = os.path.join(checkpoint_dir, 'best_geometric_no_head.pth')
            torch.save(model_without_head.state_dict(), save_path_no_head)
        
        # Save regular checkpoint
        save_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        save_checkpoint(model, optimizer, epoch + 1, val_metrics, save_path)
    
    print(f'\nTraining complete! Best val error: {best_val_error:.2f}')


if __name__ == '__main__':
    main()
