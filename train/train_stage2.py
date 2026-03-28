"""
Stage 2 Training Script: Semantic-Geometric Fusion.

This script trains the full model (without causal queries) to fuse semantic
features (from SAM) and geometric features (from DeepLabV3+) for reading
generation.

Key features of Stage 2:
- Geometric branch is frozen (pre-trained in Stage 1)
- SAM branch has lower learning rate (fine-tuned)
- Uses cross-attention with gate scores for feature fusion
- Optional sparsity regularization on gate scores

Data Flow:
Input Image [B, 3, 448, 448]
├─→ Semantic Branch (SAM) → Semantic Tokens [B, 784, dim]
└─→ Geometric Branch (frozen) → Geometric Features [B, 256, 112, 112]
       └─→ Gate Generator → Gate Scores [B, 784]

Cross-Attention: Semantic × Gate + Geometric → Fused Tokens
→ Causal Encoder (no causal queries)
→ LLM Decoder (with teacher forcing)
→ Cross-Entropy Loss [+ Sparsity Loss]
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

from data import ReadingDataset, get_train_transforms, get_val_transforms
from models import CausalGauge
from utils import load_config, set_seed, save_checkpoint, load_checkpoint, compute_reading_error, compute_accuracy_epsilon


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        args: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Train Stage 2: Semantic-Geometric Fusion')
    parser.add_argument('--config', type=str, default='configs/train_stage2.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, tokenizer, config, device, sparse_reg_weight=0.0, scaler=None):
    """
    Train the model for one epoch.
    
    Args:
        model: CausalGauge model
        dataloader: Training data loader
        optimizer: Optimizer with different learning rates for different parts
        tokenizer: Qwen2 tokenizer
        config: Configuration dictionary
        device: Device to use ('cuda' or 'cpu')
        sparse_reg_weight: Weight for sparsity regularization on gate scores
        scaler: Gradient scaler for mixed precision training
    
    Returns:
        metrics: Dictionary with average losses for the epoch
    """
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_sparse_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch in progress_bar:
        # Get batch data
        # Images shape: [B, 3, 448, 448]
        # Readings: list of strings
        images = batch['image'].to(device)
        readings = batch['reading']
        
        # Tokenize text readings for teacher forcing
        tokenized = tokenizer(
            readings,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        text_inputs = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        # Create labels by cloning inputs - we'll mask padding positions
        labels = text_inputs.clone()
        labels[attention_mask == 0] = -100  # -100 means ignore in loss
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if available
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # Forward pass through model (stage2 mode, no causal queries)
                outputs = model(
                    images,
                    text_inputs=text_inputs,
                    labels=labels,
                    stage='stage2',
                    use_causal_queries=False
                )
                
                # Get cross-entropy loss from LLM decoder
                ce_loss = outputs['loss']
                gate_scores = outputs['gate_scores']
                # Sparsity regularization: encourage gate scores to be sparse
                sparse_loss = torch.mean(torch.abs(gate_scores))
                # Total loss: CE + sparsity regularization
                loss = ce_loss + sparse_reg_weight * sparse_loss
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass without mixed precision
            outputs = model(
                images,
                text_inputs=text_inputs,
                labels=labels,
                stage='stage2',
                use_causal_queries=False
            )
            
            ce_loss = outputs['loss']
            gate_scores = outputs['gate_scores']
            sparse_loss = torch.mean(torch.abs(gate_scores))
            loss = ce_loss + sparse_reg_weight * sparse_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_sparse_loss += sparse_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{ce_loss.item():.4f}',
            'sparse': f'{sparse_loss.item():.4f}'
        })
    
    # Compute average losses
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'ce_loss': total_ce_loss / num_batches,
        'sparse_loss': total_sparse_loss / num_batches
    }


def validate(model, dataloader, tokenizer, config, device):
    """
    Validate the model on the validation set.
    
    Generates readings autoregressively and computes error metrics.
    
    Args:
        model: CausalGauge model
        dataloader: Validation data loader
        tokenizer: Qwen2 tokenizer
        config: Configuration dictionary
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        metrics: Dictionary with validation metrics
    """
    model.eval()
    all_preds = []
    all_gts = []
    
    # Disable gradient computation for validation
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validating')
        
        for batch in progress_bar:
            # Get batch data
            images = batch['image'].to(device)
            readings = batch['reading']
            
            # Generate readings autoregressively
            preds = model.generate(
                images,
                max_length=50,
                use_causal_queries=False
            )
            
            # Collect predictions and ground truths
            all_preds.extend(preds)
            all_gts.extend(readings)
    
    # Compute error metrics
    mean_abs_error, mean_rel_error = compute_reading_error(all_preds, all_gts)
    accuracy_epsilon = compute_accuracy_epsilon(all_preds, all_gts, epsilon=0.05)
    
    # Print sample predictions for debugging
    print(f'\nSample predictions:')
    for i in range(min(5, len(all_preds))):
        print(f'  GT: {all_gts[i]}, Pred: {all_preds[i]}')
    
    return {
        'mean_abs_error': mean_abs_error,
        'mean_rel_error': mean_rel_error,
        'accuracy_epsilon': accuracy_epsilon
    }


def main():
    """
    Main training function for Stage 2.
    
    Steps:
    1. Parse arguments and load config
    2. Set random seed for reproducibility
    3. Create datasets and dataloaders
    4. Initialize model and load geometric branch from Stage 1
    5. Freeze geometric branch (optional)
    6. Set up optimizer with different learning rates
    7. Train and validate for multiple epochs
    8. Save best model and checkpoints
    """
    args = parse_args()
    config = load_config(args.config)
    set_seed(args.seed)
    
    device = args.device
    print(f'Using device: {device}')
    
    # Create checkpoint directory
    checkpoint_dir = config['train']['stage2']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load datasets
    print('Loading datasets...')
    train_dataset = ReadingDataset(
        ann_file=config['data']['train_ann_file'],
        image_size=config['data']['image_size'],
        transforms=get_train_transforms()
    )
    val_dataset = ReadingDataset(
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
    
    # Initialize full model
    print('Initializing model...')
    model = CausalGauge(config).to(device)
    
    # Freeze geometric branch if specified
    if config['train']['stage2']['freeze_geometric']:
        print('Freezing geometric branch...')
        for param in model.geometric_branch.parameters():
            param.requires_grad = False
    
    # Load pre-trained geometric branch from Stage 1
    geometric_checkpoint = config['model'].get('geometric_checkpoint')
    if geometric_checkpoint and os.path.exists(geometric_checkpoint):
        print(f'Loading geometric branch from: {geometric_checkpoint}')
        model.geometric_branch.load_state_dict(torch.load(geometric_checkpoint, map_location=device), strict=False)
    
    # Get tokenizer from model
    tokenizer = model.llm_decoder.tokenizer
    
    # Set up optimizer with different learning rates
    # - SAM branch: lower LR (0.1x) to avoid destroying pre-trained features
    # - Other parts: full LR
    optimizer_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'semantic_branch' in name:
                lr = config['train']['stage2']['lr'] * 0.1
            else:
                lr = config['train']['stage2']['lr']
            optimizer_params.append({'params': param, 'lr': lr})
    
    optimizer = AdamW(optimizer_params, weight_decay=1e-4)
    
    # Initialize learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['train']['stage2']['epochs'],
        eta_min=config['train']['stage2']['lr_decay']
    )
    
    # Get sparsity regularization weight
    sparse_reg_weight = config['train']['stage2'].get('sparse_reg_weight', 0.0)
    
    # Initialize gradient scaler
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    best_val_error = float('inf')
    
    # Training loop
    print('Starting training...')
    for epoch in range(config['train']['stage2']['epochs']):
        print(f'\nEpoch {epoch + 1}/{config["train"]["stage2"]["epochs"]}')
        
        # Train one epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, tokenizer,
            config, device, sparse_reg_weight, scaler
        )
        print(f'Train - Loss: {train_metrics["loss"]:.4f}, CE: {train_metrics["ce_loss"]:.4f}')
        
        # Validate
        val_metrics = validate(model, val_loader, tokenizer, config, device)
        print(f'Val - Abs Error: {val_metrics["mean_abs_error"]:.4f}, Rel Error: {val_metrics["mean_rel_error"]:.4f}, Acc@0.05: {val_metrics["accuracy_epsilon"]:.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Save best model based on mean absolute error
        if val_metrics['mean_abs_error'] < best_val_error:
            best_val_error = val_metrics['mean_abs_error']
            save_path = os.path.join(checkpoint_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch + 1, val_metrics, save_path)
            print(f'Saved best model with abs error: {best_val_error:.4f}')
        
        # Save regular checkpoint
        save_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        save_checkpoint(model, optimizer, epoch + 1, val_metrics, save_path)
    
    print(f'\nTraining complete! Best val abs error: {best_val_error:.4f}')


if __name__ == '__main__':
    main()
