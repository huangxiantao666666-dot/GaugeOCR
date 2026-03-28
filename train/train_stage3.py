"""
Stage 3 Training Script: Causal Flow Reordering.

This script trains the causal encoder and LLM decoder with causal queries
for better spatial-temporal reasoning and reading generation.

Key features of Stage 3:
- Vision branches (SAM, geometric, gate, cross-attention) are frozen
- Only causal encoder and LLM decoder are trained
- Uses causal queries for autoregressive reordering
- Causal queries have higher learning rate (10x)

Data Flow:
Input Image [B, 3, 448, 448]
├─→ Semantic Branch (frozen) → Semantic Tokens
└─→ Geometric Branch (frozen) → Geometric Features
       └─→ Gate Generator (frozen) → Gate Scores

Cross-Attention (frozen): Fused Tokens
→ Causal Encoder (with causal queries) → Encoded Tokens
   - Fused tokens + causal queries concatenated
   - Causal mask prevents attending to future tokens
→ LLM Decoder (with teacher forcing)
→ Cross-Entropy Loss
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
    parser = argparse.ArgumentParser(description='Train Stage 3: Causal Flow Reordering')
    parser.add_argument('--config', type=str, default='configs/train_stage3.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, tokenizer, config, device, scaler=None):
    """
    Train the model for one epoch.
    
    Args:
        model: CausalGauge model
        dataloader: Training data loader
        optimizer: Optimizer for causal encoder and LLM decoder
        tokenizer: Qwen2 tokenizer
        config: Configuration dictionary
        device: Device to use ('cuda' or 'cpu')
        scaler: Gradient scaler for mixed precision training
    
    Returns:
        metrics: Dictionary with average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch in progress_bar:
        # Get batch data
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
        labels = text_inputs.clone()
        labels[attention_mask == 0] = -100
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if available
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # Forward pass through model (stage3 mode, with causal queries)
                outputs = model(
                    images,
                    text_inputs=text_inputs,
                    labels=labels,
                    stage='stage3',
                    use_causal_queries=True
                )
                
                # Get cross-entropy loss from LLM decoder
                loss = outputs['loss']
            
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
                stage='stage3',
                use_causal_queries=True
            )
            
            loss = outputs['loss']
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}'
        })
    
    # Compute average loss
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches
    }


def validate(model, dataloader, tokenizer, config, device):
    """
    Validate the model on the validation set.
    
    Generates readings autoregressively using causal queries.
    
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
            
            # Generate readings autoregressively with causal queries
            preds = model.generate(
                images,
                max_length=50,
                use_causal_queries=True
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
    Main training function for Stage 3.
    
    Steps:
    1. Parse arguments and load config
    2. Set random seed for reproducibility
    3. Create datasets and dataloaders
    4. Initialize model and load from Stage 2 checkpoint
    5. Freeze vision branches
    6. Set up optimizer for causal encoder and LLM decoder
    7. Train and validate for multiple epochs
    8. Save best model and checkpoints
    """
    args = parse_args()
    config = load_config(args.config)
    set_seed(args.seed)
    
    device = args.device
    print(f'Using device: {device}')
    
    # Create checkpoint directory
    checkpoint_dir = config['train']['stage3']['checkpoint_dir']
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
    
    # Load checkpoint from Stage 2
    stage2_checkpoint = config['model'].get('stage2_checkpoint')
    if stage2_checkpoint and os.path.exists(stage2_checkpoint):
        print(f'Loading stage2 checkpoint from: {stage2_checkpoint}')
        model, _, _, _ = load_checkpoint(model, stage2_checkpoint, device=device)
    
    # Freeze vision branches if specified
    if config['train']['stage3']['freeze_vision']:
        print('Freezing vision branches...')
        for param in model.semantic_branch.parameters():
            param.requires_grad = False
        for param in model.geometric_branch.parameters():
            param.requires_grad = False
        for param in model.gate_generator.parameters():
            param.requires_grad = False
        for param in model.cross_attention.parameters():
            param.requires_grad = False
    
    # Get tokenizer from model
    tokenizer = model.llm_decoder.tokenizer
    
    # Set up optimizer: only train causal encoder and LLM decoder
    # Causal queries get 10x higher LR for faster adaptation
    optimizer_params = [
        {'params': model.causal_encoder.parameters(), 'lr': config['train']['stage3']['lr']},
        {'params': model.llm_decoder.parameters(), 'lr': config['train']['stage3']['lr']},
        {'params': model.causal_encoder.causal_queries, 'lr': config['train']['stage3']['lr'] * 10}
    ]
    
    # Filter out parameters that don't require gradients
    optimizer = AdamW([p for p in optimizer_params if p['params'][0].requires_grad], weight_decay=1e-4)
    
    # Initialize learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['train']['stage3']['epochs'],
        eta_min=config['train']['stage3']['lr_decay']
    )
    
    # Initialize gradient scaler
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    best_val_error = float('inf')
    
    # Training loop
    print('Starting training...')
    for epoch in range(config['train']['stage3']['epochs']):
        print(f'\nEpoch {epoch + 1}/{config["train"]["stage3"]["epochs"]}')
        
        # Train one epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, tokenizer, config, device, scaler
        )
        print(f'Train - Loss: {train_metrics["loss"]:.4f}')
        
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
