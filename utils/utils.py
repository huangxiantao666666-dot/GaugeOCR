import yaml
import torch
import random
import numpy as np
import os


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Data Flow:
    config_path (str)
    → Open and read YAML file
    → Parse to Python dictionary
    Output: config dict
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        config: Dictionary containing configuration
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """
    Set random seed for reproducibility across all frameworks.
    
    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CuDNN (deterministic mode)
    
    Args:
        seed: Random seed integer
    """
    # Python random module
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    # PyTorch CUDA (all GPUs)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    # Disable CuDNN benchmark for reproducibility
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """
    Save model checkpoint to disk.
    
    Checkpoint contains:
    - Current epoch
    - Model state dict
    - Optimizer state dict (optional)
    - Metrics dictionary
    
    Data Flow:
    model, optimizer, epoch, metrics
    → Create checkpoint dict
    → Create directory if needed
    → Save with torch.save()
    Output: file written to disk
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer to save (can be None)
        epoch: Current training epoch
        metrics: Dictionary of metrics
        save_path: Path to save checkpoint
    """
    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'metrics': metrics
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save checkpoint
    torch.save(checkpoint, save_path)


def load_checkpoint(model, checkpoint_path, optimizer=None, device='cpu'):
    """
    Load model checkpoint from disk.
    
    Data Flow:
    checkpoint_path (str)
    → torch.load()
    → Load state dict into model
    → (Optional) Load state dict into optimizer
    Output: (model, optimizer, epoch, metrics)
    
    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optimizer to load state into (optional)
        device: Device to map tensors to
    
    Returns:
        tuple: (model, optimizer, epoch, metrics)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state dict if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get epoch and metrics
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    return model, optimizer, epoch, metrics


def get_parameter_names(model, forbidden_layer_types):
    """
    Get parameter names, excluding those in forbidden layer types.
    
    Recursively traverses the model hierarchy to collect parameter names.
    
    Args:
        model: PyTorch model
        forbidden_layer_types: List of layer types to exclude
    
    Returns:
        result: List of parameter names
    """
    result = []
    for name, child in model.named_children():
        # Recursively get parameters from child modules
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add parameters from this module (not from children)
    result += list(model._parameters.keys())
    return result


def get_optimizer_params(model, weight_decay=0.01):
    """
    Get parameter groups for optimizer, separating params with/without weight decay.
    
    Weight decay is NOT applied to:
    - LayerNorm parameters
    - Bias terms
    
    Data Flow:
    model
    → Identify decay/no-decay parameters
    → Create two parameter groups
    Output: optimizer_grouped_parameters list
    
    Args:
        model: PyTorch model
        weight_decay: Weight decay factor to apply
    
    Returns:
        optimizer_grouped_parameters: List of parameter group dicts
    """
    # Get all parameter names excluding LayerNorm
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    # Exclude bias terms
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    
    # Create two parameter groups
    optimizer_grouped_parameters = [
        # Group 1: Parameters with weight decay
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        # Group 2: Parameters without weight decay (LayerNorm, biases)
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    
    return optimizer_grouped_parameters
