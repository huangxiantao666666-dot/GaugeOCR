"""
Stage 1 Training Script: DeepLabV3+ Segmentation Training.

This script trains DeepLabV3+ for semantic segmentation of gauges:
- Class 0: Background
- Class 1: Pointer
- Class 2: Scale marks

Training Strategy:
- Stage 1: Freeze backbone, train only the segmentation head (higher LR)
- Stage 2: Unfreeze all layers, fine-tune with lower LR

Data Flow:
Input Image [B, 3, H, W]
→ DeepLabV3+ (ResNet50 backbone)
→ Segmentation Head [B, 3, H, W] (3 classes)
→ CrossEntropy + Dice Loss
"""

import os
import sys
import argparse
import yaml
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.deeplabv3plus import create_deeplabv3plus


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GaugeSegmentationDataset(Dataset):
    """
    Dataset for gauge segmentation task.
    
    The annotation images contain values 0, 1, 2 representing:
    - 0: Background
    - 1: Pointer
    - 2: Scale marks
    
    These are converted to class indices 0, 1, 2 for CrossEntropy loss.
    """
    
    def __init__(self, images_dir, annotations_dir, transform=None, image_size=448):
        """
        Args:
            images_dir: Path to images folder
            annotations_dir: Path to annotations folder
            transform: Optional transform for images
            image_size: Target image size
        """
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Find all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.image_files.extend(list(self.images_dir.glob(ext)))
        
        # Sort for consistency
        self.image_files = sorted(self.image_files)
        print(f"Found {len(self.image_files)} images in {images_dir}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image path
        img_path = self.image_files[idx]
        img_name = img_path.stem
        
        # Try to find corresponding annotation (try both .png and .jpg)
        ann_path_png = self.annotations_dir / f"{img_name}.png"
        ann_path_jpg = self.annotations_dir / f"{img_name}.jpg"
        
        if ann_path_png.exists():
            ann_path = ann_path_png
        elif ann_path_jpg.exists():
            ann_path = ann_path_jpg
        else:
            # Try without extension
            ann_path = self.annotations_dir / img_name
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load annotation - annotations are single-channel images with values 0, 1, 2
        if ann_path.exists():
            annotation = Image.open(ann_path)
            # Convert to numpy and ensure it's grayscale
            annotation = np.array(annotation)
            if len(annotation.shape) > 2:
                annotation = annotation[:, :, 0]  # Take first channel if RGB
            
            # Ensure annotation values are valid (0, 1, 2)
            # Clip to valid range and convert to uint8
            annotation = np.clip(annotation, 0, 2).astype(np.uint8)
        else:
            # If no annotation found, create empty mask (all background)
            annotation = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        
        # Resize
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        annotation = Image.fromarray(annotation).resize((self.image_size, self.image_size), Image.NEAREST)
        annotation = np.array(annotation)
        
        # Apply transforms to image
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # Convert annotation to tensor (long type for CrossEntropyLoss)
        annotation = torch.from_numpy(annotation).long()
        
        return image, annotation


def get_transforms(is_train=True, image_size=448):
    """Get transforms for training/validation."""
    if is_train:
        return transforms.Compose([

            transforms.Resize((image_size, image_size)),
            # transforms.RandomHorizontalFlip(p=0.5), gauge data can not be flipped
            transforms.RandomRotation(degrees=15),
            
            transforms.ToTensor(),
            
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([

            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Dice Loss measures the overlap between predicted and ground truth masks.
    """
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits [B, C, H, W]
            target: Ground truth labels [B, H, W]
        
        Returns:
            Dice loss value
        """
        # Get predictions
        pred = F.softmax(pred, dim=1)
        
        # Convert target to one-hot
        B, C, H, W = pred.shape
        target_one_hot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()
        
        # Calculate Dice coefficient for each class
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return average Dice loss across classes
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for segmentation tasks.
    
    Focal Loss adds a modulating factor (1 - pt)^gamma to focus on hard examples
    and reduce the importance of easy examples.
    
    Args:
        alpha: Class weights (can be tensor or list)
        gamma: Focusing parameter (default: 2.0)
        ignore_index: Index to ignore in loss calculation
        reduction: 'mean' or 'sum'
    
    Reference:
        Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
        Focal Loss for Dense Object Detection. ICCV.
    """
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # Handle alpha (class weights)
        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            else:
                self.alpha = None
        else:
            self.alpha = None
    
    def _get_alpha(self, device):
        """Get alpha tensor on correct device"""
        if self.alpha is not None:
            return self.alpha.to(device)
        return None
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits [B, C, H, W]
            target: Ground truth labels [B, H, W]
        
        Returns:
            Focal loss value
        """
        device = pred.device
        
        # Ensure target is on correct device
        if target.device != device:
            target = target.to(device)
        
        # Get alpha on correct device
        alpha = self._get_alpha(device)
        
        # Compute log softmax
        log_softmax = F.log_softmax(pred, dim=1)
        
        # Compute softmax probabilities
        probs = torch.exp(log_softmax)
        
        # Create one-hot encoding of target
        B, C, H, W = pred.shape
        target_one_hot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()
        
        # Get log probabilities for the target classes
        # log_softmax: [B, C, H, W], target_one_hot: [B, C, H, W]
        log_pt = (log_softmax * target_one_hot).sum(dim=1)  # [B, H, W]
        pt = torch.exp(log_pt)  # [B, H, W]
        
        # Apply focal loss formula: -alpha * (1 - pt)^gamma * log(pt)
        # For multi-class, we need to apply alpha per class
        if alpha is not None:
            # Get alpha weights for each target class
            alpha_t = torch.gather(alpha.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W),
                                   1, target.unsqueeze(1)).squeeze(1)  # [B, H, W]
            focal_weight = alpha_t * (1 - pt) ** self.gamma
        else:
            focal_weight = (1 - pt) ** self.gamma
        
        # Calculate loss
        loss = -focal_weight * log_pt
        
        # Apply ignore index mask
        if self.ignore_index is not None:
            ignore_mask = (target != self.ignore_index).float()
            loss = loss * ignore_mask
        
        # Reduction
        if self.reduction == 'mean':
            return loss.sum() / (ignore_mask.sum() + 1e-6) if self.ignore_index is not None else loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class InstanceSeparationLoss(nn.Module):
    """
    鼓励分离目标的损失函数
    惩罚将不同刻度连接在一起的预测
    """
    def __init__(self, min_distance=5, penalty_weight=1.0, scale_class_idx=2):
        super().__init__()
        self.min_distance = min_distance
        self.penalty_weight = penalty_weight
        self.scale_class_idx = scale_class_idx  # 刻度类的索引
        
    def get_connected_components(self, pred_mask):
        """
        获取连通域
        Args:
            pred_mask: [B, H, W] 二值掩码 (0或1)
        Returns:
            components: [B, H, W] 连通域标签
        """
        batch_size = pred_mask.shape[0]
        device = pred_mask.device
        components_list = []
        
        for b in range(batch_size):
            # 转换为numpy进行连通域分析
            mask = pred_mask[b].cpu().numpy().astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
            # 转回tensor并保持设备
            components_list.append(torch.from_numpy(labels).to(device))
        
        return torch.stack(components_list)
    
    def compute_separation_penalty(self, pred_binary, target):
        """
        计算分离惩罚
        Args:
            pred_binary: [B, H, W] 预测的二值掩码 (只包含刻度类)
            target: [B, H, W] ground truth标签
        Returns:
            penalty: 分离惩罚值
        """
        B, H, W = target.shape
        device = target.device
        
        # 获取目标中刻度的连通域
        target_scale = (target == self.scale_class_idx).float()
        if target_scale.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        target_components = self.get_connected_components(target_scale)
        
        # 获取预测中刻度的连通域
        if pred_binary.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        pred_components = self.get_connected_components(pred_binary)
        
        penalty = 0.0
        valid_samples = 0
        
        for b in range(B):
            # 找到目标中不同的刻度实例
            unique_targets = torch.unique(target_components[b])
            unique_targets = unique_targets[unique_targets > 0]  # 移除背景
            
            if len(unique_targets) == 0:
                continue
            
            valid_samples += 1
            
            # 惩罚1: 多个预测连通域覆盖同一个目标（预测断裂）
            for t_id in unique_targets:
                target_mask = (target_components[b] == t_id).float()
                
                # 找到预测中覆盖该目标区域的连通域
                overlap = (pred_binary[b] * target_mask).float()
                if overlap.sum() > 0:
                    # 获取覆盖的预测连通域ID
                    pred_ids = pred_components[b][overlap > 0]
                    if len(pred_ids) > 0:
                        pred_ids = torch.unique(pred_ids)
                        pred_ids = pred_ids[pred_ids > 0]
                        
                        # 如果多个预测连通域覆盖同一个目标，说明预测有断裂
                        if len(pred_ids) > 1:
                            penalty += 1.0
            
            # 惩罚2: 一个预测连通域覆盖多个目标实例（预测粘连）
            unique_preds = torch.unique(pred_components[b])
            unique_preds = unique_preds[unique_preds > 0]
            
            for p_id in unique_preds:
                pred_region = (pred_components[b] == p_id).float()
                
                # 检查这个预测区域覆盖了多少个目标实例
                target_ids = target_components[b][pred_region > 0]
                if len(target_ids) > 0:
                    target_ids = torch.unique(target_ids)
                    target_ids = target_ids[target_ids > 0]
                    
                    # 如果一个预测连通域覆盖多个目标实例，给予惩罚
                    if len(target_ids) > 1:
                        penalty += 1.0
        
        if valid_samples > 0:
            penalty = penalty / (valid_samples * 2)  # 归一化
        else:
            penalty = torch.tensor(0.0, device=device)
        
        return penalty
    
    def forward(self, pred, target):
        """
        计算分离损失
        Args:
            pred: 预测logits [B, C, H, W]
            target: ground truth [B, H, W]
        Returns:
            separation_loss: 分离损失值
        """
        device = pred.device
        
        # 确保target在正确的设备上
        if target.device != device:
            target = target.to(device)
        
        # 获取刻度类的概率
        pred_probs = F.softmax(pred, dim=1)
        pred_scale = pred_probs[:, self.scale_class_idx:self.scale_class_idx+1, :, :]  # [B, 1, H, W]
        
        # 二值化预测（阈值0.5）
        pred_binary = (pred_scale > 0.5).float().squeeze(1)  # [B, H, W]
        
        # 计算分离惩罚
        separation_penalty = self.compute_separation_penalty(pred_binary, target)
        
        return self.penalty_weight * separation_penalty


class CombinedLoss(nn.Module):
    """
    Combined CrossEntropy + Focal + Dice + Instance Separation Loss with class weights.
    
    Class weights allow emphasizing different classes during training:
    - Pointer (class 1): highest weight (10.0) - most important for gauge reading
    - Scale (class 2): medium weight (3.0) - important for scale detection
    - Background (class 0): lowest weight (0.05) - least important
    
    Focal Loss focuses on hard examples.
    Instance Separation Loss helps separate connected scale marks.
    """
    
    def __init__(self, dice_weight=0.5, focal_weight=0.0, separation_weight=0.3,
                 ignore_index=255, class_weights=None, separation_config=None,
                 focal_gamma=2.0):
        """
        Args:
            dice_weight: Weight for Dice loss
            focal_weight: Weight for Focal loss (0.0 to disable)
            separation_weight: Weight for instance separation loss
            ignore_index: Index to ignore in cross entropy loss
            class_weights: List or tensor of class weights [w_bg, w_pointer, w_scale]
            separation_config: Dict with config for InstanceSeparationLoss
            focal_gamma: Focusing parameter for Focal Loss (default: 2.0)
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.separation_weight = separation_weight
        self.ignore_index = ignore_index
        self.class_weights = class_weights  # Store as list or tensor
        
        # Initialize losses
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma, 
                                    ignore_index=ignore_index)
        
        # Initialize instance separation loss with config
        if separation_config is None:
            separation_config = {
                'min_distance': 5,
                'penalty_weight': 1.0,
                'scale_class_idx': 2
            }
        self.separation_loss = InstanceSeparationLoss(**separation_config)
    
    def _get_class_weights(self, device):
        """Get class weights tensor on the correct device"""
        if self.class_weights is not None:
            if isinstance(self.class_weights, list):
                return torch.tensor(self.class_weights, dtype=torch.float32, device=device)
            elif isinstance(self.class_weights, torch.Tensor):
                return self.class_weights.to(device)
            else:
                return None
        return None
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits [B, C, H, W]
            target: Ground truth labels [B, H, W]
        
        Returns:
            Combined loss value
        """
        # Get device from pred
        device = pred.device
        
        # Move target to same device if needed
        if target.device != device:
            target = target.to(device)
        
        # Get class weights on correct device
        class_weights = self._get_class_weights(device)
        
        # Cross Entropy Loss
        ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=self.ignore_index)
        ce = ce_loss_fn(pred, target)
        
        # Focal Loss
        focal = self.focal_loss(pred, target)
        
        # Dice Loss
        dice = self.dice_loss(pred, target)
        
        # Calculate base weight (remaining weight after CE, Focal, Dice, and Separation)
        # Note: CE weight is automatically adjusted
        remaining_weight = 1.0 - self.focal_weight - self.dice_weight - self.separation_weight
        
        # Combined loss: CE + Focal + Dice
        total_loss = ce * remaining_weight + focal * self.focal_weight + dice * self.dice_weight
        
        # Instance Separation Loss (only on scale class)
        if self.separation_weight > 0:
            separation = self.separation_loss(pred, target)
            total_loss = total_loss + separation * self.separation_weight
        
        return total_loss


def create_model(num_classes=3, pretrained_path=None):
    """
    Create DeepLabV3+ model with custom segmentation head.
    
    Uses HuggingFace-style API that automatically:
    1. Creates model with 21 classes (for pretrained weights)
    2. Loads pretrained weights (PASCAL VOC)
    3. Replaces classification head with new head for num_classes
    
    Args:
        num_classes: Number of segmentation classes (e.g., 3 for gauge segmentation)
        pretrained_path: Optional path to pretrained weights
    
    Returns:
        Model with custom classification head
    """
    # Use the new HuggingFace-style API
    # This automatically handles: 21-class model -> load weights -> replace head
    model = create_deeplabv3plus(
        backbone='resnet50',
        num_classes=num_classes,
        pretrained=True,
        pretrained_path=pretrained_path
    )
    
    return model


def freeze_backbone(model):
    """Freeze backbone and decoder layers, only train classification head (1x1 conv)."""
    trainable_names = []
    frozen_names = []
    
    for name, param in model.named_parameters():
        if 'classification_head' in name:
            param.requires_grad = True
            trainable_names.append(name)
        else:
            param.requires_grad = False
            frozen_names.append(name)
    
    print(f"Backbone frozen. Only classification_head will be trained.")
    print(f"Frozen layers: {len(frozen_names)}, Trainable layers: {len(trainable_names)}")
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def unfreeze_all(model):
    """Unfreeze all layers for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    print("All layers unfrozen. Full model will be fine-tuned.")
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def compute_metrics(pred, target, num_classes=3):
    """
    Compute segmentation metrics.
    
    Returns:
        IoU for each class and mean IoU
    """
    # Get predictions
    pred = pred.argmax(dim=1)  # [B, H, W]
    
    # Calculate IoU for each class
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            iou = 1.0 if target_cls.sum() == 0 else 0.0
        else:
            iou = (intersection / union).item()
        
        ious.append(iou)
    
    return ious, np.mean(ious)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_ious = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for images, annotations in pbar:
        images = images.to(device)
        annotations = annotations.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, annotations)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, mean_iou = compute_metrics(outputs.detach(), annotations)
        total_ious.append(mean_iou)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mIoU': f'{mean_iou:.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = np.mean(total_ious)
    return avg_loss, avg_iou


@torch.no_grad()
def validate(model, dataloader, criterion, device, num_classes=3):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_ious = []
    
    pbar = tqdm(dataloader, desc="Validation")
    for images, annotations in pbar:
        images = images.to(device)
        annotations = annotations.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, annotations)
        total_loss += loss.item()
        
        # Metrics
        ious, mean_iou = compute_metrics(outputs, annotations, num_classes)
        all_ious.append(ious)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mIoU': f'{mean_iou:.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    avg_ious = np.mean(all_ious, axis=0)
    mean_iou = np.mean(avg_ious)
    
    return avg_loss, avg_ious, mean_iou


def plot_training_curves(train_losses, val_losses, train_ious, val_ious, save_path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # IoU plot
    axes[1].plot(epochs, train_ious, 'b-', label='Train mIoU')
    axes[1].plot(epochs, val_ious, 'r-', label='Val mIoU')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mIoU')
    axes[1].set_title('Training and Validation mIoU')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


@torch.no_grad()
def visualize_results(model, dataloader, device, num_samples=5, save_dir="outputs/segmentation"):
    """Visualize segmentation results on validation set."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # Get a batch
    images, annotations = next(iter(dataloader))
    images = images.to(device)
    
    # Predict
    outputs = model(images)
    preds = outputs.argmax(dim=1)
    
    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    # Color map for classes: Background=gray, Pointer=red, Scale=green
    class_colors = [
        [128, 128, 128],  # 0: Background (gray)
        [255, 0, 0],      # 1: Pointer (red)
        [0, 255, 0],      # 2: Scale (green)
    ]
    # 关键修复：将 class_colors 移到和模型相同的设备上
    class_colors = torch.tensor(class_colors, dtype=torch.float32).to(device) / 255.0
    # Expand to [1, num_classes, 3] for broadcasting
    class_colors = class_colors.unsqueeze(0)  # [1, 3, 3]
    
    # Create visualization for each sample
    for i in range(min(num_samples, images.shape[0])):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        img = images_denorm[i].cpu().permute(1, 2, 0).numpy()
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth - convert class indices to color mask
        gt = annotations[i].cpu().numpy()  # [H, W]
        # 使用 CPU 上的 class_colors 进行可视化
        gt_tensor = torch.from_numpy(gt).long()
        gt_one_hot = F.one_hot(gt_tensor, num_classes=3).float()  # [H, W, 3]
        # 使用 CPU 上的 class_colors
        gt_colored = torch.matmul(gt_one_hot, class_colors.squeeze(0).cpu()).numpy()  # [H, W, 3]
        axes[1].imshow(gt_colored.astype(np.uint8))
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction - convert class indices to color mask
        pred = preds[i].cpu().numpy()  # [H, W]
        # 使用 CPU 上的 class_colors 进行可视化
        pred_tensor = torch.from_numpy(pred).long()
        pred_one_hot = F.one_hot(pred_tensor, num_classes=3).float()  # [H, W, 3]
        # 使用 CPU 上的 class_colors
        pred_colored = torch.matmul(pred_one_hot, class_colors.squeeze(0).cpu()).numpy()  # [H, W, 3]
        axes[2].imshow(pred_colored.astype(np.uint8))
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{i+1}.png'), dpi=150)
        plt.close()
    
    print(f"Visualization saved to {save_dir}")


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert numeric config values to proper types
    for stage in ['training_stage1', 'training_stage2']:
        if stage in config:
            for key in ['num_epochs', 'batch_size', 'save_interval', 'eval_interval', 'early_stopping_patience']:
                if key in config[stage]:
                    config[stage][key] = int(config[stage][key])
            for key in ['learning_rate', 'weight_decay', 'eta_min']:
                if key in config[stage]:
                    config[stage][key] = float(config[stage][key])
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    # Datasets
    train_dataset = GaugeSegmentationDataset(
        images_dir=config['dataset']['images_train'],
        annotations_dir=config['dataset']['annotations_train'],
        transform=get_transforms(is_train=True, image_size=config['input']['image_size'][0]),
        image_size=config['input']['image_size'][0]
    )
    
    val_dataset = GaugeSegmentationDataset(
        images_dir=config['dataset']['images_val'],
        annotations_dir=config['dataset']['annotations_val'],
        transform=get_transforms(is_train=False, image_size=config['input']['image_size'][0]),
        image_size=config['input']['image_size'][0]
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training_stage1']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training_stage1']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Model
    model = create_model(
        num_classes=config['model']['num_classes'],
        pretrained_path=config['model'].get('pretrained_path')
    )
    model = model.to(device)
    
    # Loss for Stage 1: CrossEntropy + Dice (no separation loss, no focal loss)
    criterion_stage1 = CombinedLoss(
        dice_weight=config['loss_stage1']['dice_weight'],
        focal_weight=config['loss_stage1'].get('focal_weight', 0.0),
        separation_weight=config['loss_stage1'].get('separation_weight', 0.0),
        ignore_index=config['loss_stage1'].get('ignore_index', 255),
        class_weights=config['loss_stage1'].get('class_weights'),
        separation_config=config['loss_stage1'].get('separation_config'),
        focal_gamma=config['loss_stage1'].get('focal_gamma', 2.0)
    )
    
    # Loss for Stage 2: CrossEntropy + Focal + Dice + Instance Separation
    criterion_stage2 = CombinedLoss(
        dice_weight=config['loss_stage2']['dice_weight'],
        focal_weight=config['loss_stage2'].get('focal_weight', 0.2),
        separation_weight=config['loss_stage2'].get('separation_weight', 0.3),
        ignore_index=config['loss_stage2'].get('ignore_index', 255),
        class_weights=config['loss_stage2'].get('class_weights'),
        separation_config=config['loss_stage2'].get('separation_config'),
        focal_gamma=config['loss_stage2'].get('focal_gamma', 2.0)
    )
    
    # Check if skip stage 1
    if args.skip_stage1:
        print("\n" + "="*60)
        print("SKIPPING STAGE 1 (Classification Head Training)")
        print("="*60)
        print("Model will start with random initialization or resume from checkpoint.")
        print("Proceeding directly to Stage 2 (Fine-tuning all layers).")
        print("Using Combined Loss with Instance Separation Loss.")
    else:
        # ===== STAGE 1: Freeze backbone, train head =====
        print("\n" + "="*60)
        print("STAGE 1: Training with frozen backbone")
        print("="*60)
        print("Using Combined Loss (CrossEntropy + Dice, no Instance Separation Loss).")
        
        freeze_backbone(model)
        
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['training_stage1']['learning_rate'],
            weight_decay=config['training_stage1']['weight_decay']
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['training_stage1']['num_epochs'],
            eta_min=config['training_stage1']['eta_min']
        )
        
        # Training tracking
        stage1_train_losses = []
        stage1_val_losses = []
        stage1_train_ious = []
        stage1_val_ious = []
        best_val_iou = 0
        best_epoch = 0
        
        for epoch in range(1, config['training_stage1']['num_epochs'] + 1):
            print(f"\n--- Stage 1, Epoch {epoch}/{config['training_stage1']['num_epochs']} ---")
            
            # Train
            train_loss, train_iou = train_one_epoch(model, train_loader, criterion_stage1, optimizer, device, epoch)
            
            # Validate
            val_loss, val_ious, val_iou = validate(model, val_loader, criterion_stage1, device, config['model']['num_classes'])
            
            # Update scheduler
            scheduler.step()
            
            # Track metrics
            stage1_train_losses.append(train_loss)
            stage1_val_losses.append(val_loss)
            stage1_train_ious.append(train_iou)
            stage1_val_ious.append(val_iou)
            
            print(f"Train Loss: {train_loss:.4f}, Train mIoU: {train_iou:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val mIoU: {val_iou:.4f}")
            print(f"Per-class IoU: Background={val_ious[0]:.4f}, Pointer={val_ious[1]:.4f}, Scale={val_ious[2]:.4f}")
            
            # Save best model
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                best_epoch = epoch
                best_model_path = os.path.join(config['checkpoint']['save_dir'], 'best_stage1.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_iou': val_iou,
                    'val_loss': val_loss,
                }, best_model_path)
                print(f"Best model saved! Val mIoU: {val_iou:.4f}")
            
            # Save periodic checkpoint
            if epoch % config['training_stage1']['save_interval'] == 0:
                ckpt_path = os.path.join(config['checkpoint']['save_dir'], f'stage1_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_iou': val_iou,
                }, ckpt_path)
        
        print(f"\nStage 1 completed! Best Val mIoU: {best_val_iou:.4f} at epoch {best_epoch}")
        
        # Save Stage 1 results
        plot_training_curves(
            stage1_train_losses, stage1_val_losses, stage1_train_ious, stage1_val_ious,
            os.path.join(config['checkpoint']['save_dir'], 'stage1_training_curves.png')
        )
    
    # ===== STAGE 2: Fine-tune all layers =====
    print("\n" + "="*60)
    print("STAGE 2: Fine-tuning all layers")
    print("="*60)
    
    # Load best model from Stage 1
    best_model_path = os.path.join(config['checkpoint']['save_dir'], 'best_stage1.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from Stage 1")
    
    # Unfreeze all layers
    unfreeze_all(model)
    
    optimizer = AdamW(
        model.parameters(),
        lr=config['training_stage2']['learning_rate'],
        weight_decay=config['training_stage2']['weight_decay']
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training_stage2']['num_epochs'],
        eta_min=config['training_stage2']['eta_min']
    )
    
    # Reset tracking
    stage2_train_losses = []
    stage2_val_losses = []
    stage2_train_ious = []
    stage2_val_ious = []
    best_val_iou_stage2 = 0
    best_epoch_stage2 = 0
    
    for epoch in range(1, config['training_stage2']['num_epochs'] + 1):
        print(f"\n--- Stage 2, Epoch {epoch}/{config['training_stage2']['num_epochs']} ---")
        
        # Train (with Instance Separation Loss)
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion_stage2, optimizer, device, epoch)
        
        # Validate (with Instance Separation Loss)
        val_loss, val_ious, val_iou = validate(model, val_loader, criterion_stage2, device, config['model']['num_classes'])
        
        # Update scheduler
        scheduler.step()
        
        # Track metrics
        stage2_train_losses.append(train_loss)
        stage2_val_losses.append(val_loss)
        stage2_train_ious.append(train_iou)
        stage2_val_ious.append(val_iou)
        
        print(f"Train Loss: {train_loss:.4f}, Train mIoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val mIoU: {val_iou:.4f}")
        print(f"Per-class IoU: Background={val_ious[0]:.4f}, Pointer={val_ious[1]:.4f}, Scale={val_ious[2]:.4f}")
        
        # Save best model
        if val_iou > best_val_iou_stage2:
            best_val_iou_stage2 = val_iou
            best_epoch_stage2 = epoch
            best_model_path_stage2 = os.path.join(config['checkpoint']['save_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss,
                'stage': 'stage2'
            }, best_model_path_stage2)
            print(f"Best model saved! Val mIoU: {val_iou:.4f}")
        
        # Save periodic checkpoint
        if epoch % config['training_stage2']['save_interval'] == 0:
            ckpt_path = os.path.join(config['checkpoint']['save_dir'], f'stage2_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_iou': val_iou,
            }, ckpt_path)
    
    print(f"\nStage 2 completed! Best Val mIoU: {best_val_iou_stage2:.4f} at epoch {best_epoch_stage2}")
    
    # Save Stage 2 results
    plot_training_curves(
        stage2_train_losses, stage2_val_losses, stage2_train_ious, stage2_val_ious,
        os.path.join(config['checkpoint']['save_dir'], 'stage2_training_curves.png')
    )
    
    # Save final model
    final_model_path = os.path.join(config['checkpoint']['save_dir'], config['checkpoint']['final_model_name'])
    torch.save({
        'epoch': config['training_stage2']['num_epochs'],
        'model_state_dict': model.state_dict(),
        'val_iou': best_val_iou_stage2,
        'stage': 'final'
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Visualize results
    print("\nGenerating visualization on validation set...")
    visualize_results(model, val_loader, device, num_samples=5, save_dir=config['checkpoint']['save_dir'])
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    if not args.skip_stage1:
        print(f"Best Stage 1 mIoU: {best_val_iou:.4f} at epoch {best_epoch}")
    print(f"Best Stage 2 mIoU: {best_val_iou_stage2:.4f} at epoch {best_epoch_stage2}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DeepLabV3+ for gauge segmentation')
    parser.add_argument('--config', type=str, default='config/segmentation_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--skip-stage1', action='store_true',
                       help='Skip stage 1 (classification head training) and directly start fine-tuning')
    args = parser.parse_args()
    
    main(args)