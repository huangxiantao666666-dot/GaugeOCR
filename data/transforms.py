import torch
import torchvision.transforms as transforms
import random
import numpy as np
import cv2


class RandomHorizontalFlip:
    """
    Random horizontal flip augmentation.
    Flips the image horizontally with probability p.
    """
    
    def __init__(self, p=0.5):
        """
        Args:
            p: Probability of applying the flip
        """
        self.p = p
    
    def __call__(self, img):
        """
        Apply random horizontal flip.
        
        Args:
            img: Input image (numpy array or PIL Image)
        
        Returns:
            Flipped image (same type as input)
        """
        if random.random() < self.p:
            if isinstance(img, np.ndarray):
                # For numpy array: [H, W, C] -> [H, W, C] (flipped horizontally)
                return img[:, ::-1, :].copy()
            else:
                # For PIL Image
                return transforms.functional.hflip(img)
        return img


class RandomRotation:
    """
    Random rotation augmentation.
    Rotates the image by a random angle within [-max_angle, max_angle].
    """
    
    def __init__(self, max_angle=15):
        """
        Args:
            max_angle: Maximum absolute rotation angle in degrees
        """
        self.max_angle = max_angle
    
    def __call__(self, img):
        """
        Apply random rotation.
        
        Args:
            img: Input image (numpy array or PIL Image)
        
        Returns:
            Rotated image (same type as input)
        """
        angle = random.uniform(-self.max_angle, self.max_angle)
        if isinstance(img, np.ndarray):
            # For numpy array: use OpenCV for rotation
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            # Get rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            # Apply rotation
            return cv2.warpAffine(img, M, (w, h))
        else:
            # For PIL Image
            return transforms.functional.rotate(img, angle)


class ColorJitter:
    """
    Color jitter augmentation.
    Randomly changes brightness, contrast, saturation, and hue.
    """
    
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        """
        Args:
            brightness: Brightness jitter range [0, 1]
            contrast: Contrast jitter range [0, 1]
            saturation: Saturation jitter range [0, 1]
            hue: Hue jitter range [0, 0.5]
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, img):
        """
        Apply color jitter.
        
        Args:
            img: Input image (numpy array or PIL Image)
        
        Returns:
            Color-jittered image (same type as input)
        """
        if isinstance(img, np.ndarray):
            # For numpy array: convert to PIL first
            img = transforms.ToPILImage()(img)
            img = transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue
            )(img)
            return np.array(img)
        return transforms.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue
        )(img)


def get_train_transforms():
    """
    Get training data augmentation transforms.
    
    Data Flow:
    PIL Image / numpy array
    → ToPILImage (if numpy)
    → Random ColorJitter (50% chance)
    → Random Horizontal Flip (50% chance)
    → Random Rotation (±15°)
    → ToTensor ([0, 255] → [0, 1])
    → Normalize (ImageNet stats)
    → Tensor [3, H, W]
    
    Returns:
        transform: Compose of training transforms
    """
    return transforms.Compose([
        # Convert numpy array to PIL Image if needed
        transforms.ToPILImage(),
        # Random color jitter with 50% probability
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5),
        # Random horizontal flip with 50% probability
        transforms.RandomHorizontalFlip(p=0.5),
        # Random rotation within [-15°, 15°]
        transforms.RandomRotation(15),
        # Convert to tensor: [H, W, C] -> [C, H, W], values [0, 255] -> [0, 1]
        transforms.ToTensor(),
        # Normalize using ImageNet statistics
        # mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms():
    """
    Get validation data transforms (no augmentation).
    
    Data Flow:
    PIL Image / numpy array
    → ToPILImage (if numpy)
    → ToTensor ([0, 255] → [0, 1])
    → Normalize (ImageNet stats)
    → Tensor [3, H, W]
    
    Returns:
        transform: Compose of validation transforms
    """
    return transforms.Compose([
        # Convert numpy array to PIL Image if needed
        transforms.ToPILImage(),
        # Convert to tensor: [H, W, C] -> [C, H, W], values [0, 255] -> [0, 1]
        transforms.ToTensor(),
        # Normalize using ImageNet statistics
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
