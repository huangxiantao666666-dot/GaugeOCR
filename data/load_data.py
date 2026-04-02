import cv2
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def load_and_display_image(image_path, image_size=448):
    """
    Load an image using PyTorch transforms and display it.
    
    This function demonstrates the complete image loading pipeline:
    1. Load image with PIL
    2. Apply transforms (resize, normalize, convert to tensor)
    3. Display the image using matplotlib
    
    Args:
        image_path: Path to the image file
        image_size: Size to resize the image (default: 448 for GaugeOCR)
    
    Returns:
        image_tensor: The loaded image as a PyTorch tensor [C, H, W]
        image_pil: The original PIL image
    """
    # Load image using PIL
    image_pil = Image.open(image_path).convert('RGB')
    
    # Define transforms
    # This matches the preprocessing used in GaugeOCR
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize to 448x448
        transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
        transforms.Normalize(  # Normalize with ImageNet statistics
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Apply transforms
    image_tensor = transform(image_pil)
    
    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(image_pil)
    plt.title(f'Image: {image_path}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print tensor information
    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"Tensor dtype: {image_tensor.dtype}")
    print(f"Value range: [{image_tensor.min():.4f}, {image_tensor.max():.4f}]")
    
    return image_tensor, image_pil


def load_image_batch(image_paths, image_size=448):
    """
    Load a batch of images and stack them into a single tensor.
    
    Args:
        image_paths: List of image file paths
        image_size: Size to resize images (default: 448)
    
    Returns:
        batch_tensor: Batch of images [B, C, H, W]
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load and transform each image
    images = [transform(Image.open(path).convert('RGB')) for path in image_paths]
    
    # Stack into batch
    batch_tensor = torch.stack(images, dim=0)
    
    print(f"Batch tensor shape: {batch_tensor.shape}")
    
    return batch_tensor


if __name__ == "__main__":
    # Example usage
    import os

    img_path = r"D:\EleEngi\meterRead\GaugeOCR\dataset\meter_seg\meter_seg\images\train\00000000000000000000000000000000.jpg"
    img = Image.open(img_path).convert('RGB')

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f'Image: {img_path}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # # Try to load a sample image
    # sample_image_path = "data/sample_image.jpg"  # Replace with actual path
    
    # if os.path.exists(sample_image_path):
    #     print(f"Loading image: {sample_image_path}")
    #     image_tensor, image_pil = load_and_display_image(sample_image_path)
    # else:
    #     print(f"Sample image not found: {sample_image_path}")
    #     print("Please provide a valid image path to test.")
        
    #     # Create a dummy image for demonstration
    #     print("\nCreating a dummy random image for demonstration...")
    #     dummy_array = torch.rand(448, 448, 3)  # Random RGB image
    #     dummy_pil = Image.fromarray((dummy_array.numpy() * 255).astype('uint8'))
        
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(dummy_pil)
    #     plt.title('Dummy Random Image')
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.show()
