import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch


def visualize_heatmaps(image, heatmaps, save_path=None, figsize=(15, 10)):
    """
    可视化关键点热图
    
    Args:
        image: 原始图像 [H, W, 3]
        heatmaps: 热图 [C, H, W]
        save_path: 保存路径
        figsize: 图像尺寸
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
    
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()
    
    num_heatmaps = heatmaps.shape[0]
    cols = min(3, num_heatmaps + 1)
    rows = (num_heatmaps + cols) // cols
    
    plt.figure(figsize=figsize)
    
    plt.subplot(rows, cols, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    for i in range(num_heatmaps):
        plt.subplot(rows, cols, i + 2)
        hm = heatmaps[i]
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        
        plt.imshow(image, alpha=0.5)
        plt.imshow(hm, cmap='jet', alpha=0.5)
        plt.title(f'Heatmap {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_gate_scores(gate_scores, grid_size=28, save_path=None, figsize=(10, 10)):
    """
    可视化门控分数
    
    Args:
        gate_scores: 门控分数 [784]
        grid_size: 网格尺寸
        save_path: 保存路径
        figsize: 图像尺寸
    """
    if isinstance(gate_scores, torch.Tensor):
        gate_scores = gate_scores.cpu().numpy()
    
    gate_map = gate_scores.reshape(grid_size, grid_size)
    
    plt.figure(figsize=figsize)
    plt.imshow(gate_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Gate Score')
    plt.title('Gate Scores Visualization')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_attention(attention_map, save_path=None, figsize=(10, 10)):
    """
    可视化注意力图
    
    Args:
        attention_map: 注意力图 [H, W]
        save_path: 保存路径
        figsize: 图像尺寸
    """
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.cpu().numpy()
    
    plt.figure(figsize=figsize)
    plt.imshow(attention_map, cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.title('Attention Map')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
