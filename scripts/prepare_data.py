import os
import argparse
import json
import cv2
import numpy as np
from tqdm import tqdm


def generate_gaussian_heatmap(shape, center, sigma=5):
    """生成高斯热图"""
    h, w = shape
    x = np.arange(0, w, 1, float)
    y = np.arange(0, h, 1, float)
    y, x = np.meshgrid(y, x)
    x0, y0 = center
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return g


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare data for Gauge OCR')
    parser.add_argument('--input_json', type=str, required=True, help='Path to input annotation JSON')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed data')
    parser.add_argument('--image_size', type=int, nargs=2, default=[448, 448], help='Image size (h, w)')
    parser.add_argument('--sigma', type=float, default=5.0, help='Sigma for Gaussian heatmap')
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    heatmap_dir = os.path.join(args.output_dir, 'heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)
    
    print(f'Loading annotations from: {args.input_json}')
    with open(args.input_json, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f'Processing {len(annotations)} samples...')
    
    processed_annotations = []
    
    for idx, ann in enumerate(tqdm(annotations)):
        img_path = ann['image_path']
        keypoints = np.array(ann['keypoints'])
        
        image = cv2.imread(img_path)
        if image is None:
            print(f'Warning: Cannot read image {img_path}, skipping...')
            continue
        
        orig_h, orig_w = image.shape[:2]
        
        image_resized = cv2.resize(image, (args.image_size[1], args.image_size[0]))
        
        keypoints_scaled = keypoints.copy()
        keypoints_scaled[:, 0] = keypoints[:, 0] * args.image_size[1]
        keypoints_scaled[:, 1] = keypoints[:, 1] * args.image_size[0]
        
        heatmaps = np.zeros((len(keypoints), args.image_size[0], args.image_size[1]), dtype=np.float32)
        for i, (x, y) in enumerate(keypoints_scaled):
            if 0 <= x < args.image_size[1] and 0 <= y < args.image_size[0]:
                heatmaps[i] = generate_gaussian_heatmap(args.image_size, (x, y), sigma=args.sigma)
        
        heatmap_path = os.path.join(heatmap_dir, f'heatmap_{idx:06d}.npy')
        np.save(heatmap_path, heatmaps)
        
        processed_ann = {
            'image_path': img_path,
            'heatmap_path': heatmap_path,
            'keypoints': keypoints.tolist(),
            'keypoints_scaled': keypoints_scaled.tolist()
        }
        processed_annotations.append(processed_ann)
    
    output_json = os.path.join(args.output_dir, 'processed_annotations.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(processed_annotations, f, ensure_ascii=False, indent=2)
    
    print(f'\nProcessing complete!')
    print(f'  Processed samples: {len(processed_annotations)}')
    print(f'  Heatmaps saved to: {heatmap_dir}')
    print(f'  Annotations saved to: {output_json}')


if __name__ == '__main__':
    main()
