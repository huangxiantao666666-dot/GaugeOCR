import os
import argparse
import cv2
import torch
import torchvision.transforms as transforms
import json
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import CausalGauge
from utils import load_config, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Gauge OCR Inference')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_json', type=str, default=None, help='Path to save output JSON')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum generation length')
    return parser.parse_args()


def preprocess_image(image_path, image_size=(448, 448)):
    """预处理图像"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f'Cannot read image: {image_path}')
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size[1], image_size[0]))
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor, image


def main():
    args = parse_args()
    config = load_config(args.config)
    
    device = args.device
    print(f'Using device: {device}')
    
    print('Initializing model...')
    model = CausalGauge(config).to(device)
    
    print(f'Loading checkpoint from: {args.checkpoint}')
    model, _, _, _ = load_checkpoint(model, args.checkpoint, device=device)
    model.eval()
    
    print(f'Processing image: {args.image_path}')
    image_tensor, original_image = preprocess_image(
        args.image_path,
        image_size=config['data']['image_size']
    )
    image_tensor = image_tensor.to(device)
    
    print('Generating reading...')
    with torch.no_grad():
        readings = model.generate(
            image_tensor,
            max_length=args.max_length,
            use_causal_queries=True
        )
    
    reading = readings[0]
    
    result = {
        'image_path': args.image_path,
        'reading': reading,
        'status': 'success'
    }
    
    print(f'\nResult:')
    print(f'  Reading: {reading}')
    
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f'\nSaved result to: {args.output_json}')
    
    return result


if __name__ == '__main__':
    main()
