import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import ReadingDataset, get_val_transforms
from models import CausalGauge
from utils import load_config, load_checkpoint, compute_reading_error, compute_accuracy_epsilon


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Gauge OCR Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--ann_file', type=str, required=True, help='Path to annotation file')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--output_json', type=str, default=None, help='Path to save results JSON')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum generation length')
    return parser.parse_args()


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
    
    print('Loading dataset...')
    dataset = ReadingDataset(
        ann_file=args.ann_file,
        image_size=config['data']['image_size'],
        transforms=get_val_transforms()
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    print(f'Evaluating on {len(dataset)} samples...')
    
    all_preds = []
    all_gts = []
    all_results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)
            readings = batch['reading']
            
            preds = model.generate(
                images,
                max_length=args.max_length,
                use_causal_queries=True
            )
            
            all_preds.extend(preds)
            all_gts.extend(readings)
            
            for pred, gt in zip(preds, readings):
                all_results.append({
                    'prediction': pred,
                    'ground_truth': gt
                })
    
    mean_abs_error, mean_rel_error = compute_reading_error(all_preds, all_gts)
    accuracy_005 = compute_accuracy_epsilon(all_preds, all_gts, epsilon=0.05)
    accuracy_010 = compute_accuracy_epsilon(all_preds, all_gts, epsilon=0.10)
    
    print('\n' + '=' * 50)
    print('Evaluation Results:')
    print('=' * 50)
    print(f'Total samples: {len(dataset)}')
    print(f'Mean Absolute Error: {mean_abs_error:.6f}')
    print(f'Mean Relative Error: {mean_rel_error:.6f}')
    print(f'Accuracy @ ε=0.05: {accuracy_005:.4f} ({accuracy_005 * 100:.2f}%)')
    print(f'Accuracy @ ε=0.10: {accuracy_010:.4f} ({accuracy_010 * 100:.2f}%)')
    print('=' * 50)
    
    print('\nSample Predictions:')
    for i in range(min(10, len(all_results))):
        print(f'  [{i+1}] GT: {all_results[i]["ground_truth"]:20} Pred: {all_results[i]["prediction"]}')
    
    results = {
        'total_samples': len(dataset),
        'mean_absolute_error': float(mean_abs_error),
        'mean_relative_error': float(mean_rel_error),
        'accuracy_epsilon_005': float(accuracy_005),
        'accuracy_epsilon_010': float(accuracy_010),
        'per_sample_results': all_results
    }
    
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'\nResults saved to: {args.output_json}')


if __name__ == '__main__':
    main()
