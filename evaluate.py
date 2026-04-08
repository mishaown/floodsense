"""
FloodSense Evaluation Script

Usage:
    python evaluate.py --config configs/s1gfloods.yaml --checkpoint weights/s1gfloods/floodsense/best.pth
    python evaluate.py --config configs/sen1floods11.yaml --checkpoint weights/sen1floods11/floodsense/best.pth
    python evaluate.py --config configs/ombrias1.yaml --checkpoint weights/ombrias1/floodsense/best.pth
"""

import argparse
import os
import sys
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.model import FloodSense, FloodSenseModelConfig
from src.metrics import FloodMetrics
from data import build_dataloader, infer_input_channels


def evaluate(model, test_loader, device, save_predictions=False, output_dir=None):
    model.eval()
    metrics = FloodMetrics()
    all_predictions, all_names = [], []
    pbar = tqdm(test_loader, desc="Evaluating", dynamic_ncols=True)

    with torch.no_grad():
        for batch in pbar:
            img_pre = batch['pre'].to(device)
            img_post = batch['post'].to(device)
            labels = batch['label'].to(device)
            names = batch['name']

            outputs = model(img_pre, img_post)
            probs = F.softmax(outputs['logits'], dim=1)
            preds = probs.argmax(dim=1)
            metrics.update(preds.cpu(), labels.cpu(), probs.cpu())

            if save_predictions:
                all_predictions.extend(preds.cpu().numpy())
                all_names.extend(names)

    results = metrics.compute_detailed()

    print("\n" + "=" * 60)
    print("FloodSense Evaluation Results")
    print("=" * 60)
    for key in ['F1', 'IoU', 'mIoU', 'Precision', 'Recall', 'OA', 'Kappa', 'MCC', 'AUC-ROC', 'AP']:
        if key in results:
            print(f"{key:<20} {results[key]:>10.4f}")
    print("=" * 60)

    if save_predictions and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.savez(os.path.join(output_dir, 'predictions.npz'), predictions=np.array(all_predictions), names=np.array(all_names))

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate FloodSense')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--save-predictions', action='store_true')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    preprocessing_config = config.preprocessing.__dict__
    effective_in_channels = infer_input_channels(
        config.dataset.name,
        config.dataset.in_channels,
        preprocessing_config
    )

    test_loader = build_dataloader(
        config.dataset.name, config.dataset.root, args.split,
        config.training.batch_size, num_workers=config.training.num_workers,
        image_size=config.dataset.image_size, shuffle=False,
        in_channels=effective_in_channels,
        preprocessing_config=preprocessing_config
    )

    model_config = FloodSenseModelConfig(
        in_channels=effective_in_channels, num_classes=config.dataset.num_classes,
        encoder=config.model.encoder, pretrained=False,
        hidden_dim=config.model.hidden_dim, lstm_hidden=config.model.lstm_hidden,
        lstm_layers=config.model.lstm_layers, use_attention=config.model.use_attention,
        dropout=config.model.dropout, img_size=config.dataset.image_size
    )

    model = FloodSense(model_config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    results = evaluate(model, test_loader, device, args.save_predictions, output_dir)

    results_path = os.path.join(output_dir, f'{args.split}_results.json')
    json_results = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results.items()}
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
