"""
FloodSense Inference Script

Usage:
    python inference.py --config configs/s1gfloods.yaml --checkpoint weights/s1gfloods/floodsense/best.pth --random-samples 5
    python inference.py --config configs/sen1floods11.yaml --checkpoint weights/sen1floods11/floodsense/best.pth --random-samples 5
    python inference.py --config configs/ombrias1.yaml --checkpoint weights/ombrias1/floodsense/best.pth --random-samples 5
    python inference.py --checkpoint outputs/best.pth --pre image_pre.png --post image_post.png
"""

import argparse
import os
import sys
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from src.model import FloodSense, FloodSenseModelConfig
from src.config import load_config
from data import (
    Sen1Floods11Dataset, S1GFloodsDataset, OmbriaS1Dataset,
    infer_input_channels,
)


def load_image(path: str, image_size: int = 256, dataset_type: str = 'sen1floods11') -> torch.Tensor:
    if dataset_type == 'sen1floods11':
        import tifffile
        img = tifffile.imread(path)
        if img.ndim == 3 and img.shape[-1] == 2:
            img = img.transpose(2, 0, 1)
        img = np.nan_to_num(img, nan=0.0)
        img = np.clip(img, -30, 0)
        img = (img + 30) / 30
        img = img.astype(np.float32)
        vv, vh = img[0:1], img[1:2]
        img = np.concatenate([vv, vh, (vv + vh) / 2.0], axis=0)
    else:
        gray = np.array(Image.open(path).convert('L'))
        gray = gray.astype(np.float32) / 255.0
        img = np.stack([gray, gray, gray], axis=0)

    if img.shape[1] != image_size or img.shape[2] != image_size:
        from skimage.transform import resize
        img = resize(img, (img.shape[0], image_size, image_size), preserve_range=True).astype(np.float32)

    return torch.from_numpy(img).unsqueeze(0)


def visualize_result(img_pre, img_post, prediction, probability, save_path=None, ground_truth=None):
    ncols = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 12))

    def _to_display(channel):
        mn, mx = channel.min(), channel.max()
        if mx > mn:
            return ((channel - mn) / (mx - mn)).astype(np.float32)
        return np.zeros_like(channel, dtype=np.float32)

    if img_pre.shape[0] in (3, 6):
        img_pre_vis = np.stack([_to_display(img_pre[0])] * 3, axis=-1)
        img_post_vis = np.stack([_to_display(img_post[0])] * 3, axis=-1)
    else:
        img_pre_vis = np.stack([_to_display(img_pre.transpose(1, 2, 0)[:, :, 0])] * 3, axis=-1)
        img_post_vis = np.stack([_to_display(img_post.transpose(1, 2, 0)[:, :, 0])] * 3, axis=-1)

    axes[0, 0].imshow(img_pre_vis)
    axes[0, 0].set_title('Pre-Flood')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_post_vis)
    axes[0, 1].set_title('Post-Flood')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(prediction, cmap='RdYlBu_r', vmin=0, vmax=1)
    axes[1, 0].set_title('Flood Prediction')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(probability, cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title('Flood Probability')
    axes[1, 1].axis('off')

    if ground_truth is not None:
        axes[0, 2].imshow(ground_truth, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[0, 2].set_title('Ground Truth')
        axes[0, 2].axis('off')
        diff = np.abs(prediction.astype(float) - ground_truth.astype(float))
        axes[1, 2].imshow(diff, cmap='Reds', vmin=0, vmax=1)
        axes[1, 2].set_title('Error')
        axes[1, 2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def inference_from_dataset(model, dataset, num_samples, device, output_dir, seed=None):
    if seed is not None:
        random.seed(seed)

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    model.eval()
    results = []

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        img_pre = sample['pre'].unsqueeze(0).to(device)
        img_post = sample['post'].unsqueeze(0).to(device)
        label = sample['label'].numpy()
        label_vis = label.copy()
        label_vis[label_vis == 255] = 0

        with torch.no_grad():
            outputs = model(img_pre, img_post)
            probs = F.softmax(outputs['logits'], dim=1)
            preds = probs.argmax(dim=1)

        prediction = preds[0].cpu().numpy()
        probability = probs[0, 1].cpu().numpy()

        valid_mask = label != 255
        if valid_mask.any():
            accuracy = (prediction[valid_mask] == label[valid_mask]).sum() / valid_mask.sum()
            pred_flood, gt_flood = prediction == 1, label == 1
            intersection = (pred_flood & gt_flood & valid_mask).sum()
            union = ((pred_flood | gt_flood) & valid_mask).sum()
            iou = intersection / (union + 1e-6)
        else:
            accuracy, iou = 0.0, 0.0

        results.append({'name': sample['name'], 'accuracy': accuracy, 'iou': iou})
        print(f"  [{i+1}/{num_samples}] {sample['name']}: Acc={accuracy:.4f}, IoU={iou:.4f}")

        save_path = os.path.join(output_dir, f'{sample["name"]}_result.png')
        visualize_result(img_pre[0].cpu().numpy(), img_post[0].cpu().numpy(), prediction, probability, save_path, label_vis)

    print(f"\nSummary: Avg Accuracy={np.mean([r['accuracy'] for r in results]):.4f}, Avg IoU={np.mean([r['iou'] for r in results]):.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description='FloodSense Inference')
    parser.add_argument('--config', type=str)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--pre', type=str)
    parser.add_argument('--post', type=str)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--image-size', type=int, default=None)
    parser.add_argument('--encoder', type=str, default=None)
    parser.add_argument('--in-channels', type=int, default=None)
    parser.add_argument('--dataset-type', type=str, default='sen1floods11', choices=['sen1floods11', 's1gfloods', 'ombrias1'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--random-samples', type=int, default=0)
    parser.add_argument('--data-root', type=str)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    config = load_config(args.config) if args.config else None
    if config:
        image_size = args.image_size or config.dataset.image_size
        encoder = args.encoder or config.model.encoder
        preprocessing_config = config.preprocessing.__dict__
        in_channels = args.in_channels or infer_input_channels(
            config.dataset.name, config.dataset.in_channels, preprocessing_config
        )
        data_root = args.data_root or config.dataset.root
        dataset_name = config.dataset.name.lower()
    else:
        image_size = args.image_size or 256
        encoder = args.encoder or 'resnet18'
        preprocessing_config = None
        in_channels = args.in_channels or 3
        data_root = args.data_root
        dataset_name = args.dataset_type

    model_config = FloodSenseModelConfig(in_channels=in_channels, num_classes=2, encoder=encoder, pretrained=False, img_size=image_size)
    model = FloodSense(model_config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if args.output_dir is None:
        output_dir = os.path.join('outputs', f'inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    else:
        output_dir = args.output_dir if args.output_dir.startswith('outputs') or os.path.isabs(args.output_dir) else os.path.join('outputs', args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.random_samples > 0:
        dataset_kwargs = dict(root=data_root, split=args.split, image_size=image_size,
                              preprocessing_config=preprocessing_config)
        if dataset_name == 'sen1floods11':
            dataset = Sen1Floods11Dataset(**dataset_kwargs)
        elif dataset_name == 'ombrias1':
            dataset = OmbriaS1Dataset(**dataset_kwargs)
        else:
            dataset = S1GFloodsDataset(**dataset_kwargs)
        inference_from_dataset(model, dataset, args.random_samples, device, output_dir, args.seed)
    elif args.pre and args.post:
        img_pre = load_image(args.pre, image_size, dataset_name).to(device)
        img_post = load_image(args.post, image_size, dataset_name).to(device)
        with torch.no_grad():
            outputs = model(img_pre, img_post)
            probs = F.softmax(outputs['logits'], dim=1)
            preds = probs.argmax(dim=1)
        save_path = os.path.join(output_dir, 'result.png')
        visualize_result(img_pre[0].cpu().numpy(), img_post[0].cpu().numpy(), preds[0].cpu().numpy(), probs[0, 1].cpu().numpy(), save_path)
    else:
        print("Provide --config with --random-samples, or --pre/--post")


if __name__ == '__main__':
    main()
