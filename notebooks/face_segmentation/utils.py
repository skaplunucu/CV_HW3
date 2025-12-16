"""
Face Segmentation Utilities

Common functions for data loading, visualization, and evaluation metrics.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from typing import Dict, List, Tuple
import glob
import random
from PIL import Image

np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = Path('../../data')
CELEBAMASK_DIR = DATA_DIR / 'CelebAMask-HQ'
MULTICLASS_DIR = DATA_DIR / 'Multiclass-Face-Segmentation'
RESULTS_DIR = Path('../../Report/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CELEBAMASK_CLASSES = {
    'skin': 1, 'l_brow': 2, 'r_brow': 3, 'l_eye': 4, 'r_eye': 5,
    'eye_g': 6, 'l_ear': 7, 'r_ear': 8, 'ear_r': 9, 'nose': 10,
    'mouth': 11, 'u_lip': 12, 'l_lip': 13, 'neck': 14, 'neck_l': 15,
    'cloth': 16, 'hair': 17, 'hat': 18
}

def combine_celebamask_parts(mask_dir, img_id, img_shape):
    combined_mask = np.zeros(img_shape, dtype=np.uint8)

    img_num = int(img_id)
    img_id_padded = f"{img_num:05d}"
    folder_num = img_num // 2000
    folder_path = mask_dir / str(folder_num)

    if not folder_path.exists():
        return combined_mask

    for part_name, class_id in CELEBAMASK_CLASSES.items():
        mask_file = folder_path / f"{img_id_padded}_{part_name}.png"
        if mask_file.exists():
            part_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if part_mask is not None:
                if part_mask.shape != img_shape:
                    part_mask = cv2.resize(part_mask, (img_shape[1], img_shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                combined_mask[part_mask > 128] = class_id

    return combined_mask


def load_celebamask_hq(data_dir, num_samples=5, random_sample=False):
    images = []
    masks = []

    if (data_dir / 'CelebAMask-HQ').exists():
        data_dir = data_dir / 'CelebAMask-HQ'

    img_dir = data_dir / 'CelebA-HQ-img'
    mask_dir = data_dir / 'CelebAMask-HQ-mask-anno'

    if not img_dir.exists():
        return images, masks

    all_img_files = sorted(list(set(glob.glob(str(img_dir / '*.jpg')))))

    if random_sample and len(all_img_files) > num_samples:
        img_files = random.sample(all_img_files, num_samples)
    else:
        img_files = all_img_files[:num_samples]

    for img_path in img_files:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

            img_id = Path(img_path).stem
            combined_mask = combine_celebamask_parts(mask_dir, img_id, img.shape[:2])
            masks.append(combined_mask)

    return images, masks


def load_dataset(dataset_name='auto', num_samples=5, random_sample=False):
    images, masks = [], []
    dataset_used = None

    if dataset_name == 'auto' or dataset_name == 'celebamask':
        images, masks = load_celebamask_hq(CELEBAMASK_DIR, num_samples, random_sample)
        if len(images) > 0:
            dataset_used = 'CelebAMask-HQ'
            return images, masks, dataset_used

    return images, masks, dataset_used


def load_celebamask_data(split='train', max_samples=100):
    data_dir = CELEBAMASK_DIR
    if (data_dir / 'CelebAMask-HQ').exists():
        data_dir = data_dir / 'CelebAMask-HQ'

    img_dir = data_dir / 'CelebA-HQ-img'
    mask_dir = data_dir / 'CelebAMask-HQ-mask-anno'

    if not img_dir.exists():
        return [], []

    all_img_files = sorted(glob.glob(str(img_dir / '*.jpg')))
    split_idx = int(len(all_img_files) * 0.8)

    if split == 'train':
        img_files = all_img_files[:split_idx]
    else:
        img_files = all_img_files[split_idx:]

    img_files = img_files[:max_samples]

    images = []
    masks = []

    for img_path in img_files:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

            img_id = Path(img_path).stem
            combined_mask = combine_celebamask_parts(mask_dir, img_id, img.shape[:2])
            masks.append(combined_mask)

    return images, masks


def visualize_predictions(images, true_masks, predictions, num_samples=2, save_path=None):
    num_samples = min(num_samples, len(images), len(true_masks), len(predictions))

    if num_samples == 0:
        return

    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        img = images[i]
        true_mask = true_masks[i]
        pred_mask = predictions[i]

        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image', fontweight='bold')
        axes[i, 0].axis('off')

        im1 = axes[i, 1].imshow(true_mask, cmap='tab20', vmin=0, vmax=19)
        axes[i, 1].set_title(f'Ground Truth ({len(np.unique(true_mask))} classes)',
                            fontweight='bold')
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        im2 = axes[i, 2].imshow(pred_mask, cmap='tab20', vmin=0, vmax=19)
        axes[i, 2].set_title(f'Prediction ({len(np.unique(pred_mask))} clusters)',
                            fontweight='bold')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

        overlay = img.copy().astype(float) / 255.0
        if pred_mask.max() > 0:
            mask_norm = pred_mask.astype(float) / pred_mask.max()
            mask_colored = plt.cm.tab20(mask_norm)[:, :, :3]
            mask_bool = pred_mask > 0
            overlay[mask_bool] = 0.6 * overlay[mask_bool] + 0.4 * mask_colored[mask_bool]

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay', fontweight='bold')
        axes[i, 3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")

    plt.show()


def calculate_semantic_metrics(pred_mask, gt_mask, num_classes=19):
    ious = []
    f1s = []
    dices = []

    for class_id in range(1, num_classes):
        pred_binary = (pred_mask == class_id)
        gt_binary = (gt_mask == class_id)

        if gt_binary.sum() == 0:
            continue

        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        pred_sum = pred_binary.sum()
        gt_sum = gt_binary.sum()

        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)

        precision = intersection / pred_sum if pred_sum > 0 else 0.0
        recall = intersection / gt_sum if gt_sum > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

        dice = 2 * intersection / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
        dices.append(dice)

    return {
        'mean_iou': np.mean(ious) if ious else 0.0,
        'mean_f1': np.mean(f1s) if f1s else 0.0,
        'mean_dice': np.mean(dices) if dices else 0.0,
        'num_classes_evaluated': len(ious)
    }


def calculate_clustering_metrics(pred_mask, gt_mask, num_classes=19):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()

    mask_valid = gt_flat > 0
    pred_valid = pred_flat[mask_valid]
    gt_valid = gt_flat[mask_valid]

    ari = adjusted_rand_score(gt_valid, pred_valid)
    nmi = normalized_mutual_info_score(gt_valid, pred_valid)

    unique_pred = np.unique(pred_valid)
    unique_gt = np.unique(gt_valid)

    ious = []
    for gt_cls in unique_gt:
        if gt_cls == 0:
            continue

        gt_pixels = (gt_flat == gt_cls)
        best_iou = 0

        for pred_cls in unique_pred:
            pred_pixels = (pred_flat == pred_cls)
            intersection = np.logical_and(gt_pixels, pred_pixels).sum()
            union = np.logical_or(gt_pixels, pred_pixels).sum()

            if union > 0:
                iou = intersection / union
                best_iou = max(best_iou, iou)

        ious.append(best_iou)

    mean_iou = np.mean(ious) if ious else 0.0

    return {
        'ARI': ari,
        'NMI': nmi,
        'Best_IoU': mean_iou,
        'num_clusters': len(unique_pred),
        'num_gt_classes': len(unique_gt)
    }