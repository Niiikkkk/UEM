import os
import argparse

import numpy as np
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import average_precision_score, roc_curve, auc
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb import Image

from modeling.ood_segmentation import OoDSegmentationModel
from easydict import EasyDict as edict

from tools import overwrite_config
from pytorch_lightning import Trainer
import torch
import albumentations as A
from torchmetrics import JaccardIndex
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.patches import Patch

ANOMALY_CLASS_INFO = {
    30: ("pothole", (1.0, 0.5, 0.0)),   # orange
    33: ("tiny", (1.0, 0.0, 0.0)),      # red
    34: ("small", (0.0, 0.0, 1.0)),     # blue
    35: ("medium", (0.0, 0.8, 0.0)),    # green
    36: ("large", (0.7, 0.0, 0.7)),     # purple
}

def main(args):

    if args.multiple_datasets:
        datasets = args.dataset.split(",")
    else:
        datasets = [args.dataset]

    transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    segmentor_ckpt = None
    if args.segmentor_ckpt is not None:
        segmentor_ckpt = args.segmentor_ckpt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OoDSegmentationModel.load_from_checkpoint(
        args.ckpt, segmentor_ckpt=segmentor_ckpt)

    img_path = "/home/nicholas/Desktop/main_UE4/output/Sunny/2/rgb/coffecup_umbrella_skateboard_table_pilesand__131554.png"
    image = np.array(Image.open(img_path).convert('RGB'))
    label = np.array(Image.open(img_path.replace("rgb", "semantic/original")))
    label = label[:, :, 0]

    aut = transform(image=image, mask=label)
    image, label = aut['image'], aut['mask']
    image = image.unsqueeze(0)
    label = label.unsqueeze(0)

    model.to(device)
    image=image.to(device)
    label=label.to(device)
    model.eval()

    with torch.no_grad():
        out = model.sliding_window_inference(image, label.shape, 30, [308, 406], [140, 140], True,True)

        #show_id_pred(out, label)
        compute_ood(out,label)

        # Visualize OOD and LLR scores
        ood_score = get_ood_score(out)
        llr_score = get_llr_score(out)
        show_scores_visualization(
            image.squeeze(0),
            ood_score,
            llr_score,
            label=label.squeeze(0),
            save_path="ood_llr_scores.png",
        )
        #show_scores_overlay(image.squeeze(0), ood_score, llr_score, alpha=0.6, save_path="ood_llr_overlay.png")



def compute_ood(out,label):
    ood_score = get_ood_score(out)
    llr_score = get_llr_score(out)
    from evaluate_ood_mod import evaluate_ood

    ood_lbl = label.cpu().numpy()

    print("LLR score")
    evaluate_ood(llr_score.cpu().numpy(), ood_lbl)
    print("OOD score")
    evaluate_ood(ood_score.cpu().numpy(), ood_lbl)

def show_id_pred(out,label):
    iou_metric = JaccardIndex(
        task="multiclass",
        num_classes=30,
        ignore_index=255,
    )

    out = out.sem_seg
    
    print(iou_metric(out.cpu(), label.cpu()))

    pred = out.squeeze()
    pred = torch.argmax(pred, dim=0)

    colors = [
        [0, 0, 0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [110, 190, 160],
        [170, 120, 50],
        [55, 90, 80],
        [45, 60, 150],
        [157, 234, 50],
        [81, 0, 81],
        [150, 100, 100],
        [230, 150, 140],
        [180, 165, 180],
        [180, 130, 70]
    ]

    color = get_seg_colormap(pred, colors)

    Image.fromarray(np.uint8(color.cpu().numpy())).show()

def get_ood_score(out):
    return out.ood_score[:, 1]

def get_llr_score(out):
    ood_score = out.ood_score[:, 1]
    idd_score_uem = out.ood_score[:, 0]
    idd_score_segmentor = out.sem_seg.max(dim = 1)[0]
    return ood_score - idd_score_segmentor - idd_score_uem

def get_seg_colormap(preds, colors):
    """
    Assuming preds.shape = (H,W)
    """
    H, W = preds.shape
    color_map = torch.zeros((H, W, 3)).long()

    for i in range(len(colors)):
        mask = (preds == i)
        if mask.sum() == 0:
            continue
        color_map[mask, :] = torch.tensor(colors[i])

    return color_map


def colorize_anomaly_labels(label_np):
    """Create an RGB map and legend patches for anomaly subclasses."""
    color_map = np.zeros((label_np.shape[0], label_np.shape[1], 3), dtype=np.float32)
    legend = []

    for class_id, (name, color) in ANOMALY_CLASS_INFO.items():
        mask = label_np == class_id
        if np.any(mask):
            color_map[mask] = color
            legend.append(Patch(facecolor=color, edgecolor="none", label=f"{name} ({class_id})"))

    # Fallback for unknown anomaly ids >= 30
    unknown_mask = (label_np >= 30) & (~np.isin(label_np, list(ANOMALY_CLASS_INFO.keys())))
    if np.any(unknown_mask):
        unknown_color = (1.0, 1.0, 0.0)
        color_map[unknown_mask] = unknown_color
        legend.append(Patch(facecolor=unknown_color, edgecolor="none", label="other anomaly"))

    return color_map, legend

def show_scores_visualization(image, ood_score, llr_score, label=None, save_path="visualization_scores.png"):
    """
    Visualize the original image alongside OOD and LLR score heatmaps.
    
    Args:
        image: torch.Tensor of shape (C, H, W) with normalized values
        ood_score: torch.Tensor of shape (H, W) with OOD scores
        llr_score: torch.Tensor of shape (H, W) with LLR scores
        label: Optional tensor of shape (H, W) with class ids
        save_path: Path to save the visualization image
    """
    # Convert image tensor to numpy for visualization
    if isinstance(image, torch.Tensor):
        img_np = image.permute(1, 2, 0).cpu().numpy()
        # Denormalize if it was normalized
        img_np = (img_np * np.array([0.229, 0.224, 0.225]) +
                  np.array([0.485, 0.456, 0.406]))
        img_np = np.clip(img_np, 0, 1)
    else:
        img_np = image
    
    # Convert scores to numpy
    ood_score_np = ood_score.squeeze().cpu().numpy() if isinstance(ood_score, torch.Tensor) else ood_score
    llr_score_np = llr_score.squeeze().cpu().numpy() if isinstance(llr_score, torch.Tensor) else llr_score
    
    # Create figure with 4 subplots (image, label, OOD, LLR)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    
    # Display original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Display anomaly labels with class colors
    axes[1].imshow(img_np)
    if label is not None:
        label_np = label.squeeze().detach().cpu().numpy() if isinstance(label, torch.Tensor) else np.squeeze(label)
        label_rgb, legend_items = colorize_anomaly_labels(label_np)
        axes[1].imshow(label_rgb, alpha=0.65)
        if legend_items:
            axes[1].legend(
                handles=legend_items,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.28),
                ncol=max(1, min(3, len(legend_items))),
                fontsize=8,
                frameon=True,
            )
    axes[1].set_title('Anomaly Labels')
    axes[1].axis('off')

    # Display OOD score heatmap
    im1 = axes[2].imshow(ood_score_np, cmap='RdYlBu_r')
    axes[2].set_title('OOD Score (Hot Map)')
    axes[2].axis('off')
    plt.colorbar(im1, ax=axes[2])

    # Display LLR score heatmap
    im2 = axes[3].imshow(llr_score_np, cmap='RdYlBu_r')
    axes[3].set_title('LLR Score')
    axes[3].axis('off')
    plt.colorbar(im2, ax=axes[3])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()

def show_scores_overlay(image, ood_score, llr_score, alpha=0.5, save_path="visualization_overlay.png"):
    """
    Visualize scores overlaid on the original image.
    
    Args:
        image: torch.Tensor of shape (C, H, W) with normalized values
        ood_score: torch.Tensor of shape (H, W) with OOD scores
        llr_score: torch.Tensor of shape (H, W) with LLR scores
        alpha: Transparency level for the overlay (0-1)
        save_path: Path to save the visualization
    """
    # Convert image tensor to numpy for visualization
    if isinstance(image, torch.Tensor):
        img_np = image.permute(1, 2, 0).cpu().numpy()
        # Denormalize if it was normalized
        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + 
                  np.array([0.485, 0.456, 0.406]))
        img_np = np.clip(img_np, 0, 1)
    else:
        img_np = image
    
    # Convert scores to numpy
    ood_score_np = ood_score.squeeze().cpu().numpy() if isinstance(ood_score, torch.Tensor) else ood_score
    llr_score_np = llr_score.squeeze().cpu().numpy() if isinstance(llr_score, torch.Tensor) else llr_score
    
    # Normalize scores to 0-1 range
    ood_norm = (ood_score_np - ood_score_np.min()) / (ood_score_np.max() - ood_score_np.min() + 1e-8)
    llr_norm = (llr_score_np - llr_score_np.min()) / (llr_score_np.max() - llr_score_np.min() + 1e-8)

    # Create figure with overlays
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original image
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # OOD score overlay
    axes[0, 1].imshow(img_np)
    im1 = axes[0, 1].imshow(ood_score_np, cmap='hot', alpha=alpha)
    axes[0, 1].set_title('OOD Score Overlay')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # LLR score overlay
    axes[1, 0].imshow(img_np)
    im2 = axes[1, 0].imshow(llr_score_np, cmap='RdYlBu_r', alpha=alpha)
    axes[1, 0].set_title('LLR Score Overlay')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Combined heatmaps side by side
    axes[1, 1].imshow(ood_score_np, cmap='hot', alpha=0.7, label='OOD')
    im3 = axes[1, 1].imshow(llr_score_np, cmap='RdYlBu_r', alpha=0.3)
    axes[1, 1].set_title('OOD (70%) + LLR (30%)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Overlay visualization saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OoD Metrics")

    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--segmentor-ckpt",
        type=str,
        default=None,
        help="Path to the segmentation model checkpoint, if none then use the checkpoint stored in the model config"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="road_anomaly",
        help="Dataset to evaluate"
    )
    parser.add_argument(
        "--multiple-datasets",
        action="store_true",
        help="Evaluate multiple datasets, names given separated by commas in --dataset argument"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="Path to save the results as a text file"
    )
    parser.add_argument(
        "--metrics-order",
        type=str,
        default="llr_AUPR,llr_AUROC,llr_FPR95",
    )
    parser.add_argument(
        "--datasets-folder",
        type=str,
        default=None,
        help="Path to the evaluation datasets folder"
    )
    parser.add_argument(
        "--store-anomaly-scores",
        action="store_true",
        help="Store anomaly scores in the output"
    )

    args = parser.parse_args()

    main(args)
