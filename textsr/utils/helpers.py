"""Utility functions for TextSR."""

from typing import Dict

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict:
    """Load YAML configuration file.

    Args:
        path: Path to YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: PyTorch module.
        trainable_only: If True, count only parameters requiring gradients.

    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a [-1,1] float tensor (C,H,W) to uint8 (H,W,C) numpy array.

    Args:
        tensor: (C, H, W) or (B, C, H, W) float32 tensor in [-1, 1].

    Returns:
        (H, W, C) or (B, H, W, C) uint8 numpy array in [0, 255].
    """
    if tensor.dim() == 4:
        # Batch mode
        imgs = []
        for t in tensor:
            imgs.append(tensor_to_image(t))
        return np.stack(imgs)

    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW → HWC
    img = (img + 1.0) * 127.5
    return np.clip(img, 0, 255).astype(np.uint8)


def image_to_tensor(image: np.ndarray, device: torch.device = None) -> torch.Tensor:
    """Convert uint8 (H,W,C) numpy array to [-1,1] float tensor (1,C,H,W).

    Args:
        image: (H, W, C) uint8 numpy array.
        device: Target device.

    Returns:
        (1, C, H, W) float32 tensor in [-1, 1].
    """
    img = image.astype(np.float32) / 127.5 - 1.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    tensor = torch.from_numpy(img).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def make_grid(images: list, nrow: int = 8, padding: int = 2) -> np.ndarray:
    """Arrange a list of (H,W,C) images into a grid.

    Args:
        images: List of (H,W,C) uint8 numpy arrays (all same size).
        nrow: Number of images per row.
        padding: Pixels between images.

    Returns:
        (grid_H, grid_W, C) uint8 numpy array.
    """
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    h, w, c = images[0].shape
    n = len(images)
    ncol = min(nrow, n)
    nrow_actual = (n + ncol - 1) // ncol

    grid_h = nrow_actual * h + (nrow_actual - 1) * padding
    grid_w = ncol * w + (ncol - 1) * padding
    grid = np.full((grid_h, grid_w, c), 255, dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // ncol
        col = idx % ncol
        y = row * (h + padding)
        x = col * (w + padding)
        grid[y:y + h, x:x + w] = img

    return grid


class AverageMeter:
    """Computes and stores running average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
