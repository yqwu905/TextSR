"""
Real-ESRGAN style degradation pipeline for synthetic LR generation.

Based on: https://arxiv.org/abs/2107.10833
Applied 20× per image during training (paper: "random high-order operations").

Degradation order (randomly shuffled):
  1. Gaussian blur / motion blur / zoom blur
  2. Downscale (bilinear / bicubic / nearest)
  3. Additive noise (Gaussian / Poisson)
  4. JPEG compression artifacts
"""

import math
import random
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Low-level ops
# ---------------------------------------------------------------------------

def random_gaussian_blur(img: np.ndarray, kernel_range=(7, 21), sigma_range=(0.5, 3.0)) -> np.ndarray:
    kernel_size = random.choice(range(kernel_range[0], kernel_range[1] + 1, 2))
    sigma = random.uniform(*sigma_range)
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def random_motion_blur(img: np.ndarray, kernel_range=(5, 15)) -> np.ndarray:
    kernel_size = random.randint(*kernel_range)
    angle = random.uniform(0, 180)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0
    kernel = kernel / kernel.sum()
    M = cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel = kernel / (kernel.sum() + 1e-8)
    return cv2.filter2D(img, -1, kernel)


def random_zoom_blur(img: np.ndarray, strength_range=(0.05, 0.3)) -> np.ndarray:
    """Simulate zoom / radial blur."""
    h, w = img.shape[:2]
    strength = random.uniform(*strength_range)
    n_iter = random.randint(3, 8)
    result = img.astype(np.float32)
    for i in range(1, n_iter + 1):
        scale = 1.0 + strength * i / n_iter
        enlarged = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        # Centre-crop back to original size
        dh = (enlarged.shape[0] - h) // 2
        dw = (enlarged.shape[1] - w) // 2
        cropped = enlarged[dh:dh + h, dw:dw + w]
        result += cropped.astype(np.float32)
    return np.clip(result / (n_iter + 1), 0, 255).astype(np.uint8)


def random_downscale(
    img: np.ndarray,
    scale_range: Tuple[float, float] = (0.5, 0.9),
    interp_choices=(cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST),
) -> np.ndarray:
    h, w = img.shape[:2]
    scale = random.uniform(*scale_range)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    interp = random.choice(interp_choices)
    downscaled = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)


def random_gaussian_noise(img: np.ndarray, sigma_range=(1, 30)) -> np.ndarray:
    sigma = random.uniform(*sigma_range)
    noise = np.random.randn(*img.shape) * sigma
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_poisson_noise(img: np.ndarray, scale_range=(0.05, 2.0)) -> np.ndarray:
    scale = random.uniform(*scale_range)
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(img.astype(np.float32) / 255.0 * vals * scale)
    noisy = noisy / (vals * scale) * 255.0
    return np.clip(noisy, 0, 255).astype(np.uint8)


def random_jpeg_compression(img: np.ndarray, quality_range=(20, 95)) -> np.ndarray:
    quality = random.randint(*quality_range)
    pil_img = Image.fromarray(img)
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf))


# ---------------------------------------------------------------------------
# High-order degradation pipeline
# ---------------------------------------------------------------------------

BLUR_OPS = [random_gaussian_blur, random_motion_blur, random_zoom_blur]
NOISE_OPS = [random_gaussian_noise, random_poisson_noise]


def apply_single_degradation(img: np.ndarray) -> np.ndarray:
    """
    One round of random degradation:
      blur → downscale → noise → jpeg
    Each step applied with some probability.
    """
    # Step 1: Random blur
    if random.random() < 0.8:
        blur_fn = random.choice(BLUR_OPS)
        img = blur_fn(img)

    # Step 2: Downscale
    if random.random() < 0.8:
        img = random_downscale(img, scale_range=(0.5, 0.95))

    # Step 3: Noise
    if random.random() < 0.5:
        noise_fn = random.choice(NOISE_OPS)
        img = noise_fn(img)

    # Step 4: JPEG
    if random.random() < 0.7:
        img = random_jpeg_compression(img)

    return img


def real_esrgan_degrade(img: np.ndarray, num_rounds: int = 20) -> np.ndarray:
    """
    Apply Real-ESRGAN high-order degradation pipeline.
    Paper: applied 20× per image with random high-order operations.

    Args:
        img: uint8 RGB image (H, W, 3)
        num_rounds: number of degradation rounds (paper uses 20)

    Returns:
        Degraded image (same spatial size as input)
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    for _ in range(num_rounds):
        img = apply_single_degradation(img)

    return img


def degrade_and_resize(
    hr_img: np.ndarray,
    lr_size: Tuple[int, int],
    num_rounds: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate (LR, HR) pair from a clean HR image.

    Args:
        hr_img: clean HR image (uint8 RGB)
        lr_size: target LR size as (H, W)
        num_rounds: number of degradation rounds

    Returns:
        (lr_img, hr_img): both uint8 RGB
    """
    # Apply degradation
    degraded = real_esrgan_degrade(hr_img.copy(), num_rounds)

    # Resize degraded to LR size
    lr_img = cv2.resize(
        degraded,
        (lr_size[1], lr_size[0]),   # cv2 uses (W, H)
        interpolation=cv2.INTER_LINEAR,
    )

    return lr_img, hr_img


# ---------------------------------------------------------------------------
# Utility: simple bicubic downscale (for inference / baseline)
# ---------------------------------------------------------------------------

def bicubic_downscale(img: np.ndarray, scale: int = 2) -> np.ndarray:
    h, w = img.shape[:2]
    lr = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    return lr


def bicubic_upsample(img: np.ndarray, scale: int = 2) -> np.ndarray:
    h, w = img.shape[:2]
    hr = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    return hr
