"""Real-ESRGAN Style High-Order Degradation Pipeline.

Implements the degradation process described in Section 4 of arXiv:2505.23119v1:
- Each clean image is duplicated 20 times with random complex degradations
- Two-stage degradation following Real-ESRGAN (Wang et al., 2021):
  Stage 1: blur → resize → noise → JPEG
  Stage 2: blur → resize → noise → JPEG → sinc filter

This produces realistic low-quality training images that mimic real-world
camera capture and processing artifacts.
"""

import random
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def random_gaussian_kernel(
    kernel_size: int,
    sigma_range: Tuple[float, float] = (0.2, 3.0),
    isotropic: bool = True,
) -> np.ndarray:
    """Generate a random Gaussian blur kernel.

    Args:
        kernel_size: Must be odd.
        sigma_range: Range of sigma values.
        isotropic: If True, use same sigma for both axes.

    Returns:
        (kernel_size, kernel_size) normalized kernel.
    """
    sigma_x = random.uniform(*sigma_range)
    if isotropic:
        sigma_y = sigma_x
    else:
        sigma_y = random.uniform(*sigma_range)

    kernel = cv2.getGaussianKernel(kernel_size, sigma_x) @ \
             cv2.getGaussianKernel(kernel_size, sigma_y).T
    kernel = kernel / kernel.sum()
    return kernel


def random_sinc_kernel(kernel_size: int = 21) -> np.ndarray:
    """Generate a random sinc filter kernel (for second-order degradation).

    This simulates sensor-level artifacts such as ringing.
    """
    omega = random.uniform(np.pi / 3, np.pi)
    half = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)

    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - half
            y = j - half
            r = np.sqrt(x ** 2 + y ** 2)
            if r == 0:
                kernel[i, j] = omega / np.pi
            else:
                kernel[i, j] = omega * np.sinc(omega * r / np.pi) / (np.pi * r)

    # Normalize
    kernel = kernel / kernel.sum()
    return kernel.astype(np.float32)


class RealESRGANDegradation:
    """Two-stage Real-ESRGAN degradation pipeline.

    Produces a degraded low-resolution image from a clean high-resolution input.
    """

    def __init__(
        self,
        # Stage 1 parameters
        blur_kernel_size_range: Tuple[int, int] = (7, 21),
        blur_sigma_range: Tuple[float, float] = (0.2, 3.0),
        downsample_range: Tuple[float, float] = (0.5, 1.0),
        noise_sigma_range: Tuple[float, float] = (0, 25),
        jpeg_quality_range: Tuple[int, int] = (30, 95),
        # Stage 2 parameters
        blur_kernel_size_range_2: Tuple[int, int] = (7, 21),
        blur_sigma_range_2: Tuple[float, float] = (0.2, 1.5),
        downsample_range_2: Tuple[float, float] = (0.5, 1.0),
        noise_sigma_range_2: Tuple[float, float] = (0, 20),
        jpeg_quality_range_2: Tuple[int, int] = (30, 95),
    ):
        self.stage1 = {
            "blur_kernel_size_range": blur_kernel_size_range,
            "blur_sigma_range": blur_sigma_range,
            "downsample_range": downsample_range,
            "noise_sigma_range": noise_sigma_range,
            "jpeg_quality_range": jpeg_quality_range,
        }
        self.stage2 = {
            "blur_kernel_size_range": blur_kernel_size_range_2,
            "blur_sigma_range": blur_sigma_range_2,
            "downsample_range": downsample_range_2,
            "noise_sigma_range": noise_sigma_range_2,
            "jpeg_quality_range": jpeg_quality_range_2,
        }

    def _apply_blur(self, img: np.ndarray, params: Dict) -> np.ndarray:
        """Apply random Gaussian blur."""
        ks_min, ks_max = params["blur_kernel_size_range"]
        kernel_size = random.choice(range(ks_min, ks_max + 1, 2))  # must be odd
        isotropic = random.random() > 0.3
        kernel = random_gaussian_kernel(
            kernel_size, params["blur_sigma_range"], isotropic
        )
        return cv2.filter2D(img, -1, kernel)

    def _apply_resize(self, img: np.ndarray, params: Dict,
                      target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Apply random downsampling then upsampling back to original size."""
        h, w = img.shape[:2]
        scale = random.uniform(*params["downsample_range"])
        interp_methods = [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            interp_down = random.choice(interp_methods)
            img = cv2.resize(img, (new_w, new_h), interpolation=interp_down)

            # Resize back
            if target_size is not None:
                out_h, out_w = target_size
            else:
                out_h, out_w = h, w
            interp_up = random.choice(interp_methods)
            img = cv2.resize(img, (out_w, out_h), interpolation=interp_up)

        return img

    def _apply_noise(self, img: np.ndarray, params: Dict) -> np.ndarray:
        """Add random Gaussian or Poisson noise."""
        noise_type = random.choice(["gaussian", "poisson"])
        sigma = random.uniform(*params["noise_sigma_range"])

        if noise_type == "gaussian" and sigma > 0:
            noise = np.random.randn(*img.shape).astype(np.float32) * sigma
            img = img.astype(np.float32) + noise
        elif noise_type == "poisson" and sigma > 0:
            # Scale-adjusted Poisson noise
            vals = len(np.unique(img))
            vals = 2 ** np.ceil(np.log2(max(vals, 1)))
            img = img.astype(np.float32)
            noisy = np.random.poisson(np.maximum(img, 0) * vals) / float(vals)
            img = img + (noisy - img) * (sigma / 25.0)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _apply_jpeg(self, img: np.ndarray, params: Dict) -> np.ndarray:
        """Apply random JPEG compression artifacts."""
        quality = random.randint(*params["jpeg_quality_range"])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encimg, 1)
        return img

    def _apply_sinc(self, img: np.ndarray) -> np.ndarray:
        """Apply sinc filter (second-order degradation only)."""
        if random.random() < 0.5:
            return img
        kernel = random_sinc_kernel(kernel_size=random.choice([7, 9, 11, 13, 15, 17, 19, 21]))
        return cv2.filter2D(img, -1, kernel)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply complete two-stage degradation to an image.

        Args:
            img: (H, W, 3) uint8 clean HR image.

        Returns:
            (H, W, 3) uint8 degraded image at same resolution.
        """
        assert img.dtype == np.uint8, "Input must be uint8"
        h, w = img.shape[:2]

        # === Stage 1 ===
        out = self._apply_blur(img, self.stage1)
        out = self._apply_resize(out, self.stage1, target_size=(h, w))
        out = self._apply_noise(out, self.stage1)
        out = self._apply_jpeg(out, self.stage1)

        # === Stage 2 ===
        out = self._apply_blur(out, self.stage2)
        out = self._apply_resize(out, self.stage2, target_size=(h, w))
        out = self._apply_noise(out, self.stage2)
        out = self._apply_jpeg(out, self.stage2)
        out = self._apply_sinc(out)

        return out


def generate_degraded_copies(
    img: np.ndarray,
    num_copies: int = 20,
    degradation: Optional[RealESRGANDegradation] = None,
) -> list:
    """Generate multiple degraded copies of an image.

    Per paper: "Each image was duplicated 20 times then each had random
    high-order Real-ESRGAN degradation operations applied."

    Args:
        img: (H, W, 3) uint8 clean image.
        num_copies: Number of degraded copies.
        degradation: Degradation pipeline. Creates default if None.

    Returns:
        List of num_copies degraded images.
    """
    if degradation is None:
        degradation = RealESRGANDegradation()

    return [degradation(img.copy()) for _ in range(num_copies)]
