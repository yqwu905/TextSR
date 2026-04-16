"""LPF Gaussian Blending for TextSR.

Implements the blending algorithm from the Supplementary Material
of arXiv:2505.23119v1:

    T̃'_i = g(T_i) + LPF(f(T_i)) - LPF(g(T_i))

Where:
- g(T_i) = TextSR output (super-resolved text crop)
- f(T_i) = Real-ESRGAN output cropped from the same region
- LPF = Low-pass filter (Gaussian filter with σ=3.0)

The intuition: Real-ESRGAN provides low-frequency elements (color, basic
structure) while TextSR contributes high-frequency details (sharp characters).
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class TextSRBlender:
    """Blend TextSR text region outputs back into the Real-ESRGAN base image.

    Algorithm:
    1. For each text region, compute the LPF-blended prediction
    2. Warp the blended prediction back to the original image coordinates
    3. Paste onto the Real-ESRGAN upscaled base image
    """

    def __init__(self, sigma: float = 3.0):
        """
        Args:
            sigma: Gaussian sigma for the low-pass filter (paper uses 3.0).
        """
        self.sigma = sigma

    def lowpass_filter(self, img: np.ndarray) -> np.ndarray:
        """Apply Gaussian low-pass filter.

        Args:
            img: (H, W, 3) float32 image.

        Returns:
            (H, W, 3) low-pass filtered image.
        """
        # Kernel size should be large enough for the sigma value
        # Rule of thumb: kernel_size = 2 * ceil(3*sigma) + 1
        ksize = int(2 * np.ceil(3 * self.sigma) + 1)
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(img, (ksize, ksize), self.sigma)

    def blend_text_region(
        self,
        textsr_output: np.ndarray,    # g(T_i): TextSR super-resolved crop
        esrgan_crop: np.ndarray,       # f(T_i): Real-ESRGAN crop of same region
    ) -> np.ndarray:
        """Blend a single text region with LPF harmonization.

        T̃'_i = g(T_i) + LPF(f(T_i)) - LPF(g(T_i))

        This preserves high-frequency text details from TextSR while
        adopting the color/tone from Real-ESRGAN for smooth transitions.

        Args:
            textsr_output: (H, W, 3) float32 TextSR result.
            esrgan_crop: (H, W, 3) float32 Real-ESRGAN result for same region.

        Returns:
            (H, W, 3) float32 harmonized text crop.
        """
        assert textsr_output.shape == esrgan_crop.shape, \
            f"Shape mismatch: {textsr_output.shape} vs {esrgan_crop.shape}"

        lpf_esrgan = self.lowpass_filter(esrgan_crop)
        lpf_textsr = self.lowpass_filter(textsr_output)

        # T̃'_i = g(T_i) + LPF(f(T_i)) - LPF(g(T_i))
        blended = textsr_output + lpf_esrgan - lpf_textsr

        return np.clip(blended, 0, 255).astype(np.float32)

    def paste_region_back(
        self,
        base_image: np.ndarray,         # Real-ESRGAN upscaled full image
        text_crop: np.ndarray,           # Blended text crop
        inverse_affine: np.ndarray,      # θ^{-1}: inverse affine transform
        crop_width: int,                 # Original crop width (before padding)
        crop_height: int = 48,           # Standard crop height
    ) -> np.ndarray:
        """Paste a blended text crop back onto the base image.

        Uses inverse affine transform to map from crop coordinates back
        to the full image coordinates.

        Args:
            base_image: (H_out, W_out, 3) Real-ESRGAN output.
            text_crop: (crop_height, crop_width, 3) blended text region.
            inverse_affine: (2, 3) inverse affine matrix.
            crop_width: Width of valid text region (before padding).
            crop_height: Height of text crop.

        Returns:
            (H_out, W_out, 3) image with text region pasted.
        """
        out_h, out_w = base_image.shape[:2]
        result = base_image.copy()

        # Take only the valid (non-padded) portion
        valid_crop = text_crop[:crop_height, :crop_width]

        # Warp the crop to full image coordinates
        warped = cv2.warpAffine(
            valid_crop, inverse_affine,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

        # Create a mask for the warped region
        mask = np.ones(valid_crop.shape[:2], dtype=np.float32)
        warped_mask = cv2.warpAffine(
            mask, inverse_affine,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Feather the mask edges for smooth transitions
        warped_mask = cv2.GaussianBlur(warped_mask, (3, 3), 0.5)
        warped_mask = np.expand_dims(warped_mask, axis=-1)

        # Composite
        result = result * (1 - warped_mask) + warped * warped_mask

        return result.astype(base_image.dtype)

    def blend_full_image(
        self,
        esrgan_image: np.ndarray,
        textsr_crops: List[np.ndarray],
        esrgan_crops: List[np.ndarray],
        inverse_affines: List[np.ndarray],
        crop_widths: List[int],
        crop_height: int = 48,
    ) -> np.ndarray:
        """Full-image blending pipeline.

        I' = f(I) ⊕ Φ(T̃'_1, θ_1^{-1}) ⊕ ... ⊕ Φ(T̃'_N, θ_N^{-1})

        Args:
            esrgan_image: (H, W, 3) float32 Real-ESRGAN upscaled full image.
            textsr_crops: List of (48, W_i, 3) TextSR outputs.
            esrgan_crops: List of (48, W_i, 3) Real-ESRGAN crops (same regions).
            inverse_affines: List of (2, 3) inverse affine matrices.
            crop_widths: List of original crop widths.
            crop_height: Standard text crop height.

        Returns:
            (H, W, 3) final super-resolved image.
        """
        result = esrgan_image.copy()

        for textsr_crop, esrgan_crop, inv_affine, cw in zip(
            textsr_crops, esrgan_crops, inverse_affines, crop_widths
        ):
            # Step 1: LPF blending for harmonization
            blended_crop = self.blend_text_region(textsr_crop, esrgan_crop)

            # Step 2: Paste back with inverse affine
            result = self.paste_region_back(
                result, blended_crop, inv_affine, cw, crop_height
            )

        return result
