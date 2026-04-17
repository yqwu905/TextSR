"""Text Region Cropping via Affine Transformation.

Implements the text region extraction described in the Supplementary Material
of arXiv:2505.23119v1:

Given detection box b^(src) = [(x0,y0), (x1,y1), (x2,y2)] and target crop
b^(dst) = [(0,0), (0,h), (w,h)], compute affine transform θ using
cv2.getAffineTransform to extract text regions at height h=48.

Width w is computed to maintain aspect ratio, then padded/sliced to 480.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class TextCropper:
    """Extract text line crops from images using affine transformations.

    Each detected text region is warped to a standardized h×w format
    (default 48×480) for input to the TextSR diffusion model.
    """

    def __init__(
        self,
        target_height: int = 48,
        target_width: int = 480,
        min_text_height: int = 16,
        max_text_height: int = 512,
    ):
        """
        Args:
            target_height: Standard output height (48 pixels per paper).
            target_width: Standard output width (480 pixels per paper).
            min_text_height: Minimum text height to keep (filter too small).
            max_text_height: Maximum text height to keep (filter too large).
        """
        self.target_height = target_height
        self.target_width = target_width
        self.min_text_height = min_text_height
        self.max_text_height = max_text_height

    def compute_crop_width(self, src_points: np.ndarray) -> int:
        """Compute the appropriate crop width to maintain aspect ratio.

        Args:
            src_points: (3, 2) source triangle points from detection box.

        Returns:
            Width in pixels, capped at target_width.
        """
        # Compute text region dimensions from detection coordinates
        # Edge 0→1 is the height edge, edge 0→2 is approximately the width edge
        edge_h = np.linalg.norm(src_points[1] - src_points[0])
        edge_w = np.linalg.norm(src_points[2] - src_points[1])

        if edge_h < 1e-5:
            return self.target_width

        aspect_ratio = edge_w / edge_h
        crop_w = int(self.target_height * aspect_ratio)
        return min(max(crop_w, 1), self.target_width)

    def estimate_text_height(self, src_points: np.ndarray) -> float:
        """Estimate the text height in the original image."""
        return float(np.linalg.norm(src_points[1] - src_points[0]))

    def crop_text_region(
        self,
        image: np.ndarray,
        src_points: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract a single text region using affine warp.

        Args:
            image: (H, W, 3) source image.
            src_points: (3, 2) float32 source triangle from text detection.
                Format: [(x0,y0), (x1,y1), (x2,y2)] — top-left, bottom-left,
                bottom-right (the 4th corner is inferred for affine).

        Returns:
            crop: (target_height, crop_w, 3) warped text crop, or None if filtered.
            affine_matrix: (2, 3) affine transform matrix θ, or None.
        """
        text_h = self.estimate_text_height(src_points)

        # Filter by height
        if text_h < self.min_text_height or text_h > self.max_text_height:
            return None, None

        # Compute output width maintaining aspect ratio
        crop_w = self.compute_crop_width(src_points)

        # Destination points: [(0,0), (0,h), (w,h)]
        dst_points = np.array([
            [0, 0],
            [0, self.target_height],
            [crop_w, self.target_height],
        ], dtype=np.float32)

        src_pts = src_points.astype(np.float32)

        # Compute affine transform: θ = cv2.getAffineTransform(src, dst)
        affine_matrix = cv2.getAffineTransform(src_pts, dst_points)

        # Apply warp
        crop = cv2.warpAffine(
            image, affine_matrix,
            (crop_w, self.target_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        return crop, affine_matrix

    def pad_to_target_width(
        self,
        crop: np.ndarray,
        pad_value: int = 0,
    ) -> np.ndarray:
        """Pad or slice crop to target_width.

        Per paper: "we either slice or pad these images to ensure a uniform
        dimension of 48×480 pixels."

        Args:
            crop: (H, W, 3) text crop where W ≤ target_width.
            pad_value: Pixel value for padding.

        Returns:
            (target_height, target_width, 3) padded/sliced crop.
        """
        h, w = crop.shape[:2]

        if w >= self.target_width:
            # Slice to target width
            return crop[:, :self.target_width]

        # Right-pad to target width
        padded = np.full(
            (self.target_height, self.target_width, 3),
            pad_value, dtype=crop.dtype,
        )
        padded[:h, :w] = crop
        return padded

    def process_image(
        self,
        image: np.ndarray,
        detection_boxes: List[np.ndarray],
    ) -> List[dict]:
        """Process all detected text regions in an image.

        Args:
            image: (H, W, 3) uint8 image.
            detection_boxes: List of (3, 2) or (4, 2) float32 detection polygons.
                If (4, 2), uses first 3 points for affine transform.

        Returns:
            List of dicts with keys:
                'crop': (48, 480, 3) uint8 padded text crop
                'affine_matrix': (2, 3) transform matrix θ
                'original_width': int, crop width before padding
        """
        results = []

        for box in detection_boxes:
            if box.shape[0] >= 4:
                # Use first 3 points for affine (4th is redundant)
                src_pts = box[:3].astype(np.float32)
            else:
                src_pts = box.astype(np.float32)

            crop, affine = self.crop_text_region(image, src_pts)
            if crop is None:
                continue

            orig_w = crop.shape[1]
            padded_crop = self.pad_to_target_width(crop)

            results.append({
                "crop": padded_crop,
                "affine_matrix": affine,
                "original_width": orig_w,
            })

        return results

    @staticmethod
    def compute_inverse_affine(
        affine_matrix: np.ndarray,
        scale_factor: int = 1,
    ) -> np.ndarray:
        """Compute inverse affine transform for pasting back.

        Per Supplementary: θ^{-1} = cv2.invertAffineTransform(θ)
        For 2x/4x upscaling, additional scaling is applied.

        Args:
            affine_matrix: (2, 3) forward transform.
            scale_factor: Upscaling factor (1, 2, or 4).

        Returns:
            (2, 3) inverse affine transform.
        """
        inv = cv2.invertAffineTransform(affine_matrix)

        if scale_factor != 1:
            # Scale the translation components
            scale_matrix = np.array([
                [scale_factor, 0, 0],
                [0, scale_factor, 0],
            ], dtype=np.float64)
            # Compose: first inverse affine, then scale
            # For the inverse: map from crop space to original at higher resolution
            inv[:, :2] *= scale_factor
            inv[:, 2] *= scale_factor

        return inv
