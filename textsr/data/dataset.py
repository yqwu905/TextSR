"""TextSR Dataset.

Provides paired (LR, HR, text) triplets for training the diffusion model.
Supports two modes:

1. **Pre-processed mode**: Reads pre-cropped and pre-degraded 48×480 image pairs
   from disk (recommended for large-scale training with 18M crops).

2. **On-the-fly mode**: Reads full images + detection annotations, applies
   cropping and degradation at load time (useful for development/debugging).

The diffusion target is the residual: x_0 = HR - LR.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .degradation import RealESRGANDegradation
from .text_crop import TextCropper


class TextSRDataset(Dataset):
    """TextSR training dataset.

    Each sample returns:
        - lr_image: (3, 48, 480) float32 tensor in [-1, 1]
        - hr_image: (3, 48, 480) float32 tensor in [-1, 1]
        - text: string (UTF-8 text content for this crop)
    """

    def __init__(
        self,
        data_root: str,
        annotation_file: Optional[str] = None,
        mode: str = "preprocessed",   # "preprocessed" or "on_the_fly"
        image_height: int = 48,
        image_width: int = 480,
        degradation: Optional[RealESRGANDegradation] = None,
    ):
        """
        Args:
            data_root: Root directory of the dataset.
            annotation_file: Path to JSON annotation file with entries:
                [{"hr_path": ..., "lr_path": ..., "text": ...}, ...] (preprocessed)
                or [{"image_path": ..., "boxes": [...], "texts": [...]}, ...] (on_the_fly)
            mode: "preprocessed" for pre-cropped pairs, "on_the_fly" for runtime processing.
            image_height: Target height (48).
            image_width: Target width (480).
            degradation: Degradation pipeline for on_the_fly mode.
        """
        self.data_root = Path(data_root)
        self.mode = mode
        self.image_height = image_height
        self.image_width = image_width
        self.degradation = degradation or RealESRGANDegradation()
        self.cropper = TextCropper(image_height, image_width)

        # Load annotations
        if annotation_file and os.path.exists(annotation_file):
            with open(annotation_file, "r", encoding="utf-8") as f:
                self.samples = json.load(f)
        else:
            # Auto-discover: look for hr/ and lr/ subdirectories
            self.samples = self._auto_discover()

    def _auto_discover(self) -> List[Dict]:
        """Auto-discover paired samples from directory structure.

        Expected structure:
            data_root/
                hr/       # High-resolution text crops
                lr/       # Low-resolution text crops
                texts.json  # {"filename": "text content", ...}
        """
        samples = []
        hr_dir = self.data_root / "hr"
        lr_dir = self.data_root / "lr"
        texts_file = self.data_root / "texts.json"

        texts = {}
        if texts_file.exists():
            with open(texts_file, "r", encoding="utf-8") as f:
                texts = json.load(f)

        if hr_dir.exists() and lr_dir.exists():
            for hr_path in sorted(hr_dir.glob("*.png")):
                name = hr_path.stem
                lr_path = lr_dir / hr_path.name
                if lr_path.exists():
                    samples.append({
                        "hr_path": str(hr_path),
                        "lr_path": str(lr_path),
                        "text": texts.get(name, ""),
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_and_preprocess(self, path: str) -> np.ndarray:
        """Load image file and resize/pad to target dimensions."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            # Return blank image as fallback
            return np.zeros(
                (self.image_height, self.image_width, 3), dtype=np.uint8
            )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]

        # Resize height to target, maintain aspect ratio
        if h != self.image_height:
            scale = self.image_height / h
            new_w = int(w * scale)
            img = cv2.resize(img, (new_w, self.image_height),
                             interpolation=cv2.INTER_LINEAR)

        # Pad or crop width
        _, curr_w = img.shape[:2]
        if curr_w < self.image_width:
            # Right-pad
            padded = np.zeros(
                (self.image_height, self.image_width, 3), dtype=np.uint8
            )
            padded[:, :curr_w] = img
            img = padded
        elif curr_w > self.image_width:
            img = img[:, :self.image_width]

        return img

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert uint8 (H,W,3) numpy to float32 (3,H,W) tensor in [-1,1]."""
        img = img.astype(np.float32) / 127.5 - 1.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        return torch.from_numpy(img)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        sample = self.samples[idx]

        if self.mode == "preprocessed":
            # Load pre-processed paired crops
            hr = self._load_and_preprocess(sample["hr_path"])
            lr = self._load_and_preprocess(sample["lr_path"])
            text = sample.get("text", "")

        elif self.mode == "on_the_fly":
            # Load full image + apply degradation
            img_path = sample.get("image_path", sample.get("hr_path"))
            img = cv2.imread(str(self.data_root / img_path), cv2.IMREAD_COLOR)

            if img is None:
                # Fallback to blank
                hr = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                lr = hr.copy()
                text = ""
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if "boxes" in sample:
                    # Crop a random text region
                    boxes = [np.array(b, dtype=np.float32) for b in sample["boxes"]]
                    texts = sample.get("texts", [""] * len(boxes))

                    idx_box = random.randint(0, len(boxes) - 1)
                    crop_result = self.cropper.crop_text_region(img, boxes[idx_box][:3])
                    hr_crop, _ = crop_result

                    if hr_crop is None:
                        hr = np.zeros((self.image_height, self.image_width, 3),
                                      dtype=np.uint8)
                    else:
                        hr = self.cropper.pad_to_target_width(hr_crop)

                    text = texts[idx_box] if idx_box < len(texts) else ""
                else:
                    # Assume already a text crop
                    hr = self._load_and_preprocess(str(self.data_root / img_path))
                    text = sample.get("text", "")

                # Apply degradation to get LR
                lr = self.degradation(hr)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return {
            "hr_image": self._to_tensor(hr),
            "lr_image": self._to_tensor(lr),
            "text": text,
        }


class SyntheticTextSRDataset(Dataset):
    """Synthetic dataset for debugging/testing the training pipeline.

    Generates random 48×480 image pairs with simple text-like patterns.
    """

    def __init__(self, num_samples: int = 1000, image_height: int = 48,
                 image_width: int = 480):
        self.num_samples = num_samples
        self.h = image_height
        self.w = image_width

        # Pre-generate some random texts
        self.texts = [
            f"sample_text_{i}" for i in range(num_samples)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, any]:
        # Create a simple synthetic HR image with text-like patterns
        hr = np.random.randint(180, 255, (self.h, self.w, 3), dtype=np.uint8)

        # Add some dark horizontal lines (simulate text)
        for y in range(10, self.h - 10, 8):
            thickness = random.randint(1, 3)
            x_start = random.randint(5, 50)
            x_end = random.randint(self.w - 100, self.w - 5)
            cv2.line(hr, (x_start, y), (x_end, y), (30, 30, 30), thickness)

        # Create degraded LR (simple blur + noise for synthetic)
        lr = cv2.GaussianBlur(hr, (5, 5), 1.5)
        noise = np.random.randn(*lr.shape).astype(np.float32) * 10
        lr = np.clip(lr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        hr_tensor = hr.astype(np.float32) / 127.5 - 1.0
        lr_tensor = lr.astype(np.float32) / 127.5 - 1.0

        return {
            "hr_image": torch.from_numpy(hr_tensor.transpose(2, 0, 1)),
            "lr_image": torch.from_numpy(lr_tensor.transpose(2, 0, 1)),
            "text": self.texts[idx],
        }
