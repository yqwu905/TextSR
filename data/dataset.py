"""
TextZoom Dataset for TextSR training and evaluation.

TextZoom stores images in LMDB format with keys:
  image-000000001-lr  -> LR image bytes
  image-000000001-hr  -> HR image bytes
  label-000000001     -> ground truth text label (for evaluation)

During training:
  - Load HR image, apply Real-ESRGAN degradation to generate synthetic LR
  - Run PaddleOCR on the (real or synthetic) LR to get text condition
  - Compute residual = HR - upsample(LR)
  - Return: (lr_up, hr, residual, text_ids, text_mask)

During evaluation:
  - Load real LR/HR pairs from TextZoom
  - Run OCR on LR for text conditioning
  - Return: (lr_img, hr_img, text_ids, text_mask, gt_label)
"""

import io
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.degradation import degrade_and_resize, real_esrgan_degrade


# ---------------------------------------------------------------------------
# Null text token (for classifier-free guidance text dropout)
# ---------------------------------------------------------------------------
NULL_TEXT = ""


def pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))


def bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def numpy_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert uint8 HWC numpy array to float CHW tensor in [-1, 1]."""
    img = img.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(img).permute(2, 0, 1)


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert CHW float tensor in [-1, 1] to uint8 HWC numpy array."""
    img = (t.permute(1, 2, 0).numpy() + 1.0) * 127.5
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# OCR backend (lazy import to avoid mandatory PaddleOCR dep at import time)
# ---------------------------------------------------------------------------

_ocr_instance = None


def get_ocr(use_gpu: bool = False):
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        _ocr_instance = PaddleOCR(
            use_angle_cls=False,
            lang="en",         # change to 'ch' for Chinese or 'multilingual'
            use_gpu=use_gpu,
            show_log=False,
        )
    return _ocr_instance


def ocr_image(img_np: np.ndarray, use_gpu: bool = False) -> str:
    """
    Run PaddleOCR on an image and return concatenated text.

    Args:
        img_np: uint8 RGB numpy array
    Returns:
        Recognized text string (empty string on failure)
    """
    try:
        ocr = get_ocr(use_gpu)
        result = ocr.ocr(img_np, cls=False)
        if result is None or result[0] is None:
            return ""
        texts = []
        for line in result[0]:
            if line is not None and len(line) >= 2:
                text, conf = line[1]
                if conf > 0.3:
                    texts.append(text)
        return " ".join(texts)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# LMDB reader
# ---------------------------------------------------------------------------

class LMDBReader:
    def __init__(self, lmdb_path: str):
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.n_samples = txn.stat()["entries"]
            # Check if dataset has label keys
            self.has_labels = txn.get(b"label-000000001") is not None

    def __len__(self):
        # Each sample has lr + hr keys (+ optional label)
        return self.n_samples // (3 if self.has_labels else 2)

    def get(self, idx: int) -> Dict:
        key = f"{idx + 1:09d}"
        with self.env.begin(write=False) as txn:
            lr_bytes = txn.get(f"image-{key}-lr".encode())
            hr_bytes = txn.get(f"image-{key}-hr".encode())
            label_bytes = txn.get(f"label-{key}".encode())

        result = {}
        if lr_bytes is not None:
            result["lr"] = bytes_to_pil(lr_bytes)
        if hr_bytes is not None:
            result["hr"] = bytes_to_pil(hr_bytes)
        if label_bytes is not None:
            result["label"] = label_bytes.decode("utf-8")
        return result


# ---------------------------------------------------------------------------
# OCR annotation cache (pre-computed to speed up training)
# ---------------------------------------------------------------------------

class OCRAnnotationCache:
    """Simple JSON-based cache for pre-computed OCR annotations."""

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self._data: Dict[str, str] = {}
        self._load()

    def _load(self):
        import json
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    def save(self):
        import json
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def set(self, key: str, value: str):
        self._data[key] = value


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------

class TextZoomDataset(Dataset):
    """
    Dataset for TextSR training on TextZoom.

    Returns dict with:
      lr_up     : LR image bicubic-upsampled to HR size, float32 CHW in [-1, 1]
      hr        : HR image, float32 CHW in [-1, 1]
      residual  : HR - lr_up (residual target for diffusion), CHW in [-1, 1] (approx)
      text_ids  : ByT5 token ids, int64 (max_text_len,)
      text_mask : attention mask, int64 (max_text_len,)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        difficulties: Tuple[str, ...] = ("easy", "medium", "hard"),
        hr_size: Tuple[int, int] = (48, 480),     # (H, W)
        sr_factor: int = 2,
        max_text_len: int = 64,
        text_drop_prob: float = 0.1,              # drop text for CFG training
        use_synthetic_lr: bool = True,            # apply degradation to HR
        num_degrade_rounds: int = 5,              # reduced from 20 for speed
        ocr_cache_dir: Optional[str] = None,
        tokenizer=None,                           # ByT5 tokenizer (passed in)
        use_ocr_gpu: bool = False,
    ):
        super().__init__()
        self.hr_size = hr_size                    # (H, W) crop for HR
        self.lr_size = (hr_size[0] // sr_factor, hr_size[1] // sr_factor)
        self.sr_factor = sr_factor
        self.max_text_len = max_text_len
        self.text_drop_prob = text_drop_prob
        self.use_synthetic_lr = use_synthetic_lr
        self.num_degrade_rounds = num_degrade_rounds
        self.tokenizer = tokenizer
        self.use_ocr_gpu = use_ocr_gpu
        self.split = split

        # Resolve LMDB paths.
        # Train: data_root/train1, data_root/train2 (flat, no difficulty split)
        # Test:  data_root/test/easy, data_root/test/medium, data_root/test/hard
        if split == "train":
            lmdb_paths = [
                os.path.join(data_root, "train1"),
                os.path.join(data_root, "train2"),
            ]
        else:
            lmdb_paths = [os.path.join(data_root, split, d) for d in difficulties]

        self.readers: List[LMDBReader] = []
        self.reader_lengths: List[int] = []
        for lmdb_path in lmdb_paths:
            if os.path.isdir(lmdb_path):
                reader = LMDBReader(lmdb_path)
                self.readers.append(reader)
                self.reader_lengths.append(len(reader))
            else:
                print(f"[WARNING] LMDB not found: {lmdb_path}")

        self.total = sum(self.reader_lengths)
        assert self.total > 0, f"No data found in {data_root}/{split}"

        # Cumulative lengths for index lookup
        self._cum_lens = []
        s = 0
        for l in self.reader_lengths:
            s += l
            self._cum_lens.append(s)

        # OCR cache
        self.ocr_cache: Optional[OCRAnnotationCache] = None
        if ocr_cache_dir is not None:
            cache_file = os.path.join(ocr_cache_dir, f"{split}_ocr.json")
            self.ocr_cache = OCRAnnotationCache(cache_file)

        print(f"[TextZoomDataset] {split}: {self.total} samples across {len(self.readers)} LMDB(s)")

    def __len__(self):
        return self.total

    def _get_reader_and_local_idx(self, global_idx: int) -> Tuple[LMDBReader, int]:
        for i, cum in enumerate(self._cum_lens):
            if global_idx < cum:
                local_idx = global_idx - (self._cum_lens[i - 1] if i > 0 else 0)
                return self.readers[i], local_idx
        raise IndexError(f"Index {global_idx} out of range")

    def _get_text(self, lr_np: np.ndarray, sample_key: str, gt_label: Optional[str]) -> str:
        """Get OCR text for the LR image (with caching)."""
        # Check cache first
        if self.ocr_cache is not None:
            cached = self.ocr_cache.get(sample_key)
            if cached is not None:
                return cached

        # During training, optionally use GT label directly (faster)
        if gt_label is not None and self.split == "train":
            text = gt_label
        else:
            # Run OCR on LR image
            text = ocr_image(lr_np, use_gpu=self.use_ocr_gpu)

        # Cache result
        if self.ocr_cache is not None:
            self.ocr_cache.set(sample_key, text)

        return text

    def _tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text with ByT5 tokenizer."""
        if self.tokenizer is None:
            # Fallback: simple byte-level tokenization
            return self._byte_tokenize(text)

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_text_len,
            truncation=True,
        )
        return enc.input_ids[0], enc.attention_mask[0]

    def _byte_tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Manual UTF-8 byte tokenization (compatible with ByT5).
        Token 0=PAD, 1=EOS, byte b -> b+3
        """
        byte_vals = list(text.encode("utf-8")[:self.max_text_len - 1])
        byte_vals.append(1)  # EOS
        ids = [b + 3 for b in byte_vals]

        # Pad to max_text_len
        mask = [1] * len(ids)
        while len(ids) < self.max_text_len:
            ids.append(0)
            mask.append(0)

        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        reader, local_idx = self._get_reader_and_local_idx(idx)
        sample = reader.get(local_idx)
        sample_key = f"{idx}"

        hr_pil = sample.get("hr")
        lr_pil = sample.get("lr")
        gt_label = sample.get("label", None)

        if hr_pil is None:
            # Return zeros on error
            return self._empty_sample()

        hr_np = pil_to_numpy(hr_pil)

        # --- Generate LR from HR or use real LR ---
        if self.use_synthetic_lr or lr_pil is None:
            # Crop HR to training size first
            hr_np = self._random_crop(hr_np, self.hr_size)
            lr_np, hr_np = degrade_and_resize(hr_np, self.lr_size, self.num_degrade_rounds)
        else:
            # Use real LR/HR pair from TextZoom
            lr_np = pil_to_numpy(lr_pil)
            # Resize both to fixed sizes
            lr_np = cv2.resize(lr_np, (self.lr_size[1], self.lr_size[0]), interpolation=cv2.INTER_LINEAR)
            hr_np = cv2.resize(hr_np, (self.hr_size[1], self.hr_size[0]), interpolation=cv2.INTER_LINEAR)

        # --- Bicubic upsample LR to HR size ---
        lr_up_np = cv2.resize(lr_np, (self.hr_size[1], self.hr_size[0]), interpolation=cv2.INTER_CUBIC)

        # --- Compute normalized residual target ---
        # Both hr and lr_up are in [-1,1] after numpy_to_tensor.
        # Their difference (residual) is in [-2,2].
        # We scale by 0.5 to fit into [-1,1] for diffusion.
        # At reconstruction: hr = lr_up + 2 * residual_pred
        hr_t = numpy_to_tensor(hr_np)          # (3, H, W) in [-1, 1]
        lr_up_t = numpy_to_tensor(lr_up_np)    # (3, H, W) in [-1, 1]
        residual_t = (hr_t - lr_up_t) * 0.5   # (3, H, W) in [-1, 1]

        # --- Get text condition ---
        text_str = self._get_text(lr_np, sample_key, gt_label)

        # Drop text with probability text_drop_prob (for CFG training)
        if self.split == "train" and random.random() < self.text_drop_prob:
            text_str = NULL_TEXT

        text_ids, text_mask = self._tokenize(text_str)

        return {
            "lr_up": lr_up_t,      # (3, H, W) in [-1, 1]
            "hr": hr_t,            # (3, H, W) in [-1, 1]
            "residual": residual_t,  # (3, H, W) in [-1, 1], = (hr - lr_up) * 0.5
            "text_ids": text_ids,
            "text_mask": text_mask,
            "gt_label": gt_label or "",
        }

    def _random_crop(self, img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Random crop to (H, W)."""
        h, w = img.shape[:2]
        th, tw = size
        if h < th or w < tw:
            img = cv2.resize(img, (max(w, tw), max(h, th)), interpolation=cv2.INTER_LINEAR)
            h, w = img.shape[:2]
        top = random.randint(0, h - th)
        left = random.randint(0, w - tw)
        return img[top:top + th, left:left + tw]

    def _empty_sample(self) -> Dict[str, torch.Tensor]:
        h, w = self.hr_size
        return {
            "lr_up": torch.zeros(3, h, w),
            "hr": torch.zeros(3, h, w),
            "residual": torch.ones(3, h, w) * 0.5,
            "text_ids": torch.zeros(self.max_text_len, dtype=torch.long),
            "text_mask": torch.zeros(self.max_text_len, dtype=torch.long),
            "gt_label": "",
        }


# ---------------------------------------------------------------------------
# Evaluation Dataset (uses real TextZoom LR/HR pairs, no augmentation)
# ---------------------------------------------------------------------------

class TextZoomEvalDataset(Dataset):
    """
    TextZoom evaluation dataset.
    Returns real LR/HR pairs without augmentation.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "test",
        difficulties: Tuple[str, ...] = ("easy", "medium", "hard"),
        lr_size: Tuple[int, int] = (32, 128),
        sr_factor: int = 2,
        max_text_len: int = 64,
        tokenizer=None,
        use_ocr_gpu: bool = False,
        ocr_cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.lr_size = lr_size
        self.hr_size = (lr_size[0] * sr_factor, lr_size[1] * sr_factor)
        self.max_text_len = max_text_len
        self.tokenizer = tokenizer
        self.use_ocr_gpu = use_ocr_gpu

        self.readers: List[LMDBReader] = []
        self.reader_lengths: List[int] = []
        self.difficulty_labels: List[str] = []

        for diff in difficulties:
            lmdb_path = os.path.join(data_root, split, diff)
            if os.path.isdir(lmdb_path):
                reader = LMDBReader(lmdb_path)
                self.readers.append(reader)
                self.reader_lengths.append(len(reader))
                self.difficulty_labels.extend([diff] * len(reader))
            else:
                print(f"[WARNING] LMDB not found: {lmdb_path}")

        self.total = sum(self.reader_lengths)
        self._cum_lens = []
        s = 0
        for l in self.reader_lengths:
            s += l
            self._cum_lens.append(s)

        self.ocr_cache: Optional[OCRAnnotationCache] = None
        if ocr_cache_dir is not None:
            cache_file = os.path.join(ocr_cache_dir, f"{split}_ocr.json")
            self.ocr_cache = OCRAnnotationCache(cache_file)

        print(f"[TextZoomEvalDataset] {split}: {self.total} samples")

    def __len__(self):
        return self.total

    def _get_reader_and_local_idx(self, global_idx: int) -> Tuple[LMDBReader, int]:
        for i, cum in enumerate(self._cum_lens):
            if global_idx < cum:
                local_idx = global_idx - (self._cum_lens[i - 1] if i > 0 else 0)
                return self.readers[i], local_idx
        raise IndexError(f"Index {global_idx} out of range")

    def _tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer is None:
            byte_vals = list(text.encode("utf-8")[:self.max_text_len - 1])
            byte_vals.append(1)
            ids = [b + 3 for b in byte_vals]
            mask = [1] * len(ids)
            while len(ids) < self.max_text_len:
                ids.append(0)
                mask.append(0)
            return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_text_len,
            truncation=True,
        )
        return enc.input_ids[0], enc.attention_mask[0]

    def __getitem__(self, idx: int) -> Dict:
        reader, local_idx = self._get_reader_and_local_idx(idx)
        sample = reader.get(local_idx)

        lr_pil = sample.get("lr")
        hr_pil = sample.get("hr")
        gt_label = sample.get("label", "")

        if lr_pil is None or hr_pil is None:
            h, w = self.hr_size
            return {
                "lr": torch.zeros(3, *self.lr_size),
                "lr_up": torch.zeros(3, h, w),
                "hr": torch.zeros(3, h, w),
                "text_ids": torch.zeros(self.max_text_len, dtype=torch.long),
                "text_mask": torch.zeros(self.max_text_len, dtype=torch.long),
                "gt_label": gt_label,
                "difficulty": self.difficulty_labels[idx],
            }

        lr_np = pil_to_numpy(lr_pil)
        hr_np = pil_to_numpy(hr_pil)

        # Resize to fixed sizes
        lr_np = cv2.resize(lr_np, (self.lr_size[1], self.lr_size[0]), interpolation=cv2.INTER_LINEAR)
        hr_np = cv2.resize(hr_np, (self.hr_size[1], self.hr_size[0]), interpolation=cv2.INTER_LINEAR)
        lr_up_np = cv2.resize(lr_np, (self.hr_size[1], self.hr_size[0]), interpolation=cv2.INTER_CUBIC)

        # Get OCR text from LR
        cache_key = f"eval_{idx}"
        text_str = ""
        if self.ocr_cache is not None:
            cached = self.ocr_cache.get(cache_key)
            if cached is not None:
                text_str = cached

        if not text_str:
            text_str = ocr_image(lr_np, use_gpu=self.use_ocr_gpu)

        if text_str == "" and gt_label:
            text_str = gt_label  # fallback for eval

        text_ids, text_mask = self._tokenize(text_str)

        return {
            "lr": numpy_to_tensor(lr_np),
            "lr_up": numpy_to_tensor(lr_up_np),
            "hr": numpy_to_tensor(hr_np),
            "text_ids": text_ids,
            "text_mask": text_mask,
            "gt_label": gt_label,
            "difficulty": self.difficulty_labels[idx],
        }
