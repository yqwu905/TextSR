"""Full TextSR Inference Pipeline.

Implements the complete inference flow from Section 3.3 of arXiv:2505.23119v1:

1. Text detection → extract bounding boxes
2. Affine crop text regions → 48×480 patches
3. OCR recognition → extract text content
4. ByT5 encoding → text features
5. DDIM sampling with CFG → super-resolved text patches
6. Residual addition → final SR text
7. Inverse affine + LPF blending → paste back to full image

Supports Iterative OCR Conditioning (R=0 or R=1):
- R=0: g^(0)(c_I) = g(c_I, ψ(c_I))  — direct OCR on LR
- R=1: g^(1)(c_I) = g(c_I, ψ(g(c_I, ∅)))  — OCR on image-only pre-enhanced
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from ..models.textsr_model import TextSRModel
from ..diffusion.ddpm import GaussianDiffusion
from ..diffusion.ddim import DDIMSampler
from ..data.text_crop import TextCropper
from .blending import TextSRBlender


class TextSRPipeline:
    """End-to-end TextSR inference pipeline.

    Usage:
        pipeline = TextSRPipeline.from_checkpoint("checkpoint.pt")
        result = pipeline.enhance_image(image, scale_factor=2)
    """

    def __init__(
        self,
        model: TextSRModel,
        diffusion: GaussianDiffusion,
        ddim_steps: int = 5,
        omega: float = 1.0,
        iterative_rounds: int = 1,      # R parameter
        blending_sigma: float = 3.0,
        image_height: int = 48,
        image_width: int = 480,
        device: torch.device = None,
    ):
        """
        Args:
            model: Trained TextSR model.
            diffusion: GaussianDiffusion instance.
            ddim_steps: Number of DDIM sampling steps (paper uses 5).
            omega: CFG guidance scale.
            iterative_rounds: R for iterative OCR conditioning.
            blending_sigma: Gaussian sigma for LPF blending.
            image_height: Standard crop height (48).
            image_width: Standard crop width (480).
            device: Torch device.
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device).eval()
        self.diffusion = diffusion.to(self.device)
        self.sampler = DDIMSampler(diffusion, num_steps=ddim_steps)
        self.omega = omega
        self.iterative_rounds = iterative_rounds
        self.cropper = TextCropper(image_height, image_width)
        self.blender = TextSRBlender(sigma=blending_sigma)
        self.image_height = image_height
        self.image_width = image_width

        # OCR engine (lazy import)
        self._ocr_engine = None

    def _get_ocr_engine(self):
        """Lazy-load OCR engine."""
        if self._ocr_engine is None:
            try:
                import easyocr
                self._ocr_engine = easyocr.Reader(
                    ['en', 'ch_sim', 'ja', 'fr', 'hi'],
                    gpu=torch.cuda.is_available()
                )
            except ImportError:
                raise RuntimeError(
                    "EasyOCR is required for inference. "
                    "Install with: pip install easyocr"
                )
        return self._ocr_engine

    def _detect_text_regions(
        self,
        image: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Detect text regions and recognize text in the image.

        Args:
            image: (H, W, 3) uint8 RGB image.

        Returns:
            boxes: List of (4, 2) float32 polygon coordinates.
            texts: List of recognized text strings.
        """
        ocr = self._get_ocr_engine()
        results = ocr.readtext(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        boxes = []
        texts = []
        for (bbox, text, conf) in results:
            pts = np.array(bbox, dtype=np.float32)
            if pts.shape[0] == 4:
                boxes.append(pts)
                texts.append(text)

        return boxes, texts

    def _ocr_text(self, image: np.ndarray) -> str:
        """Run OCR on a single text crop image.

        Args:
            image: (H, W, 3) uint8 RGB image.

        Returns:
            Recognized text string.
        """
        ocr = self._get_ocr_engine()
        results = ocr.readtext(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # Concatenate all recognized text
        return " ".join([text for (_, text, _) in results])

    def _preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """Convert uint8 (H,W,3) crop to model input tensor."""
        # Ensure correct size
        h, w = crop.shape[:2]
        if h != self.image_height:
            scale = self.image_height / h
            new_w = int(w * scale)
            crop = cv2.resize(crop, (new_w, self.image_height))

        if crop.shape[1] < self.image_width:
            padded = np.zeros(
                (self.image_height, self.image_width, 3), dtype=np.uint8
            )
            padded[:, :crop.shape[1]] = crop
            crop = padded
        elif crop.shape[1] > self.image_width:
            crop = crop[:, :self.image_width]

        # Normalize to [-1, 1]
        tensor = crop.astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(tensor.transpose(2, 0, 1))  # CHW
        return tensor.unsqueeze(0).to(self.device)

    def _postprocess_output(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert model output tensor back to uint8 image."""
        img = tensor.squeeze(0).cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # CHW → HWC
        img = (img + 1.0) * 127.5
        return np.clip(img, 0, 255).astype(np.uint8)

    @torch.no_grad()
    def super_resolve_crop(
        self,
        lr_crop: np.ndarray,
        text: Optional[str] = None,
        omega: Optional[float] = None,
    ) -> np.ndarray:
        """Super-resolve a single text crop.

        Args:
            lr_crop: (48, W, 3) uint8 low-resolution text crop.
            text: UTF-8 text content. None for image-only mode.
            omega: CFG guidance scale override.

        Returns:
            (48, W, 3) uint8 super-resolved text crop.
        """
        omega = omega if omega is not None else self.omega
        c_I = self._preprocess_crop(lr_crop)

        # Encode text
        text_features = None
        if text is not None and text.strip():
            text_features = self.model.text_encoder(texts=[text])

        # DDIM sampling → predicts residual
        residual = self.sampler.sample(
            model=self.model,
            c_I=c_I,
            text_features=text_features,
            omega=omega,
        )

        # Add residual to LR to get SR
        sr = c_I + residual
        return self._postprocess_output(sr)

    @torch.no_grad()
    def enhance_image(
        self,
        image: np.ndarray,
        esrgan_image: Optional[np.ndarray] = None,
        scale_factor: int = 1,
        texts: Optional[List[str]] = None,
        boxes: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Full-image enhancement pipeline.

        Args:
            image: (H, W, 3) uint8 RGB input image.
            esrgan_image: (H*s, W*s, 3) uint8 Real-ESRGAN upscaled image.
                         If None, uses the input image as base.
            scale_factor: Upscaling factor (1, 2, or 4).
            texts: Pre-recognized text for each region. If None, runs OCR.
            boxes: Pre-detected text regions. If None, runs detection.

        Returns:
            (H_out, W_out, 3) uint8 enhanced image.
        """
        # Step 1: Detect text regions if not provided
        if boxes is None:
            boxes, detected_texts = self._detect_text_regions(image)
            if texts is None:
                texts = detected_texts
        elif texts is None:
            texts = [""] * len(boxes)

        # Base image for blending
        if esrgan_image is None:
            base = image.copy()
        else:
            base = esrgan_image.copy()

        base_float = base.astype(np.float32)

        # Step 2: Process each text region
        textsr_crops = []
        esrgan_crops = []
        inverse_affines = []
        crop_widths = []

        for box, text in zip(boxes, texts):
            # Crop text region from LR image
            src_pts = box[:3].astype(np.float32)
            crop, affine = self.cropper.crop_text_region(image, src_pts)
            if crop is None:
                continue

            orig_w = crop.shape[1]
            padded_crop = self.cropper.pad_to_target_width(crop)

            # Iterative OCR Conditioning
            if self.iterative_rounds >= 1:
                # R=1: First run image-only to get better input for OCR
                sr_image_only = self.super_resolve_crop(
                    padded_crop, text=None, omega=0.0
                )
                # Re-run OCR on the enhanced image
                text = self._ocr_text(sr_image_only)

            # Final super-resolution with text guidance
            sr_crop = self.super_resolve_crop(
                padded_crop, text=text, omega=self.omega
            )
            textsr_crops.append(sr_crop.astype(np.float32))

            # Get corresponding Real-ESRGAN crop for blending
            if esrgan_image is not None:
                inv_affine = self.cropper.compute_inverse_affine(
                    affine, scale_factor
                )
                # Forward crop from esrgan_image
                fwd_affine = cv2.getAffineTransform(
                    src_pts * scale_factor,
                    np.array([[0, 0], [0, self.image_height],
                              [orig_w, self.image_height]], dtype=np.float32),
                )
                esrgan_crop = cv2.warpAffine(
                    esrgan_image, fwd_affine,
                    (orig_w, self.image_height),
                )
                esrgan_crop_padded = self.cropper.pad_to_target_width(esrgan_crop)
                esrgan_crops.append(esrgan_crop_padded.astype(np.float32))
            else:
                inv_affine = self.cropper.compute_inverse_affine(affine)
                esrgan_crops.append(padded_crop.astype(np.float32))

            inverse_affines.append(inv_affine)
            crop_widths.append(orig_w)

        # Step 3: Blend all text regions back to base image
        if textsr_crops:
            result = self.blender.blend_full_image(
                base_float, textsr_crops, esrgan_crops,
                inverse_affines, crop_widths,
                crop_height=self.image_height,
            )
        else:
            result = base_float

        return np.clip(result, 0, 255).astype(np.uint8)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None,
    ) -> "TextSRPipeline":
        """Load pipeline from a saved checkpoint.

        Args:
            checkpoint_path: Path to checkpoint .pt file.
            config: Config dict. If None, loaded from checkpoint.
            device: Target device.

        Returns:
            Initialized TextSRPipeline.
        """
        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if config is None:
            config = checkpoint.get("config", {})

        # Build model
        model = TextSRModel.from_config(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Build diffusion
        diff_cfg = config.get("diffusion", {})
        diffusion = GaussianDiffusion(
            num_timesteps=diff_cfg.get("num_timesteps", 1000),
            beta_start=diff_cfg.get("beta_start", 0.0001),
            beta_end=diff_cfg.get("beta_end", 0.02),
            beta_schedule=diff_cfg.get("beta_schedule", "linear"),
        )

        inf_cfg = config.get("inference", {})
        return cls(
            model=model,
            diffusion=diffusion,
            ddim_steps=inf_cfg.get("ddim_steps", 5),
            omega=inf_cfg.get("omega", 1.0),
            iterative_rounds=inf_cfg.get("iterative_ocr_rounds", 1),
            blending_sigma=inf_cfg.get("blending_sigma", 3.0),
            device=device,
        )
