"""
TextSR: Full model combining ByT5 encoder + U-Net diffusion.

Implements:
  1. Training forward pass with CFG (random text dropout)
  2. Inference with DDIM + iterative OCR refinement (Algorithm in paper)
  3. Blending output with Real-ESRGAN for low-frequency components

Paper inference procedure:
  R=0: g(c_i, ψ(c_i))            -- OCR on LR, SR with that text
  R=1: g(c_i, ψ(g(c_i, ∅)))     -- image-only SR first, then OCR on that
  R>1: iterative refinement
"""

from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from models.byt5_encoder import ByT5TextEncoder, get_byt5_tokenizer, tokenize_text
from models.diffusion import GaussianDiffusion
from models.unet import UNet


class TextSR(nn.Module):
    """
    TextSR model: diffusion-based text image super-resolution with OCR conditioning.

    Components:
      - ByT5 text encoder (frozen)
      - U-Net denoiser (trainable)
      - Gaussian diffusion scheduler
    """

    def __init__(
        self,
        # ByT5 config
        byt5_model: str = "google/byt5-base",
        byt5_num_layers: int = 2,
        max_text_len: int = 64,
        # U-Net config
        unet_base_channels: int = 32,
        unet_channel_mult: Tuple[int, ...] = (1, 2, 4, 8, 8),
        unet_num_res_blocks: int = 2,
        unet_attention_levels: Tuple[int, ...] = (3, 4),
        unet_dropout: float = 0.0,
        unet_time_embed_dim: int = 256,
        # Diffusion config
        diffusion_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        prediction_type: str = "epsilon",
    ):
        super().__init__()
        self.max_text_len = max_text_len

        # --- Text encoder (frozen) ---
        self.text_encoder = ByT5TextEncoder(
            model_name=byt5_model,
            num_layers=byt5_num_layers,
            max_length=max_text_len,
        )
        text_context_dim = self.text_encoder.hidden_size  # 1536

        # --- U-Net (trainable) ---
        self.unet = UNet(
            in_channels=6,              # 3 noisy residual + 3 LR upsampled
            out_channels=3,
            base_channels=unet_base_channels,
            channel_mult=unet_channel_mult,
            num_res_blocks=unet_num_res_blocks,
            attention_levels=unet_attention_levels,
            text_context_dim=text_context_dim,
            time_embed_dim=unet_time_embed_dim,
            dropout=unet_dropout,
        )

        # --- Diffusion ---
        self.diffusion = GaussianDiffusion(
            timesteps=diffusion_timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            prediction_type=prediction_type,
        )

        # Tokenizer (shared with encoder)
        self.tokenizer = get_byt5_tokenizer(byt5_model)

    def encode_text(
        self,
        texts: List[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and encode text strings."""
        input_ids, attention_mask = tokenize_text(
            texts, self.tokenizer, self.max_text_len, device
        )
        with torch.no_grad():
            text_emb = self.text_encoder(input_ids, attention_mask)
        return text_emb, attention_mask

    def get_null_text_emb(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get null text embedding for CFG (all-PAD tokens)."""
        null_ids, null_mask = self.text_encoder.get_null_embedding(batch_size, device)
        with torch.no_grad():
            null_emb = self.text_encoder(null_ids, null_mask)
        return null_emb, null_mask

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def forward(
        self,
        lr_up: torch.Tensor,     # (B, 3, H, W) LR upsampled, [-1, 1]
        residual: torch.Tensor,  # (B, 3, H, W) HR - LR_up (normalized to [-1,1])
        text_ids: torch.Tensor,  # (B, max_text_len) token ids
        text_mask: torch.Tensor, # (B, max_text_len) attention mask
    ) -> torch.Tensor:
        """
        Training forward pass.

        Returns:
            diffusion loss (scalar)
        """
        device = lr_up.device

        # Encode text (frozen encoder, no grad needed)
        with torch.no_grad():
            text_emb = self.text_encoder(text_ids, text_mask)  # (B, L, 1536)

        # Where text_mask is all zeros (text dropped), set text_emb to None
        # Actually, we pass the embeddings but with the mask zeros → cross-attn ignores them
        # This is equivalent to null conditioning for CFG training.

        # Compute diffusion loss
        loss = self.diffusion.training_loss(
            model=self.unet,
            x0=residual,
            image_cond=lr_up,
            text_emb=text_emb,
            text_mask=text_mask,
        )
        return loss

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def super_resolve(
        self,
        lr_up: torch.Tensor,                      # (B, 3, H, W) LR bicubic-upsampled
        texts: Optional[List[str]] = None,        # OCR text conditions
        cfg_weight: float = 2.0,
        ddim_steps: int = 5,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Single SR pass given LR upsampled and optional text strings.

        Returns:
            hr_pred: (B, 3, H, W) super-resolved image in [-1, 1]
        """
        B, C, H, W = lr_up.shape
        device = lr_up.device

        # Encode text if provided
        if texts is not None:
            text_emb, text_mask = self.encode_text(texts, device)
        else:
            text_emb, text_mask = None, None

        # DDIM sampling
        residual_pred = self.diffusion.ddim_sample(
            model=self.unet,
            shape=(B, 3, H, W),
            image_cond=lr_up,
            text_emb=text_emb,
            text_mask=text_mask,
            cfg_weight=cfg_weight,
            num_steps=ddim_steps,
            eta=eta,
        )

        # Reconstruct HR: HR = LR_up + 2 * residual_pred
        # (factor 2 undoes the *0.5 normalization applied during training)
        hr_pred = lr_up + residual_pred * 2.0
        hr_pred = hr_pred.clamp(-1, 1)
        return hr_pred

    @torch.no_grad()
    def super_resolve_iterative(
        self,
        lr_np: np.ndarray,                     # (H, W, 3) uint8 LR image
        ocr_fn: Optional[Callable[[np.ndarray], str]] = None,  # OCR function
        sr_factor: int = 2,
        cfg_weight: float = 2.0,
        ddim_steps: int = 5,
        num_rounds: int = 1,                   # R in paper
        device: torch.device = None,
    ) -> np.ndarray:
        """
        Iterative OCR refinement SR (Algorithm from paper).

        R=0: Direct SR conditioned on LR OCR text
          g(c_i, ψ(c_i))

        R=1: Image-only SR first, then OCR on that, then conditioned SR
          step1: g(c_i, ∅)  → intermediate SR
          step2: g(c_i, ψ(intermediate SR))  → final SR

        R>1: Continue iterating

        Args:
            lr_np:      LR image as uint8 numpy array (H, W, 3) RGB
            ocr_fn:     OCR function that takes numpy image and returns text string
            sr_factor:  upscale factor
            cfg_weight: ω (text guidance scale)
            ddim_steps: DDIM inference steps
            num_rounds: R (0=direct, 1=one refinement, etc.)
            device:     torch device

        Returns:
            HR image as uint8 numpy array (H*sr, W*sr, 3) RGB
        """
        if device is None:
            device = next(self.parameters()).device

        H, W = lr_np.shape[:2]
        hr_h, hr_w = H * sr_factor, W * sr_factor

        # Bicubic upsample LR to HR size
        lr_up_np = cv2.resize(lr_np, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
        lr_up_t = _numpy_to_tensor(lr_up_np).unsqueeze(0).to(device)  # (1, 3, H, W)

        # OCR function (default: return empty string)
        if ocr_fn is None:
            ocr_fn = lambda img: ""

        if num_rounds == 0:
            # R=0: OCR on original LR, then SR
            text = ocr_fn(lr_np) if ocr_fn is not None else ""
            hr_pred = self.super_resolve(lr_up_t, [text], cfg_weight, ddim_steps)
        else:
            # R>=1: First do image-only SR, then iterate
            # Step 0: image-only SR (no text)
            sr_np = _tensor_to_numpy(
                self.super_resolve(lr_up_t, None, cfg_weight=1.0, ddim_steps=ddim_steps)[0]
            )

            # Iterative refinement
            hr_pred = self.super_resolve(lr_up_t, None, cfg_weight=1.0, ddim_steps=ddim_steps)
            for _ in range(num_rounds):
                text = ocr_fn(sr_np) if ocr_fn is not None else ""
                hr_pred = self.super_resolve(lr_up_t, [text], cfg_weight, ddim_steps)
                sr_np = _tensor_to_numpy(hr_pred[0])

        return _tensor_to_numpy(hr_pred[0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numpy_to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HWC [0,255] -> float CHW [-1,1]"""
    x = img.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(x).permute(2, 0, 1)


def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """float CHW [-1,1] -> uint8 HWC [0,255]"""
    img = (t.cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_textsr_from_config(cfg) -> TextSR:
    """Build TextSR model from OmegaConf config."""
    model_cfg = cfg.model
    unet_cfg = model_cfg.unet
    diff_cfg = model_cfg.diffusion

    return TextSR(
        byt5_model=model_cfg.byt5_model,
        byt5_num_layers=model_cfg.byt5_num_layers,
        max_text_len=model_cfg.max_text_len,
        unet_base_channels=unet_cfg.base_channels,
        unet_channel_mult=tuple(unet_cfg.channel_mult),
        unet_num_res_blocks=unet_cfg.num_res_blocks,
        unet_attention_levels=tuple(unet_cfg.attention_levels),
        unet_dropout=unet_cfg.dropout,
        unet_time_embed_dim=unet_cfg.time_embed_dim,
        diffusion_timesteps=diff_cfg.timesteps,
        beta_schedule=diff_cfg.beta_schedule,
        beta_start=diff_cfg.beta_start,
        beta_end=diff_cfg.beta_end,
        prediction_type=diff_cfg.prediction_type,
    )
