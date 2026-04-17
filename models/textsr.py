"""
TextSR: Full model combining ByT5 encoder + U-Net + Flow Matching.

Implements:
  1. Training forward pass with CFG (random text dropout)
  2. Inference with Euler ODE + iterative OCR refinement (Algorithm from paper)

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
from models.flow_matching import FlowMatching
from models.unet import UNet


class TextSR(nn.Module):
    """
    TextSR model: flow-matching-based text image super-resolution with OCR conditioning.

    Components:
      - ByT5 text encoder (frozen)
      - U-Net velocity predictor (trainable)
      - Conditional Flow Matching scheduler
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

        # --- U-Net velocity predictor (trainable) ---
        self.unet = UNet(
            in_channels=6,              # 3 (x_t residual) + 3 (LR upsampled)
            out_channels=3,             # predicted velocity v = x_1 - eps
            base_channels=unet_base_channels,
            channel_mult=unet_channel_mult,
            num_res_blocks=unet_num_res_blocks,
            attention_levels=unet_attention_levels,
            text_context_dim=text_context_dim,
            time_embed_dim=unet_time_embed_dim,
            dropout=unet_dropout,
        )

        # --- Flow Matching scheduler ---
        self.flow = FlowMatching()

        # Tokenizer (shared with encoder)
        self.tokenizer = get_byt5_tokenizer(byt5_model)

    def encode_text(
        self,
        texts: List[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and encode text strings to ByT5 embeddings."""
        input_ids, attention_mask = tokenize_text(
            texts, self.tokenizer, self.max_text_len, device
        )
        with torch.no_grad():
            text_emb = self.text_encoder(input_ids, attention_mask)
        return text_emb, attention_mask

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def forward(
        self,
        lr_up: torch.Tensor,     # (B, 3, H, W) LR upsampled, [-1, 1]
        residual: torch.Tensor,  # (B, 3, H, W) = (HR - LR_up) * 0.5, in [-1, 1]
        text_ids: torch.Tensor,  # (B, max_text_len) token ids
        text_mask: torch.Tensor, # (B, max_text_len) attention mask
    ) -> torch.Tensor:
        """
        Flow matching training loss.

        When text is dropped (text_drop_prob), the dataset passes tokenized ""
        so cross-attention always runs (with EOS-only context), matching the
        training distribution for the CFG unconditioned pass.
        """
        with torch.no_grad():
            text_emb = self.text_encoder(text_ids, text_mask)  # (B, L, 1536)

        return self.flow.training_loss(
            model=self.unet,
            x1=residual,
            image_cond=lr_up,
            text_emb=text_emb,
            text_mask=text_mask,
        )

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def super_resolve(
        self,
        lr_up: torch.Tensor,                      # (B, 3, H, W) LR bicubic-upsampled
        texts: Optional[List[str]] = None,        # OCR text conditions
        cfg_weight: float = 2.0,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Single SR pass via Euler ODE integration.

        Returns:
            hr_pred: (B, 3, H, W) super-resolved image in [-1, 1]
        """
        B, _, H, W = lr_up.shape
        device = lr_up.device

        # Null text embedding — encode "" to match training-time null condition
        # (training drops text by tokenizing "", so cross-attn ran with EOS+PAD)
        null_emb, null_mask = self.encode_text([""] * B, device)

        if texts is not None:
            text_emb, text_mask = self.encode_text(texts, device)
        else:
            # Image-only SR: use null text (in-distribution)
            text_emb, text_mask = null_emb, null_mask

        residual_pred = self.flow.sample(
            model=self.unet,
            shape=(B, 3, H, W),
            image_cond=lr_up,
            text_emb=text_emb,
            text_mask=text_mask,
            null_text_emb=null_emb,
            null_text_mask=null_mask,
            cfg_weight=cfg_weight,
            num_steps=num_steps,
        )

        # HR = LR_up + 2 * residual_pred
        # (factor 2 undoes the *0.5 normalization in the dataset)
        hr_pred = lr_up + residual_pred * 2.0
        return hr_pred.clamp(-1, 1)

    @torch.no_grad()
    def super_resolve_iterative(
        self,
        lr_np: np.ndarray,                                      # (H, W, 3) uint8 RGB
        ocr_fn: Optional[Callable[[np.ndarray], str]] = None,
        sr_factor: int = 2,
        cfg_weight: float = 2.0,
        num_steps: int = 10,
        num_rounds: int = 1,             # R in paper
        device: torch.device = None,
    ) -> np.ndarray:
        """
        Iterative OCR-refinement SR (Algorithm from paper).

        R=0: g(c_i, ψ(c_i))           — OCR on LR, then SR
        R=1: g(c_i, ψ(g(c_i, ∅)))    — image-only SR → OCR → conditioned SR
        R>1: continue iterating

        Returns uint8 RGB numpy array (H*sr, W*sr, 3).
        """
        if device is None:
            device = next(self.parameters()).device
        if ocr_fn is None:
            ocr_fn = lambda img: ""

        H, W = lr_np.shape[:2]
        hr_h, hr_w = H * sr_factor, W * sr_factor

        lr_up_np = cv2.resize(lr_np, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
        lr_up_t = _numpy_to_tensor(lr_up_np).unsqueeze(0).to(device)

        if num_rounds == 0:
            text = ocr_fn(lr_np)
            hr_pred = self.super_resolve(lr_up_t, [text], cfg_weight, num_steps)
        else:
            sr_np = _tensor_to_numpy(
                self.super_resolve(lr_up_t, None, cfg_weight=1.0, num_steps=num_steps)[0]
            )
            hr_pred = None
            for _ in range(num_rounds):
                text = ocr_fn(sr_np)
                hr_pred = self.super_resolve(lr_up_t, [text], cfg_weight, num_steps)
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
    """Build TextSR (Flow Matching) model from OmegaConf config."""
    model_cfg = cfg.model
    unet_cfg = model_cfg.unet

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
    )
