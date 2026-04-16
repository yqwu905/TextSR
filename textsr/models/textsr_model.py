"""TextSR Top-Level Model.

Wraps together:
1. ByT5 Text Encoder (frozen)
2. U-Net Denoiser
3. Convenience methods for training and inference
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .unet import UNet
from .text_encoder import ByT5TextEncoder


class TextSRModel(nn.Module):
    """TextSR: Multimodal Diffusion Model for Scene Text Image Super-Resolution.

    This wrapper combines the frozen ByT5 text encoder with the trainable
    U-Net denoiser. It provides a clean interface for both training (noise
    prediction) and inference (producing the denoised output).
    """

    def __init__(
        self,
        # U-Net params
        base_channels: int = 32,
        channel_multipliers: List[int] = None,
        num_down_layers: int = 2,
        num_up_layers: int = 3,
        cross_attn_levels: List[int] = None,
        time_embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.0,
        # Text encoder params
        text_encoder_name: str = "google/byt5-base",
        text_encoder_layers: int = 2,
        text_embed_dim: int = 1536,
        max_text_length: int = 256,
    ):
        super().__init__()
        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 8, 8]
        if cross_attn_levels is None:
            cross_attn_levels = [3, 4]

        self.max_text_length = max_text_length
        self.text_embed_dim = text_embed_dim

        # --- Frozen text encoder ---
        self.text_encoder = ByT5TextEncoder(
            model_name=text_encoder_name,
            num_layers=text_encoder_layers,
            max_length=max_text_length,
        )

        # --- Trainable U-Net ---
        self.unet = UNet(
            in_channels=6,       # concat(x_t, c_I)
            out_channels=3,      # predicted noise
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_down_layers=num_down_layers,
            num_up_layers=num_up_layers,
            cross_attn_levels=cross_attn_levels,
            time_embed_dim=time_embed_dim,
            text_dim=text_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        x_t: torch.Tensor,          # (B, 3, H, W) noisy residual
        t: torch.Tensor,             # (B,) timesteps
        c_I: torch.Tensor,           # (B, 3, H, W) LR condition
        texts: Optional[List[str]] = None,  # List of B text strings
        text_features: Optional[torch.Tensor] = None,  # (B, M, 1536) pre-encoded
    ) -> torch.Tensor:
        """Predict noise from the noisy residual image.

        Args:
            x_t: Noisy residual image at timestep t.
            t: Diffusion timesteps.
            c_I: Low-resolution condition image.
            texts: Raw text strings (will be encoded by ByT5). Mutually
                   exclusive with text_features.
            text_features: Pre-computed ByT5 features. Use None/zero for
                          unconditional (CFG) generation.

        Returns:
            (B, 3, H, W) predicted noise ε_θ.
        """
        # Encode text if raw strings provided
        if text_features is None:
            if texts is not None:
                text_features = self.text_encoder(texts=texts)
            else:
                # Unconditional: empty text (∅)
                text_features = None

        return self.unet(x_t, t, c_I, text_features)

    def predict_noise_cfg(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c_I: torch.Tensor,
        text_features: torch.Tensor,
        omega: float = 1.0,
    ) -> torch.Tensor:
        """Classifier-Free Guidance prediction (Eq. 2 in paper).

        ε_guided = ε_θ(x_t, t, c_I, ∅) + ω * (ε_θ(x_t, t, c_I, c_T) - ε_θ(x_t, t, c_I, ∅))

        Args:
            x_t: Noisy residual image.
            t: Timesteps.
            c_I: LR condition image.
            text_features: (B, M, 1536) ByT5 text features.
            omega: Guidance scale. 3.0 for GT text, <1.0 for noisy OCR.

        Returns:
            (B, 3, H, W) guided noise prediction.
        """
        # Unconditional: image-only prediction
        noise_uncond = self.unet(x_t, t, c_I, c_T=None)

        # Conditional: image + text prediction
        noise_cond = self.unet(x_t, t, c_I, c_T=text_features)

        # CFG combination
        return noise_uncond + omega * (noise_cond - noise_uncond)

    @classmethod
    def from_config(cls, config: Dict) -> "TextSRModel":
        """Create model from a config dictionary."""
        model_cfg = config.get("model", config)
        return cls(
            base_channels=model_cfg.get("base_channels", 32),
            channel_multipliers=model_cfg.get("channel_multipliers", [1, 2, 4, 8, 8]),
            num_down_layers=model_cfg.get("num_down_layers", 2),
            num_up_layers=model_cfg.get("num_up_layers", 3),
            cross_attn_levels=model_cfg.get("cross_attn_levels", [3, 4]),
            time_embed_dim=model_cfg.get("time_embed_dim", 128),
            num_heads=model_cfg.get("num_heads", 8),
            dropout=model_cfg.get("dropout", 0.0),
            text_encoder_name=model_cfg.get("text_encoder_name", "google/byt5-base"),
            text_encoder_layers=model_cfg.get("text_encoder_layers", 2),
            text_embed_dim=model_cfg.get("text_embed_dim", 1536),
            max_text_length=model_cfg.get("max_text_length", 256),
        )
