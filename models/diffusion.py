"""
Gaussian Diffusion (DDPM + DDIM) for TextSR.

Training:
  - Add noise to residual target x_0 at random timestep t
  - U-Net predicts the added noise (epsilon parameterization)
  - Loss: MSE(predicted_noise, actual_noise)

Inference (DDIM, 5 steps):
  - Start from pure Gaussian noise x_T
  - Iteratively denoise conditioned on image + text
  - Supports classifier-free guidance (CFG) with dual conditions

Paper formula for CFG:
  ε̃ = (1-ω)·ε(x_t, c_i, ∅) + ω·ε(x_t, c_i, c_t)
  (where ω=1 reduces to standard conditioning; ω>1 amplifies text guidance)
"""

import math
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Beta schedules
# ---------------------------------------------------------------------------

def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule from Improved DDPM (Nichol & Dhariwal, 2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, min=0, max=0.999)


# ---------------------------------------------------------------------------
# GaussianDiffusion
# ---------------------------------------------------------------------------

class GaussianDiffusion(nn.Module):
    """
    DDPM training and DDIM inference for TextSR residual diffusion.

    The diffusion operates on the *residual* x_0 = HR - LR_upsampled,
    which is shifted to [0, 255] by adding 128 (handled in dataset).
    In model space, tensors are in [-1, 1].

    Usage:
      # Training
      loss = diffusion.training_loss(unet, x0, t, image_cond, text_emb, text_mask)

      # Inference (DDIM)
      x0_pred = diffusion.ddim_sample(unet, shape, image_cond, text_emb, text_mask,
                                      cfg_weight=2.0, num_steps=5)
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        prediction_type: str = "epsilon",  # "epsilon" or "x0"
    ):
        super().__init__()
        self.timesteps = timesteps
        self.prediction_type = prediction_type

        # Build beta schedule
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Pre-compute useful quantities
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (moved to device automatically)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        # For DDPM posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    # -----------------------------------------------------------------------
    # Forward diffusion (add noise)
    # -----------------------------------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,   # (B, C, H, W) clean residual in [-1, 1]
        t: torch.Tensor,     # (B,) integer timesteps
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t from q(x_t | x_0):
          x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        x_t = sqrt_alpha * x0 + sqrt_one_minus * noise
        return x_t, noise

    # -----------------------------------------------------------------------
    # Training loss
    # -----------------------------------------------------------------------

    def training_loss(
        self,
        model: nn.Module,                          # U-Net
        x0: torch.Tensor,                          # (B, C, H, W) clean residual [-1,1]
        image_cond: torch.Tensor,                  # (B, 3, H, W) LR upsampled [-1,1]
        text_emb: Optional[torch.Tensor] = None,   # (B, L, D) ByT5 embedding
        text_mask: Optional[torch.Tensor] = None,  # (B, L) mask
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute diffusion training loss:
          L = E_t,x0,ε [ ||ε - ε_θ(x_t, t, c_i, c_t)||² ]

        Args:
            model:       U-Net denoiser
            x0:          clean residual target (B, 3, H, W) in [-1, 1]
            image_cond:  LR upsampled to HR size (B, 3, H, W) in [-1, 1]
            text_emb:    ByT5 text embeddings (None = image-only, i.e., text dropped)
            text_mask:   attention mask for text_emb

        Returns:
            scalar MSE loss
        """
        B = x0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x0.device)

        x_t, noise = self.q_sample(x0, t, noise)

        # Concatenate x_t with image condition along channel dim
        model_input = torch.cat([x_t, image_cond], dim=1)  # (B, 6, H, W)

        # Predict noise
        noise_pred = model(model_input, t, text_emb, text_mask)

        if self.prediction_type == "epsilon":
            target = noise
        else:  # x0 prediction
            target = x0

        return F.mse_loss(noise_pred, target)

    # -----------------------------------------------------------------------
    # DDIM sampling (fast inference)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: Tuple,                              # (B, 3, H, W)
        image_cond: torch.Tensor,                  # (B, 3, H, W) LR upsampled
        text_emb: Optional[torch.Tensor] = None,   # (B, L, D) text embeddings
        text_mask: Optional[torch.Tensor] = None,
        null_text_emb: Optional[torch.Tensor] = None,  # (B, L, D) null text for CFG uncond pass
        null_text_mask: Optional[torch.Tensor] = None,
        cfg_weight: float = 2.0,                   # ω in paper
        num_steps: int = 5,
        eta: float = 0.0,                          # 0 = deterministic DDIM
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        DDIM sampling with classifier-free guidance.

        ε̃(x_t,t,c_i,c_t) = (1-ω)·ε(x_t,t,c_i,∅) + ω·ε(x_t,t,c_i,c_t)

        When text_emb is None or cfg_weight=1: no CFG (image-only conditioning).

        Args:
            shape:      (B, 3, H, W) output shape
            image_cond: LR upsampled image condition
            text_emb:   ByT5 text features (None = image-only)
            cfg_weight: ω parameter (1.0 = no text boost, >1.0 = stronger text)
            num_steps:  number of DDIM steps (paper: 5)
            eta:        DDIM stochasticity (0 = fully deterministic)

        Returns:
            x0_pred: (B, 3, H, W) denoised residual in [-1, 1]
        """
        device = image_cond.device
        B = shape[0]

        # Start from pure noise
        x_t = torch.randn(*shape, device=device)

        # Select DDIM timesteps (uniformly spaced in [0, T])
        ddim_timesteps = torch.linspace(0, self.timesteps - 1, num_steps + 1, dtype=torch.long)
        ddim_timesteps = list(reversed(ddim_timesteps.tolist()))  # T -> 0

        intermediates = []

        for i in range(len(ddim_timesteps) - 1):
            t_cur = int(ddim_timesteps[i])
            t_prev = int(ddim_timesteps[i + 1])

            t_tensor = torch.full((B,), t_cur, dtype=torch.long, device=device)

            # --- Predict noise with CFG ---
            model_input = torch.cat([x_t, image_cond], dim=1)

            if text_emb is not None and cfg_weight != 1.0:
                # Dual-condition CFG: run model twice
                # Unconditioned pass uses null text embeddings (matching training distribution
                # where text_drop_prob dropped text → null ByT5 embedding, NOT skipped attn)
                uncond_emb = null_text_emb if null_text_emb is not None else text_emb
                uncond_mask = null_text_mask if null_text_mask is not None else text_mask
                noise_uncond = model(model_input, t_tensor, uncond_emb, uncond_mask)
                # Conditioned pass: image + text
                noise_text = model(model_input, t_tensor, text_emb, text_mask)
                # CFG combination: ε̃ = (1-ω)·ε_uncond + ω·ε_text
                noise_pred = (1 - cfg_weight) * noise_uncond + cfg_weight * noise_text
            else:
                # Single forward pass (no CFG or image-only)
                noise_pred = model(model_input, t_tensor, text_emb, text_mask)

            # --- DDIM update step ---
            x_t = self._ddim_step(x_t, noise_pred, t_cur, t_prev, eta)

            if return_intermediates:
                intermediates.append(x_t.clone())

        if return_intermediates:
            return x_t, intermediates
        return x_t

    def _ddim_step(
        self,
        x_t: torch.Tensor,
        noise_pred: torch.Tensor,
        t: int,
        t_prev: int,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        One DDIM denoising step from timestep t to t_prev.
        """
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Predict x_0 from noise prediction
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t]
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t]
        x0_pred = sqrt_recip * x_t - sqrt_recipm1 * noise_pred
        x0_pred = x0_pred.clamp(-1, 1)  # clip to valid range

        # DDIM update
        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
        direction = torch.sqrt(1 - alpha_prev - sigma ** 2) * noise_pred
        noise = sigma * torch.randn_like(x_t)

        x_prev = torch.sqrt(alpha_prev) * x0_pred + direction + noise
        return x_prev

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Extract values from 1D tensor `a` at indices `t` and reshape to x_shape."""
        B = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(B, *([1] * (len(x_shape) - 1)))
