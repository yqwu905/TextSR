"""DDPM Forward Process and Training Loss for TextSR.

Implements the standard Denoising Diffusion Probabilistic Model:
- Linear beta schedule from β_start to β_end over T timesteps
- Forward process: q(x_t | x_0) = N(x_t; √(ᾱ_t) x_0, (1-ᾱ_t) I)
- Training target: predict noise ε from x_t

Key TextSR-specific detail (Section 3.2):
  The diffusion target x_0 is the **residual** image (HR - LR),
  NOT the HR image itself. This prevents hallucination artifacts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Dict, Optional, Tuple


class GaussianDiffusion(nn.Module):
    """Gaussian Diffusion process for training and loss computation.

    This module handles:
    1. Computing the noise schedule (alphas, betas)
    2. Forward diffusion: adding noise to clean residuals
    3. Training loss: MSE between predicted and true noise
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        # Compute noise schedule
        if beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
        elif beta_schedule == "cosine":
            # Cosine schedule from Nichol & Dhariwal (2021)
            steps = num_timesteps + 1
            s = 0.008
            t_vals = np.linspace(0, num_timesteps, steps, dtype=np.float64)
            alphas_bar = np.cos((t_vals / num_timesteps + s) / (1 + s) * np.pi / 2) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            betas = np.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        # Register as buffers (not parameters — no gradients)
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.float32))
        self.register_buffer("alphas_cumprod",
                             torch.tensor(alphas_cumprod, dtype=torch.float32))
        self.register_buffer("alphas_cumprod_prev",
                             torch.tensor(alphas_cumprod_prev, dtype=torch.float32))

        # Pre-compute useful quantities
        self.register_buffer("sqrt_alphas_cumprod",
                             torch.tensor(np.sqrt(alphas_cumprod), dtype=torch.float32))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.tensor(np.sqrt(1.0 - alphas_cumprod), dtype=torch.float32))

        # For posterior q(x_{t-1}|x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance",
                             torch.tensor(posterior_variance, dtype=torch.float32))
        self.register_buffer("posterior_log_variance_clipped",
                             torch.tensor(
                                 np.log(np.maximum(posterior_variance, 1e-20)),
                                 dtype=torch.float32
                             ))
        self.register_buffer("posterior_mean_coef1",
                             torch.tensor(
                                 np.sqrt(alphas_cumprod_prev) * betas / (1.0 - alphas_cumprod),
                                 dtype=torch.float32
                             ))
        self.register_buffer("posterior_mean_coef2",
                             torch.tensor(
                                 np.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
                                 dtype=torch.float32
                             ))

    def _extract(self, arr: torch.Tensor, t: torch.Tensor,
                 x_shape: torch.Size) -> torch.Tensor:
        """Extract values from a 1D tensor for a batch of indices.

        Reshapes to broadcast against x of shape (B, C, H, W).
        """
        batch_size = t.shape[0]
        out = arr.gather(0, t)
        return out.reshape(batch_size, *([1] * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: sample x_t from q(x_t | x_0).

        x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε

        Args:
            x_0: (B, 3, H, W) clean residual images (HR - LR).
            t: (B,) timesteps.
            noise: Optional pre-generated noise. If None, sampled from N(0,I).

        Returns:
            x_t: (B, 3, H, W) noisy residual at timestep t.
            noise: (B, 3, H, W) the noise that was added.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def training_loss(
        self,
        model: nn.Module,
        hr_images: torch.Tensor,      # (B, 3, H, W) high-res images
        lr_images: torch.Tensor,      # (B, 3, H, W) low-res images
        text_features: Optional[torch.Tensor] = None,  # (B, M, D) or None
        text_dropout_prob: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Compute DDPM training loss.

        Steps:
        1. Compute residual: x_0 = HR - LR (the diffusion target)
        2. Sample random timestep t
        3. Add noise to get x_t
        4. Randomly drop text condition (for CFG training)
        5. Predict noise with model
        6. Return MSE loss

        Args:
            model: TextSR model (or just U-Net).
            hr_images: Ground-truth high-resolution text crops.
            lr_images: Degraded low-resolution text crops.
            text_features: Pre-encoded ByT5 features. None = unconditional only.
            text_dropout_prob: Probability of dropping text for CFG.

        Returns:
            Dictionary with 'loss' and optional diagnostics.
        """
        batch_size = hr_images.shape[0]
        device = hr_images.device

        # Step 1: Compute residual target (crucial for preventing hallucination)
        x_0 = hr_images - lr_images  # Residual domain

        # Step 2: Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # Step 3: Forward diffusion — add noise to residual
        x_t, noise = self.q_sample(x_0, t)

        # Step 4: Random text dropout for CFG training
        if text_features is not None and text_dropout_prob > 0:
            # Create dropout mask (per-sample)
            drop_mask = torch.rand(batch_size, device=device) < text_dropout_prob
            if drop_mask.any():
                # Zero out text features for dropped samples
                text_features = text_features.clone()
                text_features[drop_mask] = 0.0

        # Step 5: Predict noise
        # Use c_T= for U-Net compatibility (train.py passes model.unet)
        noise_pred = model(x_t, t, lr_images, c_T=text_features)

        # Step 6: MSE loss
        loss = F.mse_loss(noise_pred, noise)

        return {
            "loss": loss,
            "noise_pred": noise_pred.detach(),
            "noise_true": noise.detach(),
        }

    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Recover x_0 from x_t and predicted noise.

        x_0 = (x_t - √(1-ᾱ_t) * ε) / √(ᾱ_t)
        """
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        return (x_t - sqrt_one_minus_alpha_bar * noise) / sqrt_alpha_bar
