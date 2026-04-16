"""DDIM Sampler for TextSR Inference.

Implements the 5-step DDIM sampling from Song et al. (2020),
with Classifier-Free Guidance (CFG) for dual-condition control.

Per Section 3.3.1 and Appendix A:
- 5 denoising steps
- CFG: ε_guided = ε_θ(x_t,t,c_I,∅) + ω·(ε_θ(x_t,t,c_I,c_T) - ε_θ(x_t,t,c_I,∅))
- ω = 3.0 for ground-truth text, ω < 1.0 for noisy OCR text
"""

from typing import Callable, Optional

import torch
import numpy as np

from .ddpm import GaussianDiffusion


class DDIMSampler:
    """DDIM deterministic sampler for fast inference.

    Performs denoising in `num_steps` steps (default 5) by skipping
    timesteps in the original T=1000 schedule.
    """

    def __init__(
        self,
        diffusion: GaussianDiffusion,
        num_steps: int = 5,
        eta: float = 0.0,          # η=0 → fully deterministic DDIM
    ):
        """
        Args:
            diffusion: GaussianDiffusion instance (provides noise schedule).
            num_steps: Number of DDIM sampling steps.
            eta: Stochasticity parameter (0=deterministic, 1=DDPM).
        """
        self.diffusion = diffusion
        self.num_steps = num_steps
        self.eta = eta

        # Compute sub-sequence of timesteps
        T = diffusion.num_timesteps
        # Uniformly spaced timesteps: e.g., for T=1000, steps=5 →
        # [199, 399, 599, 799, 999] (reversed during sampling)
        self.timesteps = np.asarray(
            list(range(0, T, T // num_steps))
        )[:num_steps]
        # Reverse for denoising (start from T-1, go to 0)
        self.timesteps = np.flip(self.timesteps).copy()

    @torch.no_grad()
    def sample(
        self,
        model: Callable,
        c_I: torch.Tensor,                          # (B, 3, H, W)
        text_features: Optional[torch.Tensor],       # (B, M, D) or None
        omega: float = 1.0,                          # CFG guidance scale
        shape: Optional[tuple] = None,               # override (B, 3, H, W)
    ) -> torch.Tensor:
        """Run DDIM sampling to generate super-resolved residual.

        Args:
            model: Callable that takes (x_t, t, c_I, text_features) and
                   returns predicted noise. Should be TextSRModel.
            c_I: Low-resolution condition image.
            text_features: ByT5 text features. None for image-only.
            omega: CFG guidance scale.
            shape: Output shape. Defaults to c_I.shape.

        Returns:
            x_0: (B, 3, H, W) denoised residual image.
        """
        device = c_I.device
        if shape is None:
            shape = c_I.shape

        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        for i, t_val in enumerate(self.timesteps):
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)

            # ---- CFG noise prediction ----
            if omega != 1.0 and text_features is not None:
                # Two forward passes for CFG
                noise_uncond = model(x_t, t, c_I, text_features=None)
                noise_cond = model(x_t, t, c_I, text_features=text_features)
                noise_pred = noise_uncond + omega * (noise_cond - noise_uncond)
            elif text_features is not None:
                # ω=1.0: just conditional prediction (no CFG amplification)
                noise_pred = model(x_t, t, c_I, text_features=text_features)
            else:
                # Image-only (unconditional text)
                noise_pred = model(x_t, t, c_I, text_features=None)

            # ---- DDIM step ----
            x_t = self._ddim_step(x_t, noise_pred, t_val, i)

        return x_t  # This is the predicted x_0 (residual)

    def _ddim_step(
        self,
        x_t: torch.Tensor,
        noise_pred: torch.Tensor,
        t: int,
        step_idx: int,
    ) -> torch.Tensor:
        """Perform one DDIM denoising step.

        x_{t-1} = √(ᾱ_{t-1}) * x_0_pred + √(1 - ᾱ_{t-1} - σ²) * ε_pred + σ * z

        where x_0_pred = (x_t - √(1-ᾱ_t) * ε_pred) / √(ᾱ_t)
        and σ = η * √((1-ᾱ_{t-1})/(1-ᾱ_t)) * √(1 - ᾱ_t/ᾱ_{t-1})
        """
        alpha_bar_t = self.diffusion.alphas_cumprod[t]

        # Previous timestep
        if step_idx < len(self.timesteps) - 1:
            t_prev = self.timesteps[step_idx + 1]
            alpha_bar_t_prev = self.diffusion.alphas_cumprod[t_prev]
        else:
            # Last step: go to t=0 (clean)
            alpha_bar_t_prev = torch.tensor(1.0, device=x_t.device)

        # Predict x_0 from current x_t and noise prediction
        x_0_pred = (
            (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred)
            / torch.sqrt(alpha_bar_t)
        )

        # Optionally clip x_0 prediction for stability
        # x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

        # Compute σ (stochasticity)
        sigma = 0.0
        if self.eta > 0 and step_idx < len(self.timesteps) - 1:
            sigma = (
                self.eta
                * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t))
                * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)
            )

        # Direction pointing to x_t
        dir_xt = torch.sqrt(
            torch.clamp(1 - alpha_bar_t_prev - sigma ** 2, min=0)
        ) * noise_pred

        # Compute x_{t-1}
        x_t_prev = torch.sqrt(alpha_bar_t_prev) * x_0_pred + dir_xt

        if sigma > 0:
            noise = torch.randn_like(x_t)
            x_t_prev = x_t_prev + sigma * noise

        return x_t_prev
