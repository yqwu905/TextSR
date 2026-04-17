"""
Conditional Flow Matching (CFM) for TextSR.

Replaces DDPM/DDIM with a simpler linear OT path:

  Forward (interpolation):
    x_t = (1 - t) * eps + t * x_1,   eps ~ N(0,I),  t ~ U[0,1]

  Velocity target (constant along the trajectory):
    v* = x_1 - eps

  Training loss:
    L = E [ || v_θ(x_t, t, c_i, c_t) - v* ||² ]

  Inference (Euler ODE from t=0 to t=1):
    x_{t+dt} = x_t + v_θ(x_t, t) * dt,   x_0 ~ N(0,I)

  CFG (same formula as DDPM):
    v̂ = (1-ω) * v_θ(x_t, t, c_i, ∅) + ω * v_θ(x_t, t, c_i, c_t)

Advantages over DDPM:
  - No noise schedule hyper-parameters (β, α, etc.)
  - Straighter trajectories → fewer inference steps needed
  - Simpler loss (no SNR weighting required)

References:
  Flow Matching for Generative Modeling (Lipman et al., 2022)
  Improving and Generalizing Flow-Matching (Albergo & Vanden-Eijnden, 2022)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatching(nn.Module):
    """
    Conditional Flow Matching training and Euler-ODE inference.

    The model (U-Net) predicts the velocity field v_θ(x_t, t, c_i, c_t).
    t ∈ [0, 1] is scaled to [0, 999] for the UNet's sinusoidal time embedding.
    """

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_loss(
        self,
        model: nn.Module,
        x1: torch.Tensor,                          # (B, 3, H, W) clean residual in [-1, 1]
        image_cond: torch.Tensor,                  # (B, 3, H, W) LR bicubic-upsampled in [-1, 1]
        text_emb: Optional[torch.Tensor] = None,   # (B, L, D) ByT5 embeddings
        text_mask: Optional[torch.Tensor] = None,  # (B, L)
    ) -> torch.Tensor:
        """
        Flow matching loss:
          1. Sample t ~ U[0, 1]
          2. Sample eps ~ N(0, I)
          3. Interpolate: x_t = (1-t)*eps + t*x_1
          4. Velocity target: v* = x_1 - eps
          5. Loss = MSE(v_θ(x_t, t, c_i, c_t), v*)
        """
        B, C, H, W = x1.shape
        device = x1.device

        t = torch.rand(B, device=device)                # (B,) in [0, 1]
        eps = torch.randn_like(x1)                      # (B, 3, H, W)

        t4 = t.view(B, 1, 1, 1)
        x_t = (1.0 - t4) * eps + t4 * x1               # linear interpolation
        v_target = x1 - eps                             # constant target velocity

        # Scale t to integer index for sinusoidal embedding (same UNet as DDPM)
        t_idx = (t * 999).long()

        model_input = torch.cat([x_t, image_cond], dim=1)  # (B, 6, H, W)
        v_pred = model(model_input, t_idx, text_emb, text_mask)

        return F.mse_loss(v_pred, v_target)

    # ------------------------------------------------------------------
    # Inference (Euler ODE solver)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple,                                      # (B, 3, H, W)
        image_cond: torch.Tensor,                          # (B, 3, H, W)
        text_emb: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        null_text_emb: Optional[torch.Tensor] = None,     # for CFG uncond pass
        null_text_mask: Optional[torch.Tensor] = None,
        cfg_weight: float = 2.0,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Euler integration of dx/dt = v_θ(x_t, t) from t=0 to t=1.

        CFG:  v̂ = (1-ω)*v_uncond + ω*v_text
        Returns x_1 (the denoised residual in approximately [-1, 1]).
        """
        B = shape[0]
        device = image_cond.device

        x = torch.randn(*shape, device=device)   # x_0 ~ N(0, I)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i * dt                               # current t in [0, 1)
            t_idx = int(t_val * 999)
            t_tensor = torch.full((B,), t_idx, dtype=torch.long, device=device)

            model_input = torch.cat([x, image_cond], dim=1)

            if text_emb is not None and cfg_weight != 1.0:
                uncond_emb = null_text_emb if null_text_emb is not None else text_emb
                uncond_mask = null_text_mask if null_text_mask is not None else text_mask
                v_uncond = model(model_input, t_tensor, uncond_emb, uncond_mask)
                v_text = model(model_input, t_tensor, text_emb, text_mask)
                v = (1.0 - cfg_weight) * v_uncond + cfg_weight * v_text
            else:
                v = model(model_input, t_tensor, text_emb, text_mask)

            x = x + v * dt

        return x  # x_1 ≈ residual in [-1, 1]
