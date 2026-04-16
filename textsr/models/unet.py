"""U-Net Denoiser for TextSR.

Architecture per arXiv:2505.23119v1 Appendix A:
- Base channels: 32, multipliers [1, 2, 4, 8, 8] → [32, 64, 128, 256, 256]
- 5 encoder levels (2 ResBlocks each) + 5 decoder levels (3 ResBlocks each)
- Input: concat(x_t, c_I) = 6 channels (3 noisy + 3 LR condition)
- Cross-attention injected at levels 3 and 4 (feature maps 3×30×256 and 1×10×256)
- Sinusoidal timestep embedding
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Timestep Embedding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal timestep embedding as in DDPM (Ho et al., 2020)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer timesteps.
        Returns:
            (B, dim) embeddings.
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)
        return emb


class TimestepEmbedding(nn.Module):
    """MLP to project sinusoidal embedding to model dimension."""

    def __init__(self, sinusoidal_dim: int, out_dim: int):
        super().__init__()
        self.sinusoidal_embed = SinusoidalPositionalEmbedding(sinusoidal_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal_embed(t))


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with timestep conditioning.

    Follows the standard DDPM ResBlock:
    GroupNorm → SiLU → Conv → (+ time_emb) → GroupNorm → SiLU → Dropout → Conv → (+ skip)
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # Add timestep embedding
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip_conv(x)


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for injecting text features into U-Net.

    Q = W_Q · φ(x_t)     (from U-Net features)
    K = W_K · τ(c_T)     (from ByT5 text features)
    V = W_V · τ(c_T)     (from ByT5 text features)

    As described in Section 3.2 of the paper, following Rombach et al. (LDM).
    """

    def __init__(self, channels: int, text_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, \
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(text_dim, channels)
        self.to_v = nn.Linear(text_dim, channels)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) U-Net feature map.
            context: (B, M, text_dim) ByT5 text features.  None → skip attention.
        Returns:
            (B, C, H, W) feature map with text information injected.
        """
        if context is None:
            return x

        B, C, H, W = x.shape
        residual = x

        # Normalize and flatten spatial dims
        x_flat = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

        q = self.to_q(x_flat)   # (B, HW, C)
        k = self.to_k(context)  # (B, M, C)
        v = self.to_v(context)  # (B, M, C)

        # Multi-head attention
        q = q.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, heads, HW, M)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, HW, head_dim)

        out = out.permute(0, 2, 1, 3).reshape(B, H * W, C)  # (B, HW, C)
        out = self.proj_out(out)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        return residual + out


class Downsample(nn.Module):
    """2x spatial downsampling with strided conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """2x spatial upsampling with nearest + conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder / Decoder Level Blocks
# ---------------------------------------------------------------------------

class EncoderLevel(nn.Module):
    """One level of the U-Net encoder.

    Contains `num_blocks` ResBlocks, optional CrossAttention, and a Downsample.
    """

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int,
                 num_blocks: int = 2, use_cross_attn: bool = False,
                 text_dim: int = 1536, num_heads: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()

        for i in range(num_blocks):
            ch_in = in_ch if i == 0 else out_ch
            self.blocks.append(ResBlock(ch_in, out_ch, time_emb_dim, dropout))
            if use_cross_attn:
                self.attns.append(CrossAttentionBlock(out_ch, text_dim, num_heads))
            else:
                self.attns.append(None)

        self.downsample = Downsample(out_ch)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                context: Optional[torch.Tensor] = None):
        skips = []
        for block, attn in zip(self.blocks, self.attns):
            x = block(x, t_emb)
            if attn is not None:
                x = attn(x, context)
            skips.append(x)
        x = self.downsample(x)
        return x, skips


class DecoderLevel(nn.Module):
    """One level of the U-Net decoder.

    Contains an Upsample, then `num_blocks` ResBlocks (with skip connections)
    and optional CrossAttention.
    """

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int,
                 time_emb_dim: int, num_blocks: int = 3,
                 use_cross_attn: bool = False, text_dim: int = 1536,
                 num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.upsample = Upsample(in_ch)
        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()

        for i in range(num_blocks):
            if i == 0:
                ch_in = in_ch + skip_ch  # concat with last skip
            elif i < len(range(num_blocks)) and i < num_blocks:
                # Subsequent blocks may also concat with remaining skips
                ch_in = out_ch + skip_ch if i < 2 else out_ch
            else:
                ch_in = out_ch
            self.blocks.append(ResBlock(ch_in, out_ch, time_emb_dim, dropout))
            if use_cross_attn:
                self.attns.append(CrossAttentionBlock(out_ch, text_dim, num_heads))
            else:
                self.attns.append(None)

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor],
                t_emb: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        for i, (block, attn) in enumerate(zip(self.blocks, self.attns)):
            if i < len(skips):
                skip = skips[-(i + 1)]
                # Handle potential size mismatch from non-power-of-2 inputs
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
                x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)
            if attn is not None:
                x = attn(x, context)
        return x


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """TextSR U-Net Denoiser.

    Paper specification (Appendix A):
    - Input: concat(x_t, c_I) → 6 channels
    - Base ch=32, mults=[1,2,4,8,8] → ch_list=[32,64,128,256,256]
    - 5 encoder levels (2 ResBlocks each), 5 decoder levels (3 ResBlocks each)
    - Cross-attention at levels 3 & 4 only (feature map sizes 3×30 and 1×10
      for a 48×480 input), injecting ByT5 text features (dim=1536)
    - Output: 3 channels (predicted noise ε in residual space)
    """

    def __init__(
        self,
        in_channels: int = 6,          # concat(x_t, c_I): 3+3
        out_channels: int = 3,         # predicted noise
        base_channels: int = 32,
        channel_multipliers: List[int] = None,
        num_down_layers: int = 2,
        num_up_layers: int = 3,
        cross_attn_levels: List[int] = None,   # 0-indexed
        time_embed_dim: int = 128,
        text_dim: int = 1536,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 8, 8]
        if cross_attn_levels is None:
            cross_attn_levels = [3, 4]

        self.num_levels = len(channel_multipliers)
        ch_list = [base_channels * m for m in channel_multipliers]

        # --- Timestep embedding ---
        self.time_embed = TimestepEmbedding(time_embed_dim, time_embed_dim)

        # --- Input convolution ---
        self.input_conv = nn.Conv2d(in_channels, ch_list[0], 3, padding=1)

        # --- Encoder ---
        self.encoders = nn.ModuleList()
        for level in range(self.num_levels):
            in_ch = ch_list[level - 1] if level > 0 else ch_list[0]
            out_ch = ch_list[level]
            use_attn = level in cross_attn_levels
            self.encoders.append(EncoderLevel(
                in_ch, out_ch, time_embed_dim,
                num_blocks=num_down_layers,
                use_cross_attn=use_attn,
                text_dim=text_dim,
                num_heads=num_heads,
                dropout=dropout,
            ))

        # --- Middle block ---
        mid_ch = ch_list[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_embed_dim, dropout)
        self.mid_attn = CrossAttentionBlock(mid_ch, text_dim, num_heads)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_embed_dim, dropout)

        # --- Decoder ---
        self.decoders = nn.ModuleList()
        for level in reversed(range(self.num_levels)):
            out_ch = ch_list[level - 1] if level > 0 else ch_list[0]
            in_ch = ch_list[level]
            skip_ch = ch_list[level]
            use_attn = level in cross_attn_levels
            self.decoders.append(DecoderLevel(
                in_ch, out_ch, skip_ch, time_embed_dim,
                num_blocks=num_up_layers,
                use_cross_attn=use_attn,
                text_dim=text_dim,
                num_heads=num_heads,
                dropout=dropout,
            ))

        # --- Output ---
        self.out_norm = nn.GroupNorm(min(32, ch_list[0]), ch_list[0])
        self.out_conv = nn.Conv2d(ch_list[0], out_channels, 3, padding=1)

    def forward(
        self,
        x_t: torch.Tensor,         # (B, 3, H, W) noisy image (residual domain)
        t: torch.Tensor,            # (B,) timesteps
        c_I: torch.Tensor,          # (B, 3, H, W) LR condition image
        c_T: Optional[torch.Tensor] = None,  # (B, M, 1536) text features or None
    ) -> torch.Tensor:
        """
        Forward pass of the U-Net denoiser.

        Returns:
            (B, 3, H, W) predicted noise ε.
        """
        # Concatenate noisy input with LR condition along channel dim
        x = torch.cat([x_t, c_I], dim=1)  # (B, 6, H, W)
        x = self.input_conv(x)

        # Timestep embedding
        t_emb = self.time_embed(t)

        # Encoder path — collect skip connections
        all_skips = []
        for encoder in self.encoders:
            x, skips = encoder(x, t_emb, c_T)
            all_skips.append(skips)

        # Middle
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x, c_T)
        x = self.mid_block2(x, t_emb)

        # Decoder path — use skip connections in reverse
        for decoder, skips in zip(self.decoders, reversed(all_skips)):
            x = decoder(x, skips, t_emb, c_T)

        # Output
        x = self.out_conv(F.silu(self.out_norm(x)))
        return x
