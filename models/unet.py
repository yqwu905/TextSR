"""
U-Net Denoiser for TextSR diffusion model.

Architecture (from paper):
  - 32 initial channels, five residual down/up block pairs
  - Channel multipliers: [1, 2, 4, 8, 8] -> channels [32, 64, 128, 256, 256]
  - Cross-attention with ByT5 text embeddings at deeper levels
  - Time embedding (sinusoidal + MLP) injected into every ResBlock
  - 6-channel input: 3 (noisy residual) + 3 (LR upsampled as image condition)
  - 3-channel output: predicted noise (epsilon parameterization)

Skip connection pattern (standard U-Net):
  Encoder level i:
    for each res block: push skip
    if not last level: downsample then push skip
  Decoder level i (reversed):
    for (num_res_blocks + 1) blocks: pop skip, cat, res block
    if not last level: upsample
"""

import math
from typing import List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

def _num_groups(num_channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return g
    return 1


def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embeddings."""
    assert len(timesteps.shape) == 1
    half = dim // 2
    freq = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ResBlock(nn.Module):
    """Residual block: GroupNorm + Swish + Conv, with time embedding injection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act2 = Swish()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class CrossAttention(nn.Module):
    """
    Cross-attention: spatial features (query) × text features (key/value).
    Spatial features are flattened, attended, then reshaped back.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        heads: int = 8,
        head_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.norm = nn.GroupNorm(_num_groups(query_dim), query_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(
        self,
        x: torch.Tensor,                              # (B, C, H, W)
        context: torch.Tensor,                        # (B, L, context_dim)
        context_mask: Optional[torch.Tensor] = None,  # (B, L)
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        h = rearrange(self.norm(x), "b c h w -> b (h w) c")

        q = rearrange(self.to_q(h), "b l (n d) -> b n l d", n=self.heads)
        k = rearrange(self.to_k(context), "b l (n d) -> b n l d", n=self.heads)
        v = rearrange(self.to_v(context), "b l (n d) -> b n l d", n=self.heads)

        attn = torch.einsum("bnid,bnjd->bnij", q, k) * self.scale
        if context_mask is not None:
            mask = context_mask[:, None, None, :].bool()
            attn = attn.masked_fill(~mask, float("-inf"))
        attn = F.softmax(attn, dim=-1).nan_to_num()

        out = torch.einsum("bnij,bnjd->bnid", attn, v)
        out = self.to_out(rearrange(out, "b n l d -> b l (n d)"))
        return x + rearrange(out, "b (h w) c -> b c h w", h=H, w=W)


class SelfAttention(nn.Module):
    """Self-attention for bottleneck."""

    def __init__(self, channels: int, heads: int = 8, head_dim: int = 32):
        super().__init__()
        inner_dim = heads * head_dim
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.to_qkv = nn.Linear(channels, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = rearrange(self.norm(x), "b c h w -> b (h w) c")
        q, k, v = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = [rearrange(t, "b l (n d) -> b n l d", n=self.heads) for t in (q, k, v)]
        attn = F.softmax(torch.einsum("bnid,bnjd->bnij", q, k) * self.scale, dim=-1)
        out = self.to_out(rearrange(torch.einsum("bnij,bnjd->bnid", attn, v), "b n l d -> b l (n d)"))
        return x + rearrange(out, "b (h w) c -> b c h w", h=H, w=W)


class Downsample(nn.Module):
    """Strided conv downsampling (2×)."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsample (2×) + conv."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    U-Net denoiser for TextSR.

    Input:  (B, in_channels=6, H, W)  [3 noisy residual + 3 LR upsampled]
    Output: (B, out_channels=3, H, W) [predicted noise]

    Skip connection counts:
      Encoder level i: num_res_blocks skips (after each ResBlock)
                        + 1 skip after Downsample (except last level)
      Decoder level i: consumes (num_res_blocks + 1) skips
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 3,
        base_channels: int = 32,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8, 8),
        num_res_blocks: int = 2,
        attention_levels: Tuple[int, ...] = (3, 4),
        text_context_dim: int = 1536,
        time_embed_dim: int = 256,
        dropout: float = 0.0,
        attn_heads: int = 8,
        attn_head_dim: int = 32,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.attention_levels: Set[int] = set(attention_levels)
        num_levels = len(channel_mult)
        ch_list = [base_channels * m for m in channel_mult]

        # --- Timestep embedding ---
        sinusoidal_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(sinusoidal_dim, time_embed_dim),
            Swish(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # --- Input projection ---
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ===== Encoder =====
        # Flat lists for sequential processing in forward()
        self.enc_res_blocks = nn.ModuleList()
        self.enc_attns = nn.ModuleList()        # CrossAttention or None
        self.enc_downsamples = nn.ModuleList()  # Downsample or None (last level)

        # Track skip channel counts for decoder building
        skip_ch_list: List[int] = []

        in_ch = base_channels
        for level, out_ch in enumerate(ch_list):
            for _ in range(num_res_blocks):
                self.enc_res_blocks.append(ResBlock(in_ch, out_ch, time_embed_dim, dropout))
                if level in self.attention_levels:
                    self.enc_attns.append(CrossAttention(
                        out_ch, text_context_dim, attn_heads, attn_head_dim, dropout
                    ))
                else:
                    self.enc_attns.append(None)
                skip_ch_list.append(out_ch)
                in_ch = out_ch

            if level < num_levels - 1:
                self.enc_downsamples.append(Downsample(out_ch))
                skip_ch_list.append(out_ch)
            else:
                self.enc_downsamples.append(None)

        # ===== Bottleneck =====
        self.mid_res1 = ResBlock(in_ch, in_ch, time_embed_dim, dropout)
        self.mid_self_attn = SelfAttention(in_ch, attn_heads, attn_head_dim)
        self.mid_cross_attn = CrossAttention(in_ch, text_context_dim, attn_heads, attn_head_dim, dropout)
        self.mid_res2 = ResBlock(in_ch, in_ch, time_embed_dim, dropout)

        # ===== Decoder =====
        # Each decoder level uses (num_res_blocks + 1) blocks
        self.dec_res_blocks = nn.ModuleList()
        self.dec_attns = nn.ModuleList()
        self.dec_upsamples = nn.ModuleList()

        # Consume skip_ch_list in reverse
        skip_iter = list(reversed(skip_ch_list))
        skip_ptr = 0

        for level in reversed(range(num_levels)):
            out_ch = ch_list[level]
            # Level 0 (shallowest encoder level) has no DS skip from level -1.
            # All other decoder levels also consume one DS skip (from the encoder level below them).
            # Decoder consumption order (from deepest to shallowest):
            #   Lev 4: r1_l4, r0_l4, DS3    (3 skips at deepest spatial scale)
            #   Lev 3: r1_l3, r0_l3, DS2    (3 skips at next scale)
            #   ...
            #   Lev 0: r1_l0, r0_l0         (2 skips, no DS from level -1)
            n_dec_blocks = num_res_blocks if level == 0 else num_res_blocks + 1
            for _ in range(n_dec_blocks):
                skip_ch = skip_iter[skip_ptr]
                skip_ptr += 1
                self.dec_res_blocks.append(ResBlock(in_ch + skip_ch, out_ch, time_embed_dim, dropout))
                if level in self.attention_levels:
                    self.dec_attns.append(CrossAttention(
                        out_ch, text_context_dim, attn_heads, attn_head_dim, dropout
                    ))
                else:
                    self.dec_attns.append(None)
                in_ch = out_ch

            if level > 0:
                self.dec_upsamples.append(Upsample(out_ch))
            else:
                self.dec_upsamples.append(None)

        # ===== Output =====
        self.out_norm = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.out_act = Swish()
        self.out_conv = nn.Conv2d(in_ch, out_channels, 3, padding=1)

        # Store sizes for forward pass control
        self._num_levels = num_levels
        self._num_res_blocks = num_res_blocks
        self._ch_list = ch_list

    def forward(
        self,
        x: torch.Tensor,                            # (B, 6, H, W)
        t: torch.Tensor,                            # (B,) timestep indices
        text_emb: Optional[torch.Tensor] = None,    # (B, L, 1536)
        text_mask: Optional[torch.Tensor] = None,   # (B, L)
    ) -> torch.Tensor:
        # --- Timestep embedding ---
        t_emb = self.time_embed(get_timestep_embedding(t, self.base_channels * 4))

        # --- Input ---
        h = self.input_conv(x)

        # --- Encoder ---
        skips: List[torch.Tensor] = []
        enc_idx = 0  # index into enc_res_blocks / enc_attns
        ds_idx = 0   # index into enc_downsamples

        for level, out_ch in enumerate(self._ch_list):
            for _ in range(self._num_res_blocks):
                h = self.enc_res_blocks[enc_idx](h, t_emb)
                attn = self.enc_attns[enc_idx]
                if attn is not None and text_emb is not None:
                    h = attn(h, text_emb, text_mask)
                skips.append(h)
                enc_idx += 1

            ds = self.enc_downsamples[ds_idx]
            if ds is not None:
                h = ds(h)
                skips.append(h)
            ds_idx += 1

        # --- Bottleneck ---
        h = self.mid_res1(h, t_emb)
        h = self.mid_self_attn(h)
        if text_emb is not None:
            h = self.mid_cross_attn(h, text_emb, text_mask)
        h = self.mid_res2(h, t_emb)

        # --- Decoder ---
        dec_idx = 0  # index into dec_res_blocks / dec_attns
        up_idx = 0   # index into dec_upsamples

        for level in reversed(range(self._num_levels)):
            n_dec_blocks = self._num_res_blocks if level == 0 else self._num_res_blocks + 1
            for _ in range(n_dec_blocks):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.dec_res_blocks[dec_idx](h, t_emb)
                attn = self.dec_attns[dec_idx]
                if attn is not None and text_emb is not None:
                    h = attn(h, text_emb, text_mask)
                dec_idx += 1

            us = self.dec_upsamples[up_idx]
            if us is not None:
                h = us(h)
            up_idx += 1

        # --- Output ---
        return self.out_conv(self.out_act(self.out_norm(h)))
