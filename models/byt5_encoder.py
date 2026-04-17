"""
ByT5 Text Encoder for TextSR.

Paper: "Only first two encoder layers utilized for computational efficiency"
       "ByT5-Base encoder (frozen during training) with 1,536 embedding dimensions"

Architecture:
  - google/byt5-base loaded from HuggingFace
  - Embedding layer + first N transformer blocks + layer norm
  - All parameters frozen
  - Output: (B, seq_len, 1536) float32 tensor
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel


class ByT5TextEncoder(nn.Module):
    """
    Frozen ByT5-Base text encoder (first `num_layers` layers only).

    Encodes UTF-8 byte sequences to contextual embeddings.
    ByT5 tokenizes at byte level: each UTF-8 byte is a token.
    Vocabulary: 259 tokens (3 special + 256 byte values).
    """

    def __init__(
        self,
        model_name: str = "google/byt5-base",
        num_layers: int = 2,
        max_length: int = 64,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.max_length = max_length
        self.hidden_size = 1536  # ByT5-base hidden dim

        # Load full model, then extract components
        print(f"[ByT5TextEncoder] Loading {model_name} ...")
        full_encoder = T5EncoderModel.from_pretrained(model_name)
        encoder_stack = full_encoder.encoder

        # Extract components we need
        self.embed_tokens = encoder_stack.embed_tokens
        self.block = nn.ModuleList(list(encoder_stack.block)[:num_layers])
        self.final_layer_norm = encoder_stack.final_layer_norm
        self.dropout = encoder_stack.dropout

        # T5 uses relative position biases computed by the first block
        # We keep the block structure intact, so this is handled automatically.

        # Freeze ALL parameters
        for param in self.parameters():
            param.requires_grad_(False)

        print(f"[ByT5TextEncoder] Using first {num_layers}/{len(encoder_stack.block)} encoder layers (frozen)")

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,           # (B, L) int64
        attention_mask: torch.Tensor = None,  # (B, L) int64
    ) -> torch.Tensor:
        """
        Encode token ids to contextual embeddings.

        Args:
            input_ids:      (B, L) - byte token ids
            attention_mask: (B, L) - 1 for real tokens, 0 for padding

        Returns:
            (B, L, 1536) float32 embeddings
        """
        # Embed tokens: (B, L) -> (B, L, hidden)
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)

        # Build extended attention mask for T5
        # T5 needs mask in shape (B, 1, 1, L) with -inf for padding
        if attention_mask is not None:
            ext_mask = _get_extended_attention_mask(attention_mask, hidden_states.dtype)
        else:
            ext_mask = None

        # Run through the first num_layers transformer blocks
        position_bias = None
        for i, layer_module in enumerate(self.block):
            # T5Block signature: forward(hidden_states, attention_mask, position_bias)
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=ext_mask,
                position_bias=position_bias,
            )
            # layer_outputs: (hidden_states, key_value_states, position_bias, ...)
            hidden_states = layer_outputs[0]
            # Reuse position bias from first block for subsequent blocks
            if position_bias is None and len(layer_outputs) > 2:
                position_bias = layer_outputs[2]

        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states  # (B, L, 1536)

    def get_null_embedding(self, batch_size: int, device: torch.device) -> tuple:
        """
        Return null text embedding for classifier-free guidance.
        Uses all-zero token sequence (PAD tokens).
        """
        null_ids = torch.zeros(batch_size, self.max_length, dtype=torch.long, device=device)
        null_mask = torch.zeros(batch_size, self.max_length, dtype=torch.long, device=device)
        return null_ids, null_mask


def _get_extended_attention_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convert 2D attention mask (B, L) to T5-compatible 4D mask (B, 1, 1, L).
    Padding positions get large negative values (approx -inf) so they're
    ignored in softmax attention.
    """
    # attention_mask: 1 for attend, 0 for ignore
    extended = attention_mask[:, None, None, :]  # (B, 1, 1, L)
    extended = extended.to(dtype=dtype)
    extended = (1.0 - extended) * torch.finfo(dtype).min
    return extended


# ---------------------------------------------------------------------------
# Standalone tokenizer helper (for use without the full model)
# ---------------------------------------------------------------------------

_tokenizer_cache = {}


def get_byt5_tokenizer(model_name: str = "google/byt5-base") -> AutoTokenizer:
    if model_name not in _tokenizer_cache:
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer_cache[model_name]


def tokenize_text(
    texts,   # str or List[str]
    tokenizer,
    max_length: int = 64,
    device: torch.device = None,
) -> tuple:
    """
    Tokenize text(s) using ByT5 tokenizer.

    Returns:
        input_ids:      (B, max_length) int64
        attention_mask: (B, max_length) int64
    """
    if isinstance(texts, str):
        texts = [texts]

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

    return input_ids, attention_mask
