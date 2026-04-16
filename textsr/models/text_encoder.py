"""ByT5 Text Encoder for TextSR.

Per Section 3.2 of arXiv:2505.23119v1:
- Uses google/byt5-base (UTF-8 byte-level tokenizer, 259 vocab)
- Only the first 2 encoder layers are used
- Completely frozen — no gradient computation
- Output: (B, M, 1536) where M = sequence length, 1536 = embedding dim
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel, T5Config


class ByT5TextEncoder(nn.Module):
    """Frozen ByT5-Base text encoder for multilingual character priors.

    The encoder converts UTF-8 byte sequences into dense feature vectors
    that capture character-to-shape priors across multiple languages.
    Only the first `num_layers` transformer layers are used, as deeper
    layers encode translation semantics rather than glyph features.
    """

    def __init__(
        self,
        model_name: str = "google/byt5-base",
        num_layers: int = 2,
        max_length: int = 256,
    ):
        """
        Args:
            model_name: HuggingFace model identifier.
            num_layers: Number of encoder layers to use (paper uses 2).
            max_length: Maximum UTF-8 token sequence length.
        """
        super().__init__()
        self.max_length = max_length
        self.num_layers = num_layers

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load full encoder then truncate to `num_layers`
        full_config = T5Config.from_pretrained(model_name)
        self.embed_dim = full_config.d_model  # 1536 for byt5-base

        # Build a config for truncated encoder
        truncated_config = T5Config.from_pretrained(model_name)
        truncated_config.num_layers = num_layers
        truncated_config.num_decoder_layers = 0

        # Load pretrained weights — we load the full model then keep first N layers
        full_encoder = T5EncoderModel.from_pretrained(model_name)
        self.encoder = T5EncoderModel(truncated_config)

        # Copy shared embedding
        self.encoder.shared.load_state_dict(full_encoder.shared.state_dict())

        # Copy first `num_layers` encoder blocks
        for i in range(num_layers):
            self.encoder.encoder.block[i].load_state_dict(
                full_encoder.encoder.block[i].state_dict()
            )
        # Copy final layer norm
        self.encoder.encoder.final_layer_norm.load_state_dict(
            full_encoder.encoder.final_layer_norm.state_dict()
        )

        # Clean up full model
        del full_encoder

        # Freeze all parameters — critical per paper Section 3.2
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(
        self,
        texts: Optional[Union[List[str], torch.Tensor]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text strings into ByT5 features.

        Args:
            texts: List of B strings (UTF-8 text content).
            input_ids: Pre-tokenized input IDs (B, M). Alternative to texts.
            attention_mask: (B, M) attention mask for input_ids.

        Returns:
            (B, M, 1536) text feature embeddings.
        """
        if input_ids is None:
            assert texts is not None, "Either texts or input_ids must be provided"
            tokenized = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

        # Move to same device as encoder
        device = next(self.encoder.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Forward through truncated encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return outputs.last_hidden_state  # (B, M, 1536)

    def encode_empty(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create empty text conditioning (for CFG unconditional branch).

        Returns a zero tensor matching the expected text feature shape.
        This represents ∅ (empty text) in the paper's notation.
        """
        return torch.zeros(
            batch_size, self.max_length, self.embed_dim,
            device=device, dtype=torch.float32
        )
