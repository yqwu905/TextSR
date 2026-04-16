# TextSR: Diffusion Super-Resolution with Multilingual OCR Guidance

PyTorch reproduction of [TextSR (arXiv:2505.23119v1)](https://arxiv.org/abs/2505.23119).

## Overview

TextSR is a multimodal diffusion model for **Scene Text Image Super-Resolution (STISR)**. It enhances text legibility in images by:

1. **Detecting** text regions via OCR detection
2. **Recognizing** text content using UTF-8 multilingual encoding
3. **Encoding** characters with frozen ByT5-Base (first 2 layers)
4. **Super-resolving** text crops using a conditional U-Net diffusion model
5. **Blending** results back into the full image with LPF harmonization

## Architecture

```
Input Image → Text Detector → Affine Crop (48×480)
                                    ↓
                              ┌─────────────┐
     LR Image (c_I) ────────>│             │
                              │   U-Net     │──→ Predicted Noise ε
     Noisy Residual (x_t) ──>│  Denoiser   │
                              │             │
     ByT5 Text Features ────>│ (CrossAttn) │
                              └─────────────┘
                                    ↓
                          DDIM 5-step Sampling
                                    ↓
                         Residual + LR = SR Crop
                                    ↓
                    Inverse Affine + LPF Blending → Full SR Image
```

### U-Net Specification
| Parameter | Value |
|-----------|-------|
| Base channels | 32 |
| Channel multipliers | [1, 2, 4, 8, 8] |
| Encoder levels | 5 (2 ResBlocks each) |
| Decoder levels | 5 (3 ResBlocks each) |
| Cross-attention levels | 3 and 4 |
| Text encoder | ByT5-Base (first 2 layers, frozen) |
| Text embedding dim | 1536 |

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
TextSR/
├── configs/default.yaml          # Training/inference config
├── textsr/
│   ├── models/
│   │   ├── unet.py              # U-Net denoiser with cross-attention
│   │   ├── text_encoder.py      # ByT5-Base frozen encoder
│   │   └── textsr_model.py      # Top-level model wrapper
│   ├── diffusion/
│   │   ├── ddpm.py              # DDPM forward process + loss
│   │   └── ddim.py              # 5-step DDIM sampler with CFG
│   ├── data/
│   │   ├── dataset.py           # Dataset classes
│   │   ├── degradation.py       # Real-ESRGAN degradation pipeline
│   │   └── text_crop.py         # Affine text cropping
│   ├── inference/
│   │   ├── pipeline.py          # Full inference pipeline
│   │   └── blending.py          # LPF Gaussian blending
│   └── utils/
│       └── helpers.py           # Utility functions
├── train.py                      # Training entry point
├── inference.py                  # Inference entry point
└── requirements.txt
```

## Training

### Quick Debug (Synthetic Data)
```bash
python train.py --config configs/default.yaml --synthetic
```

### Full Training
1. Prepare your dataset in the expected format (see `textsr/data/dataset.py`)
2. Run training:
```bash
python train.py --config configs/default.yaml --data_root /path/to/data
```

### Key Training Details
- **Residual prediction**: The diffusion target is `HR - LR` (not HR itself)
- **Text dropout**: 10% probability of dropping text condition for CFG training
- **Gradient accumulation**: Achieves effective batch size 1024
- **Mixed precision**: FP16 training for memory efficiency

## Inference

### Single Image
```bash
python inference.py \
    --checkpoint checkpoints/textsr_final.pt \
    --input input.jpg \
    --output output.jpg \
    --omega 1.0 \
    --iterative_rounds 1
```

### With Real-ESRGAN Base
```bash
python inference.py \
    --checkpoint checkpoints/textsr_final.pt \
    --input input.jpg \
    --output output.jpg \
    --esrgan_input esrgan_output.jpg \
    --omega 1.0 \
    --scale_factor 2
```

### CFG Guidance Scale (ω)
| Text Source | Recommended ω |
|-------------|--------------|
| Ground truth | 3.0 |
| Good OCR (Aster) | 1.0 |
| Weak OCR (CRNN) | 0.5 |
| No text | 0.0 |

## Citation

```bibtex
@article{textsr2025,
  title={TextSR: Diffusion Super-Resolution with Multilingual OCR Guidance},
  year={2025},
  journal={arXiv preprint arXiv:2505.23119}
}
```

## License

This is a research reproduction. Please refer to the original paper for usage terms.
