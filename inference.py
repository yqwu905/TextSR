"""
TextSR Inference Script

Applies super-resolution to single images or directories.
Supports iterative OCR refinement (Algorithm from paper).

Usage:
  # Single image
  python inference.py --config configs/textzoom_small.yaml \\
                      --checkpoint checkpoints/epoch_0099_ema.pt \\
                      --input path/to/lr_image.png \\
                      --output outputs/sr_image.png \\
                      --cfg_weight 2.0 --iter_rounds 1

  # Directory of LR images
  python inference.py --config configs/textzoom_small.yaml \\
                      --checkpoint checkpoints/epoch_0099_ema.pt \\
                      --input_dir data/TextZoom/test/easy/lr_imgs/ \\
                      --output_dir outputs/easy_sr/
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from models.textsr import TextSR, build_textsr_from_config, _numpy_to_tensor, _tensor_to_numpy
from data.dataset import ocr_image


def load_model(cfg, checkpoint_path: str, device: torch.device) -> TextSR:
    """Load TextSR model from config and checkpoint."""
    model = build_textsr_from_config(cfg)
    model = model.to(device)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        # Handle both full checkpoint and state_dict
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("[WARNING] No checkpoint loaded - using random weights!")

    model.eval()
    return model


def process_single_image(
    model: TextSR,
    lr_path: str,
    output_path: str,
    sr_factor: int = 2,
    cfg_weight: float = 2.0,
    num_steps: int = 10,
    iter_rounds: int = 1,
    use_ocr_gpu: bool = False,
    device: torch.device = None,
    blend_with_esrgan: bool = False,
):
    """
    Super-resolve a single LR image.

    Args:
        lr_path:           Path to input LR image
        output_path:       Path to save SR output
        sr_factor:         Upscaling factor
        cfg_weight:        ω - text guidance scale (0.0=no text, 2.0=paper default)
        num_steps:         Number of Euler ODE steps
        iter_rounds:       R - iterative OCR refinement rounds
        blend_with_esrgan: Blend high-freq (TextSR) with low-freq (Real-ESRGAN)
    """
    if device is None:
        device = next(model.parameters()).device

    # Load LR image
    lr_bgr = cv2.imread(lr_path)
    if lr_bgr is None:
        print(f"[ERROR] Cannot read: {lr_path}")
        return
    lr_rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)

    # OCR function
    def ocr_fn(img_np: np.ndarray) -> str:
        return ocr_image(img_np, use_gpu=use_ocr_gpu)

    # Super-resolve with iterative OCR refinement
    sr_rgb = model.super_resolve_iterative(
        lr_np=lr_rgb,
        ocr_fn=ocr_fn,
        sr_factor=sr_factor,
        cfg_weight=cfg_weight,
        num_steps=num_steps,
        num_rounds=iter_rounds,
        device=device,
    )

    # Optional: blend with Real-ESRGAN for better color/structure
    if blend_with_esrgan:
        sr_rgb = blend_with_realesrgan(lr_rgb, sr_rgb, sr_factor)

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sr_bgr = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, sr_bgr)
    print(f"  Saved: {output_path}")


def blend_with_realesrgan(
    lr_rgb: np.ndarray,
    sr_rgb: np.ndarray,
    sr_factor: int = 2,
    sigma: float = 3.0,
) -> np.ndarray:
    """
    Blend TextSR high-frequency details with Real-ESRGAN low-frequency components.

    Paper formula: T_i' = LPF(f(T_i)) + [g(T_i) - LPF(g(T_i))]
    where f = Real-ESRGAN, g = TextSR, LPF = low-pass filter (Gaussian)

    Args:
        lr_rgb:    Original LR image
        sr_rgb:    TextSR output
        sr_factor: Upscaling factor
        sigma:     Gaussian blur sigma for LPF
    """
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        # Load Real-ESRGAN (will download model weights automatically)
        esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        upsampler = RealESRGANer(
            scale=sr_factor,
            model_path=f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x{sr_factor}plus.pth",
            model=esrgan_model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
        )
        esrgan_output, _ = upsampler.enhance(
            cv2.cvtColor(lr_rgb, cv2.COLOR_RGB2BGR), outscale=sr_factor
        )
        esrgan_rgb = cv2.cvtColor(esrgan_output, cv2.COLOR_BGR2RGB).astype(np.float32)
    except Exception as e:
        print(f"[WARNING] Real-ESRGAN unavailable ({e}), skipping blend")
        return sr_rgb

    # Low-pass filter: Gaussian blur
    ksize = int(6 * sigma + 1) | 1  # ensure odd
    sr_f = sr_rgb.astype(np.float32)
    sr_lpf = cv2.GaussianBlur(sr_f, (ksize, ksize), sigma)
    esrgan_lpf = cv2.GaussianBlur(esrgan_rgb, (ksize, ksize), sigma)

    # Blend: LPF(ESRGAN) + HPF(TextSR)
    blended = esrgan_lpf + (sr_f - sr_lpf)
    return np.clip(blended, 0, 255).astype(np.uint8)


def process_directory(
    model: TextSR,
    input_dir: str,
    output_dir: str,
    sr_factor: int = 2,
    cfg_weight: float = 2.0,
    num_steps: int = 10,
    iter_rounds: int = 1,
    use_ocr_gpu: bool = False,
    device: torch.device = None,
    extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp"),
):
    """Process all images in a directory, or an LMDB directory (TextZoom format)."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect LMDB: TextZoom stores images in data.mdb / lock.mdb
    if (input_dir / "data.mdb").exists():
        _process_lmdb(
            model=model,
            lmdb_dir=input_dir,
            output_dir=output_dir,
            sr_factor=sr_factor,
            cfg_weight=cfg_weight,
            num_steps=num_steps,
            iter_rounds=iter_rounds,
            use_ocr_gpu=use_ocr_gpu,
            device=device,
        )
        return

    image_files = [
        f for f in sorted(input_dir.iterdir())
        if f.suffix.lower() in extensions
    ]

    if not image_files:
        print(f"[WARNING] No images found in {input_dir} (not an LMDB dir either)")
        return

    print(f"Processing {len(image_files)} images from {input_dir}")
    print(f"  SR factor: {sr_factor}×, CFG weight: {cfg_weight}, ODE steps: {num_steps}, Iter rounds: {iter_rounds}")

    for img_path in tqdm(image_files, desc="Super-resolving"):
        out_path = output_dir / img_path.name
        process_single_image(
            model=model,
            lr_path=str(img_path),
            output_path=str(out_path),
            sr_factor=sr_factor,
            cfg_weight=cfg_weight,
            num_steps=num_steps,
            iter_rounds=iter_rounds,
            use_ocr_gpu=use_ocr_gpu,
            device=device,
        )


def _process_lmdb(
    model: TextSR,
    lmdb_dir: Path,
    output_dir: Path,
    sr_factor: int,
    cfg_weight: float,
    num_steps: int,
    iter_rounds: int,
    use_ocr_gpu: bool,
    device: torch.device,
):
    """Read LR images from a TextZoom LMDB and write SR results as PNG files."""
    from data.dataset import LMDBReader, pil_to_numpy

    reader = LMDBReader(str(lmdb_dir))
    n = len(reader)
    print(f"Processing {n} samples from LMDB: {lmdb_dir}")
    print(f"  SR factor: {sr_factor}×, CFG weight: {cfg_weight}, ODE steps: {num_steps}, Iter rounds: {iter_rounds}")

    def ocr_fn(img_np: np.ndarray) -> str:
        return ocr_image(img_np, use_gpu=use_ocr_gpu)

    for idx in tqdm(range(n), desc="Super-resolving"):
        sample = reader.get(idx)
        lr_pil = sample.get("lr")
        if lr_pil is None:
            continue

        lr_np = pil_to_numpy(lr_pil)

        sr_np = model.super_resolve_iterative(
            lr_np=lr_np,
            ocr_fn=ocr_fn,
            sr_factor=sr_factor,
            cfg_weight=cfg_weight,
            num_steps=num_steps,
            num_rounds=iter_rounds,
            device=device,
        )

        out_path = output_dir / f"{idx:06d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR))


def batch_inference_from_lmdb(
    model: TextSR,
    lmdb_dir: str,
    output_dir: str,
    cfg,
    device: torch.device,
    cfg_weight: float = 2.0,
    num_steps: int = 10,
    iter_rounds: int = 1,
):
    """
    Run inference on TextZoom LMDB test set.
    Used for evaluation.
    """
    from data.dataset import TextZoomEvalDataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_dataset = TextZoomEvalDataset(
        data_root=str(Path(lmdb_dir).parent),
        split="test",
        lr_size=(32, 128),
        sr_factor=cfg.data.sr_factor,
        max_text_len=cfg.data.max_text_len,
        tokenizer=model.tokenizer,
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=2)

    all_results = []

    for idx, batch in enumerate(tqdm(loader, desc="Inference")):
        lr_up = batch["lr_up"].to(device)
        text_ids = batch["text_ids"].to(device)
        text_mask = batch["text_mask"].to(device)
        gt_label = batch["gt_label"][0]
        difficulty = batch["difficulty"][0]

        # Get text embeddings
        with torch.no_grad():
            text_emb = model.text_encoder(text_ids, text_mask)

        # SR inference
        with torch.no_grad():
            if iter_rounds == 0:
                sr_tensor = model.super_resolve(
                    lr_up, cfg_weight=cfg_weight, num_steps=num_steps
                )
            else:
                # Iterative: first image-only, then with text
                sr_tensor = model.super_resolve(lr_up, cfg_weight=1.0, num_steps=num_steps)

                # Re-run OCR on SR image
                sr_np = _tensor_to_numpy(sr_tensor[0])
                ocr_text = ocr_image(sr_np)

                # Final SR with OCR text
                with torch.no_grad():
                    text_ids2, text_mask2 = model.tokenizer(
                        [ocr_text],
                        return_tensors="pt",
                        padding="max_length",
                        max_length=model.max_text_len,
                        truncation=True,
                    ).values()
                    text_emb2 = model.text_encoder(text_ids2.to(device), text_mask2.to(device))

                sr_tensor = model.super_resolve(
                    lr_up,
                    texts=[ocr_text],
                    cfg_weight=cfg_weight,
                    num_steps=num_steps,
                )

        sr_np = _tensor_to_numpy(sr_tensor[0])

        # Save SR image
        out_path = output_dir / f"{idx:06d}_{difficulty}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR))

        all_results.append({
            "idx": idx,
            "difficulty": difficulty,
            "gt_label": gt_label,
            "output_path": str(out_path),
        })

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TextSR Inference")
    parser.add_argument("--config", default="configs/textzoom_fm.yaml")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    # Single image mode
    parser.add_argument("--input", default=None, help="Input LR image path")
    parser.add_argument("--output", default=None, help="Output SR image path")
    # Directory mode
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    # Inference settings
    parser.add_argument("--sr_factor", type=int, default=2)
    parser.add_argument("--cfg_weight", type=float, default=2.0, help="Text guidance scale ω")
    parser.add_argument("--num_steps", type=int, default=10, help="Euler ODE integration steps")
    parser.add_argument("--iter_rounds", type=int, default=1, help="R=0,1,2... OCR refinement rounds")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--blend_esrgan", action="store_true", help="Blend with Real-ESRGAN")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(cfg, args.checkpoint, device)

    if args.input is not None:
        output_path = args.output or args.input.replace(".png", "_sr.png").replace(".jpg", "_sr.jpg")
        process_single_image(
            model=model,
            lr_path=args.input,
            output_path=output_path,
            sr_factor=args.sr_factor,
            cfg_weight=args.cfg_weight,
            num_steps=args.num_steps,
            iter_rounds=args.iter_rounds,
            use_ocr_gpu=args.use_gpu,
            device=device,
            blend_with_esrgan=args.blend_esrgan,
        )
    elif args.input_dir is not None:
        output_dir = args.output_dir or (args.input_dir + "_sr")
        process_directory(
            model=model,
            input_dir=args.input_dir,
            output_dir=output_dir,
            sr_factor=args.sr_factor,
            cfg_weight=args.cfg_weight,
            num_steps=args.num_steps,
            iter_rounds=args.iter_rounds,
            use_ocr_gpu=args.use_gpu,
            device=device,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
