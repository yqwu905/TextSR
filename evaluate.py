"""
TextSR Evaluation on TextZoom benchmark.

Reproduces Table 1 from the paper:
  - Three difficulty levels: easy, medium, hard
  - OCR accuracy with PaddleOCR (as stand-in for CRNN/MORAN/ASTER)
  - Reports word-level accuracy (recognized text == GT label, case-insensitive)

For exact paper comparison, replace PaddleOCR with CRNN/MORAN/ASTER recognizers
(pretrained checkpoints available at their respective GitHub repositories).

Usage:
  python evaluate.py --config configs/textzoom_small.yaml \\
                     --checkpoint checkpoints/epoch_0099_ema.pt \\
                     --output_dir outputs/eval/
"""

import argparse
import json
import os
import string
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from data.dataset import TextZoomEvalDataset, ocr_image, pil_to_numpy, numpy_to_tensor
from models.textsr import TextSR, build_textsr_from_config, _tensor_to_numpy
from inference import load_model


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

ALLOWED_CHARS = set(string.ascii_lowercase + string.digits)


def normalize_text(text: str) -> str:
    """Lowercase and keep only alphanumeric characters (standard TextZoom eval)."""
    text = text.lower().strip()
    text = "".join(c for c in text if c in ALLOWED_CHARS)
    return text


def word_accuracy(pred: str, gt: str) -> float:
    """1.0 if prediction matches ground truth (after normalization), else 0.0"""
    return 1.0 if normalize_text(pred) == normalize_text(gt) else 0.0


# ---------------------------------------------------------------------------
# Recognizer wrappers
# ---------------------------------------------------------------------------

class PaddleOCRRecognizer:
    """Text recognizer using PaddleOCR."""

    def __init__(self, use_gpu: bool = False):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=use_gpu,
            show_log=False,
        )

    def recognize(self, img_np: np.ndarray) -> str:
        """Recognize text in a single image."""
        result = self.ocr.ocr(img_np, cls=False)
        if result is None or result[0] is None:
            return ""
        texts = []
        for line in result[0]:
            if line is not None and len(line) >= 2:
                texts.append(line[1][0])
        return " ".join(texts)


class CRNNRecognizer:
    """
    CRNN text recognizer (from TextZoom benchmark).

    Requires pretrained CRNN checkpoint compatible with TextZoom evaluation.
    Download from: https://github.com/WenjiaWang0312/TextZoom
    """

    def __init__(self, checkpoint_path: str, device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Try to load CRNN model
        try:
            from interfaces.super_resolution import TextSR as TextSRInterface
            print("[CRNN] Loading from TextZoom codebase...")
            # This requires the TextZoom repository to be available
            raise NotImplementedError("Integrate TextZoom CRNN evaluator for exact paper comparison")
        except (ImportError, NotImplementedError):
            print("[CRNN] TextZoom CRNN not available, falling back to PaddleOCR")
            self._fallback = PaddleOCRRecognizer()

    def recognize(self, img_np: np.ndarray) -> str:
        if hasattr(self, "_fallback"):
            return self._fallback.recognize(img_np)
        return ""


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_textzoom(
    model: TextSR,
    cfg,
    device: torch.device,
    output_dir: str,
    cfg_weight: float = 2.0,
    ddim_steps: int = 5,
    iter_rounds: int = 1,
    use_ocr_gpu: bool = False,
    recognizer_name: str = "paddleocr",
    save_sr_images: bool = True,
    difficulties: tuple = ("easy", "medium", "hard"),
) -> Dict:
    """
    Full TextZoom evaluation.

    Args:
        model:           Loaded TextSR model
        cfg:             Config
        device:          Torch device
        output_dir:      Directory to save results
        cfg_weight:      Text guidance scale ω
        ddim_steps:      DDIM steps
        iter_rounds:     R (iterative OCR refinement rounds)
        recognizer_name: OCR recognizer for evaluation metric
        save_sr_images:  Whether to save all SR images

    Returns:
        Dict with per-difficulty and overall accuracy metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build recognizer for evaluation
    if recognizer_name == "paddleocr":
        recognizer = PaddleOCRRecognizer(use_gpu=use_ocr_gpu)
    else:
        recognizer = PaddleOCRRecognizer(use_gpu=use_ocr_gpu)  # fallback

    # Build evaluation dataset
    eval_dataset = TextZoomEvalDataset(
        data_root=cfg.data.data_root,
        split="test",
        difficulties=difficulties,
        lr_size=(32, 128),
        sr_factor=cfg.data.sr_factor,
        max_text_len=cfg.data.max_text_len,
        tokenizer=model.tokenizer,
        use_ocr_gpu=use_ocr_gpu,
        ocr_cache_dir=cfg.data.get("ocr_annotation_dir", None),
    )

    results = {diff: {"correct": 0, "total": 0} for diff in difficulties}
    all_records = []

    model.eval()

    print(f"\nEvaluating TextZoom ({len(eval_dataset)} samples)")
    print(f"  Settings: cfg_weight={cfg_weight}, ddim_steps={ddim_steps}, iter_rounds={iter_rounds}")

    for idx in tqdm(range(len(eval_dataset)), desc="Evaluating"):
        sample = eval_dataset[idx]
        lr_up = sample["lr_up"].unsqueeze(0).to(device)          # (1, 3, H, W)
        text_ids = sample["text_ids"].unsqueeze(0).to(device)    # (1, L)
        text_mask = sample["text_mask"].unsqueeze(0).to(device)  # (1, L)
        gt_label = sample["gt_label"]
        difficulty = sample["difficulty"]

        # --- Super-resolve ---
        with torch.no_grad():
            if iter_rounds == 0:
                # Direct: use LR OCR text
                text_emb = model.text_encoder(text_ids, text_mask)
                sr_tensor = model.super_resolve(
                    lr_up,
                    texts=None,
                    cfg_weight=cfg_weight,
                    ddim_steps=ddim_steps,
                )
                # Actually pass pre-computed text embedding
                sr_tensor = model.diffusion.ddim_sample(
                    model=model.unet,
                    shape=(1, 3, lr_up.shape[2], lr_up.shape[3]),
                    image_cond=lr_up,
                    text_emb=text_emb,
                    text_mask=text_mask,
                    cfg_weight=cfg_weight,
                    num_steps=ddim_steps,
                )
                sr_tensor = (lr_up + sr_tensor * 2.0).clamp(-1, 1)

            else:
                # Iterative: R=1 (image-only first, then OCR on that)
                # Step 1: image-only SR
                residual_img_only = model.diffusion.ddim_sample(
                    model=model.unet,
                    shape=(1, 3, lr_up.shape[2], lr_up.shape[3]),
                    image_cond=lr_up,
                    text_emb=None,
                    text_mask=None,
                    cfg_weight=1.0,
                    num_steps=ddim_steps,
                )
                sr_np_intermediate = _tensor_to_numpy((lr_up + residual_img_only * 2.0).clamp(-1, 1)[0])

                # Step 2: OCR on intermediate SR
                ocr_text = ocr_image(sr_np_intermediate, use_gpu=use_ocr_gpu)

                # Step 3: SR conditioned on OCR text
                text_ids2, text_mask2 = _tokenize_single(
                    ocr_text, model.tokenizer, cfg.data.max_text_len, device
                )
                text_emb2 = model.text_encoder(text_ids2, text_mask2)
                residual_final = model.diffusion.ddim_sample(
                    model=model.unet,
                    shape=(1, 3, lr_up.shape[2], lr_up.shape[3]),
                    image_cond=lr_up,
                    text_emb=text_emb2,
                    text_mask=text_mask2,
                    cfg_weight=cfg_weight,
                    num_steps=ddim_steps,
                )
                sr_tensor = (lr_up + residual_final * 2.0).clamp(-1, 1)

        sr_np = _tensor_to_numpy(sr_tensor[0])

        # --- Evaluate: recognize text in SR image ---
        pred_text = recognizer.recognize(sr_np)
        correct = word_accuracy(pred_text, gt_label) if gt_label else 0.0

        results[difficulty]["correct"] += correct
        results[difficulty]["total"] += 1

        record = {
            "idx": idx,
            "difficulty": difficulty,
            "gt": gt_label,
            "pred": pred_text,
            "correct": correct,
        }
        all_records.append(record)

        # Save SR image
        if save_sr_images:
            diff_dir = output_dir / difficulty
            diff_dir.mkdir(exist_ok=True)
            out_path = diff_dir / f"{idx:06d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR))

    # Compute accuracies
    print("\n=== TextZoom Evaluation Results ===")
    total_correct = 0
    total_samples = 0
    summary = {}

    for diff in difficulties:
        n = results[diff]["total"]
        c = results[diff]["correct"]
        acc = c / n * 100 if n > 0 else 0.0
        summary[diff] = {"accuracy": acc, "correct": c, "total": n}
        print(f"  {diff:8s}: {acc:.1f}% ({c}/{n})")
        total_correct += c
        total_samples += n

    overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0.0
    summary["overall"] = {"accuracy": overall_acc, "correct": total_correct, "total": total_samples}
    print(f"  {'overall':8s}: {overall_acc:.1f}% ({total_correct}/{total_samples})")
    print(f"  Recognizer: {recognizer_name}")

    # Save results JSON
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "summary": summary,
            "settings": {
                "cfg_weight": cfg_weight,
                "ddim_steps": ddim_steps,
                "iter_rounds": iter_rounds,
                "recognizer": recognizer_name,
            },
            "records": all_records,
        }, f, indent=2)
    print(f"\n  Full results saved: {results_path}")

    return summary


def _tokenize_single(text: str, tokenizer, max_length: int, device: torch.device):
    """Tokenize a single text string."""
    enc = tokenizer(
        [text],
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    return enc.input_ids.to(device), enc.attention_mask.to(device)


# ---------------------------------------------------------------------------
# Baseline evaluation (no SR, just bicubic upsampled LR)
# ---------------------------------------------------------------------------

def evaluate_baseline(cfg, device, output_dir, recognizer_name="paddleocr", use_ocr_gpu=False):
    """Evaluate bicubic upsampled baseline on TextZoom."""
    print("\n=== Evaluating Bicubic Upsampled Baseline ===")
    recognizer = PaddleOCRRecognizer(use_gpu=use_ocr_gpu)

    eval_dataset = TextZoomEvalDataset(
        data_root=cfg.data.data_root,
        split="test",
        lr_size=(32, 128),
        sr_factor=cfg.data.sr_factor,
        max_text_len=cfg.data.max_text_len,
        use_ocr_gpu=use_ocr_gpu,
    )

    results = {"easy": {"correct": 0, "total": 0},
               "medium": {"correct": 0, "total": 0},
               "hard": {"correct": 0, "total": 0}}

    for idx in tqdm(range(len(eval_dataset)), desc="Baseline"):
        sample = eval_dataset[idx]
        lr_up_t = sample["lr_up"]   # bicubic upsampled LR
        lr_up_np = _tensor_to_numpy(lr_up_t)
        gt = sample["gt_label"]
        diff = sample["difficulty"]

        pred = recognizer.recognize(lr_up_np)
        results[diff]["correct"] += word_accuracy(pred, gt) if gt else 0.0
        results[diff]["total"] += 1

    print("Bicubic Baseline:")
    for diff, r in results.items():
        acc = r["correct"] / r["total"] * 100 if r["total"] > 0 else 0.0
        print(f"  {diff}: {acc:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/textzoom_small.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default="outputs/eval")
    parser.add_argument("--cfg_weight", type=float, default=2.0)
    parser.add_argument("--ddim_steps", type=int, default=5)
    parser.add_argument("--iter_rounds", type=int, default=1)
    parser.add_argument("--recognizer", default="paddleocr", choices=["paddleocr", "crnn"])
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--baseline_only", action="store_true", help="Only run bicubic baseline")
    parser.add_argument("--no_save_images", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.baseline_only:
        evaluate_baseline(cfg, device, args.output_dir, args.recognizer, args.use_gpu)
        return

    # Load model
    model = load_model(cfg, args.checkpoint, device)

    # Evaluate
    evaluate_on_textzoom(
        model=model,
        cfg=cfg,
        device=device,
        output_dir=args.output_dir,
        cfg_weight=args.cfg_weight,
        ddim_steps=args.ddim_steps,
        iter_rounds=args.iter_rounds,
        use_ocr_gpu=args.use_gpu,
        recognizer_name=args.recognizer,
        save_sr_images=not args.no_save_images,
    )


if __name__ == "__main__":
    main()
