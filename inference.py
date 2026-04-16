"""TextSR Inference Script.

Run the full TextSR inference pipeline on images.

Usage:
    # Single image
    python inference.py --checkpoint checkpoints/textsr_final.pt --input image.jpg --output result.jpg

    # Directory of images
    python inference.py --checkpoint checkpoints/textsr_final.pt --input ./input_dir --output ./output_dir

    # With esrgan base & custom omega
    python inference.py --checkpoint checkpoints/textsr_final.pt --input image.jpg --output result.jpg \
        --esrgan_input esrgan_result.jpg --omega 3.0 --iterative_rounds 1
"""

import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from textsr.inference.pipeline import TextSRPipeline
from textsr.utils.helpers import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="TextSR Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML (auto-loaded from checkpoint if not specified)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input image or directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output image or directory")
    parser.add_argument("--esrgan_input", type=str, default=None,
                        help="Real-ESRGAN upscaled image/directory for blending")
    parser.add_argument("--omega", type=float, default=None,
                        help="CFG guidance scale (overrides config)")
    parser.add_argument("--iterative_rounds", type=int, default=None,
                        help="Iterative OCR conditioning rounds R (overrides config)")
    parser.add_argument("--scale_factor", type=int, default=1, choices=[1, 2, 4],
                        help="Upscaling factor")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    return parser.parse_args()


def process_single_image(
    pipeline: TextSRPipeline,
    input_path: str,
    output_path: str,
    esrgan_path: str = None,
    scale_factor: int = 1,
):
    """Process a single image through the TextSR pipeline."""
    logger.info(f"Processing: {input_path}")

    # Load input image
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        logger.error(f"Failed to load image: {input_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load ESRGAN image if provided
    esrgan_image = None
    if esrgan_path and os.path.exists(esrgan_path):
        esrgan_image = cv2.imread(esrgan_path, cv2.IMREAD_COLOR)
        if esrgan_image is not None:
            esrgan_image = cv2.cvtColor(esrgan_image, cv2.COLOR_BGR2RGB)

    # Run pipeline
    result = pipeline.enhance_image(
        image=image,
        esrgan_image=esrgan_image,
        scale_factor=scale_factor,
    )

    # Save result
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, result_bgr)
    logger.info(f"Saved: {output_path}")


def main():
    args = parse_args()

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        # Load from checkpoint
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        config = ckpt.get("config", {})

    # Override inference params if specified
    if args.omega is not None:
        config.setdefault("inference", {})["omega"] = args.omega
    if args.iterative_rounds is not None:
        config.setdefault("inference", {})["iterative_ocr_rounds"] = args.iterative_rounds

    # Build pipeline
    logger.info("Loading TextSR pipeline...")
    pipeline = TextSRPipeline.from_checkpoint(
        args.checkpoint, config=config, device=device
    )
    logger.info("Pipeline loaded successfully!")

    # Process input
    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        # Single image
        esrgan_path = args.esrgan_input
        process_single_image(
            pipeline, str(input_path), str(output_path),
            esrgan_path=esrgan_path,
            scale_factor=args.scale_factor,
        )
    elif input_path.is_dir():
        # Directory of images
        output_path.mkdir(parents=True, exist_ok=True)
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

        for img_file in sorted(input_path.iterdir()):
            if img_file.suffix.lower() not in image_extensions:
                continue

            out_file = output_path / img_file.name
            esrgan_file = None
            if args.esrgan_input:
                esrgan_file = str(Path(args.esrgan_input) / img_file.name)

            process_single_image(
                pipeline, str(img_file), str(out_file),
                esrgan_path=esrgan_file,
                scale_factor=args.scale_factor,
            )
    else:
        logger.error(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()
