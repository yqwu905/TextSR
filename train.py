"""TextSR Training Script.

PyTorch training loop for the TextSR diffusion model.
Uses gradient accumulation to achieve effective batch size of 1024
on limited GPU memory.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --synthetic  # debug mode
"""

import argparse
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from textsr.models.textsr_model import TextSRModel
from textsr.diffusion.ddpm import GaussianDiffusion
from textsr.data.dataset import TextSRDataset, SyntheticTextSRDataset
from textsr.utils.helpers import load_config, count_parameters, AverageMeter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train TextSR Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override data root directory")
    parser.add_argument("--annotation_file", type=str, default=None,
                        help="Override annotation file path")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for debugging")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    return parser.parse_args()


def build_dataloader(config, args):
    """Build training DataLoader."""
    if args.synthetic:
        logger.info("Using SYNTHETIC dataset for debugging")
        dataset = SyntheticTextSRDataset(
            num_samples=2000,
            image_height=config["data"]["image_height"],
            image_width=config["data"]["image_width"],
        )
    else:
        data_root = args.data_root or config.get("paths", {}).get("data_root", "./data")
        annotation_file = args.annotation_file
        dataset = TextSRDataset(
            data_root=data_root,
            annotation_file=annotation_file,
            mode="preprocessed",
            image_height=config["data"]["image_height"],
            image_width=config["data"]["image_width"],
        )

    batch_size = config["training"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def train(config, args):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Build model ---
    logger.info("Building TextSR model...")
    model = TextSRModel.from_config(config)
    model = model.to(device)

    trainable_params = count_parameters(model, trainable_only=True)
    total_params = count_parameters(model, trainable_only=False)
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")

    # --- Build diffusion ---
    diff_cfg = config["diffusion"]
    diffusion = GaussianDiffusion(
        num_timesteps=diff_cfg["num_timesteps"],
        beta_start=diff_cfg["beta_start"],
        beta_end=diff_cfg["beta_end"],
        beta_schedule=diff_cfg["beta_schedule"],
    ).to(device)

    # --- Build optimizer ---
    train_cfg = config["training"]
    optimizer = torch.optim.Adam(
        model.unet.parameters(),  # Only train U-Net (text encoder is frozen)
        lr=train_cfg["learning_rate"],
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    # --- Mixed precision ---
    use_amp = train_cfg.get("mixed_precision", "fp16") == "fp16" and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # --- Gradient accumulation ---
    effective_batch_size = train_cfg.get("effective_batch_size", train_cfg["batch_size"])
    accumulation_steps = max(1, effective_batch_size // train_cfg["batch_size"])
    logger.info(f"Effective batch size: {effective_batch_size} "
                f"(batch={train_cfg['batch_size']} × accum={accumulation_steps})")

    # --- Resume from checkpoint ---
    global_step = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.unet.load_state_dict(ckpt["unet_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt.get("global_step", 0)
        logger.info(f"Resumed at step {global_step}")

    # --- Build DataLoader ---
    loader = build_dataloader(config, args)
    logger.info(f"Dataset size: {len(loader.dataset)}")

    # --- Output dirs ---
    output_dir = Path(config.get("paths", {}).get("output_dir", "./outputs"))
    ckpt_dir = Path(config.get("paths", {}).get("checkpoint_dir", "./checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    max_steps = train_cfg["max_steps"]
    log_interval = train_cfg.get("log_interval", 100)
    save_interval = train_cfg.get("save_interval", 5000)
    text_dropout_prob = train_cfg.get("text_dropout_prob", 0.1)
    gradient_clip = train_cfg.get("gradient_clip", 1.0)

    loss_meter = AverageMeter()
    model.train()
    model.text_encoder.eval()  # Always eval mode for frozen encoder

    logger.info(f"Starting training from step {global_step}...")
    start_time = time.time()

    while global_step < max_steps:
        for batch in loader:
            if global_step >= max_steps:
                break

            hr_images = batch["hr_image"].to(device)
            lr_images = batch["lr_image"].to(device)
            texts = batch["text"]  # List of strings

            # Encode text with frozen ByT5
            with torch.no_grad():
                text_features = model.text_encoder(texts=texts)

            # Forward + loss
            with autocast(enabled=use_amp):
                loss_dict = diffusion.training_loss(
                    model=model.unet,
                    hr_images=hr_images,
                    lr_images=lr_images,
                    text_features=text_features,
                    text_dropout_prob=text_dropout_prob,
                )
                loss = loss_dict["loss"] / accumulation_steps

            # Backward
            scaler.scale(loss).backward()

            # Optimizer step (after accumulation)
            if (global_step + 1) % accumulation_steps == 0:
                if gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.unet.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loss_meter.update(loss_dict["loss"].item())
            global_step += 1

            # Logging
            if global_step % log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Step {global_step}/{max_steps} | "
                    f"Loss: {loss_meter.avg:.6f} | "
                    f"Steps/s: {steps_per_sec:.2f}"
                )
                loss_meter.reset()

            # Save checkpoint
            if global_step % save_interval == 0:
                ckpt_path = ckpt_dir / f"textsr_step{global_step}.pt"
                torch.save({
                    "global_step": global_step,
                    "unet_state_dict": model.unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "model_state_dict": model.state_dict(),
                }, ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}")

    # Final save
    final_path = ckpt_dir / "textsr_final.pt"
    torch.save({
        "global_step": global_step,
        "unet_state_dict": model.unet.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "model_state_dict": model.state_dict(),
    }, final_path)
    logger.info(f"Training complete! Final checkpoint: {final_path}")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config, args)
