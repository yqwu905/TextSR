"""
TextSR Training Script (Multi-GPU DDP)

Usage:
  # Single GPU
  python train.py --config configs/textzoom_small.yaml

  # Multi-GPU DDP (e.g., 4 GPUs)
  torchrun --nproc_per_node=4 train.py --config configs/textzoom_small.yaml

  # Resume from checkpoint
  python train.py --config configs/textzoom_small.yaml --resume checkpoints/epoch_10.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
from tqdm import tqdm

from data.dataset import TextZoomDataset
from models.textsr import TextSR, build_textsr_from_config


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) for model weights
# ---------------------------------------------------------------------------

class EMA:
    """Maintains exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply(self, model: nn.Module):
        """Apply EMA weights to model (for evaluation/saving)."""
        self._backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original model weights after EMA evaluation."""
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    """Initialize DDP process group."""
    if "RANK" not in os.environ:
        # Single GPU / non-distributed
        return 0, 1, False
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size, True


def cleanup_ddp(is_distributed: bool):
    if is_distributed:
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer,
    scheduler,
    ema: EMA,
    cfg,
):
    """Save training checkpoint."""
    # Get underlying model if DDP-wrapped
    raw_model = model.module if hasattr(model, "module") else model

    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "ema_shadow": ema.shadow,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(ckpt, path)
    print(f"  [CKPT] Saved checkpoint: {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None, ema: EMA = None):
    """Load training checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if ema is not None and "ema_shadow" in ckpt:
        ema.shadow = ckpt["ema_shadow"]

    print(f"  [CKPT] Loaded from {path}, epoch={ckpt['epoch']}, step={ckpt['global_step']}")
    return ckpt["epoch"], ckpt["global_step"]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def log_val_images(
    raw_model: nn.Module,
    writer,
    global_step: int,
    cfg,
    device: torch.device,
    n_images: int = 9,
):
    """Inference on fixed test images; log LR / SR / HR side-by-side to TensorBoard."""
    import cv2
    from data.dataset import LMDBReader, pil_to_numpy, numpy_to_tensor

    test_dir = Path(cfg.data.data_root) / "test" / "easy"
    if not (test_dir / "data.mdb").exists():
        return

    try:
        reader = LMDBReader(str(test_dir))
        n_total = len(reader)
        if n_total == 0:
            return

        # Fixed evenly-spaced indices for consistent comparison across epochs
        n_pick = min(n_images, n_total)
        indices = [int(i * n_total / n_pick) for i in range(n_pick)]

        sr_factor = cfg.data.sr_factor
        vis_h, vis_w = 64, 256  # display size for all thumbnails

        raw_model.eval()
        grid_rows = []  # each element: (3, vis_h, vis_w) tensor in [0, 1]

        for idx in indices:
            sample = reader.get(idx)
            lr_pil = sample.get("lr")
            hr_pil = sample.get("hr")
            if lr_pil is None:
                continue

            lr_np = pil_to_numpy(lr_pil)
            H, W = lr_np.shape[:2]
            hr_h, hr_w = H * sr_factor, W * sr_factor

            lr_up_np = cv2.resize(lr_np, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
            lr_up_t = numpy_to_tensor(lr_up_np).unsqueeze(0).to(device)  # (1,3,H,W) [-1,1]

            sr_t = raw_model.super_resolve(
                lr_up_t, texts=None, cfg_weight=1.0,
                num_steps=cfg.model.flow_matching.num_steps,
            )  # (1,3,H,W) [-1,1]

            def to_vis(t):
                """(1,3,H,W) or (3,H,W) in [-1,1]  →  (3,vis_h,vis_w) in [0,1]."""
                if t.dim() == 4:
                    t = t[0]
                arr = ((t.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype("uint8")
                arr = cv2.resize(arr, (vis_w, vis_h), interpolation=cv2.INTER_LINEAR)
                return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

            hr_vis = to_vis(
                numpy_to_tensor(
                    cv2.resize(pil_to_numpy(hr_pil), (hr_w, hr_h), interpolation=cv2.INTER_LINEAR)
                ).unsqueeze(0).to(device)
            ) if hr_pil is not None else torch.zeros(3, vis_h, vis_w)

            grid_rows.extend([to_vis(lr_up_t), to_vis(sr_t), hr_vis])

        if not grid_rows:
            return

        # grid_rows: N*3 tensors of (3, vis_h, vis_w) — order: lr1, sr1, hr1, lr2, …
        writer.add_images("val/lr_sr_hr", torch.stack(grid_rows), global_step)

    except Exception as e:
        print(f"[WARNING] val image logging failed: {e}")
    finally:
        raw_model.train()


def train_epoch(
    model,
    dataloader,
    optimizer,
    scaler,
    scheduler,
    ema: EMA,
    cfg,
    epoch: int,
    global_step: int,
    rank: int,
    world_size: int,
    writer=None,
) -> int:
    model.train()
    raw_model = model.module if hasattr(model, "module") else model

    log_interval = cfg.training.log_interval
    precision = cfg.training.get("precision", "fp32")  # "fp32", "fp16", "bf16"
    grad_clip = cfg.training.get("grad_clip", 1.0)

    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    use_autocast = autocast_dtype is not None
    use_scaler = precision == "fp16"  # bf16 has fp32 dynamic range, no scaler needed

    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main_process(rank))

    for batch in pbar:
        lr_up = batch["lr_up"].cuda(rank, non_blocking=True)
        residual = batch["residual"].cuda(rank, non_blocking=True)
        text_ids = batch["text_ids"].cuda(rank, non_blocking=True)
        text_mask = batch["text_mask"].cuda(rank, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype or torch.float16):
            loss = model(lr_up, residual, text_ids, text_mask)

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in raw_model.parameters() if p.requires_grad], grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in raw_model.parameters() if p.requires_grad], grad_clip
            )
            optimizer.step()

        scheduler.step()

        # EMA update
        ema.update(raw_model)

        loss_val = loss.item()
        total_loss += loss_val
        n_batches += 1
        global_step += 1

        if is_main_process(rank) and global_step % log_interval == 0:
            avg_loss = total_loss / n_batches
            pbar.set_postfix(loss=f"{loss_val:.4f}", avg=f"{avg_loss:.4f}", step=global_step)

            if writer is not None:
                writer.add_scalar("train/loss", loss_val, global_step)
                writer.add_scalar("train/loss_avg", avg_loss, global_step)

    return global_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/textzoom_fm.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb", action="store_true", help="Use WandB logging")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # DDP setup
    rank, world_size, is_distributed = setup_ddp()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process(rank):
        print(f"\n{'='*60}")
        print(f"TextSR Training")
        print(f"  Config: {args.config}")
        print(f"  GPUs: {world_size}")
        print(f"  Batch/GPU: {cfg.training.batch_size}")
        print(f"  Effective batch: {cfg.training.batch_size * world_size}")
        print(f"{'='*60}\n")

    # Build model
    model = build_textsr_from_config(cfg)
    model = model.to(device)

    if is_distributed:
        # Only wrap the U-Net in DDP (text encoder is frozen)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    raw_model = model.module if hasattr(model, "module") else model

    # EMA
    ema = EMA(raw_model, decay=cfg.training.get("ema_decay", 0.9999))

    # Dataset and DataLoader
    tokenizer = raw_model.tokenizer

    train_dataset = TextZoomDataset(
        data_root=cfg.data.data_root,
        split="train",
        hr_size=tuple(cfg.data.train_hr_size),
        sr_factor=cfg.data.sr_factor,
        max_text_len=cfg.data.max_text_len,
        text_drop_prob=cfg.training.get("text_drop_prob", 0.1),
        use_synthetic_lr=False,  # use real TextZoom LR/HR pairs, not synthetic degradation
        num_degrade_rounds=5,
        ocr_cache_dir=cfg.data.get("ocr_annotation_dir", None),
        tokenizer=tokenizer,
    )

    if is_distributed:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    # Only optimize U-Net parameters (text encoder is frozen)
    trainable_params = [p for p in raw_model.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.training.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    # LR scheduler: linear warmup then cosine annealing (per-step)
    total_steps = cfg.training.num_epochs * len(train_loader) // world_size
    warmup_steps = min(cfg.training.get("warmup_steps", 1000), total_steps)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - warmup_steps, 1),
        eta_min=cfg.training.learning_rate * 0.1,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # Mixed precision scaler (only needed for fp16, not bf16)
    precision = cfg.training.get("precision", "fp32")
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))

    # WandB / TensorBoard
    writer = None
    if is_main_process(rank):
        if args.wandb:
            import wandb
            wandb.init(project="textsr", config=OmegaConf.to_container(cfg))
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = os.path.join(cfg.training.output_dir, "logs")
                writer = SummaryWriter(log_dir)
                print(f"  TensorBoard logs: {log_dir}")
            except ImportError:
                pass

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        start_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, scheduler, ema
        )
        start_epoch += 1

    # Create output dirs
    if is_main_process(rank):
        Path(cfg.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.training.output_dir).mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, cfg.training.num_epochs):
        if is_distributed:
            sampler.set_epoch(epoch)

        t0 = time.time()
        global_step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            ema=ema,
            cfg=cfg,
            epoch=epoch,
            global_step=global_step,
            rank=rank,
            world_size=world_size,
            writer=writer,
        )

        if is_main_process(rank):
            elapsed = time.time() - t0
            print(f"Epoch {epoch} done in {elapsed:.1f}s | lr={optimizer.param_groups[0]['lr']:.2e}")

            # Save checkpoint
            if (epoch + 1) % cfg.training.save_interval == 0 or epoch == cfg.training.num_epochs - 1:
                ckpt_path = os.path.join(cfg.training.checkpoint_dir, f"epoch_{epoch:04d}.pt")
                save_checkpoint(ckpt_path, epoch, global_step, model, optimizer, scheduler, ema, cfg)

                # Also save EMA weights separately
                ema_path = os.path.join(cfg.training.checkpoint_dir, f"epoch_{epoch:04d}_ema.pt")
                ema.apply(raw_model)
                torch.save(raw_model.state_dict(), ema_path)
                print(f"  [EMA] Saved EMA weights: {ema_path}")

                # Log 9 val images to TensorBoard using EMA weights
                if writer is not None:
                    log_val_images(raw_model, writer, global_step, cfg, device)

                ema.restore(raw_model)

    if writer is not None:
        writer.close()
    cleanup_ddp(is_distributed)
    print("Training complete!")


if __name__ == "__main__":
    main()
