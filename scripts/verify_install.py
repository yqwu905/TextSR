"""
Quick verification script - check if all components initialize correctly.
Run this before training to catch any import/setup issues.

Usage:
  python scripts/verify_install.py
"""

import sys
import traceback


def check(name, fn):
    try:
        result = fn()
        print(f"  [OK]   {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("TextSR Installation Verification")
    print("=" * 60)
    failures = []

    # --- Core dependencies ---
    print("\n[1] Core dependencies:")

    if not check("torch", lambda: __import__("torch").__version__):
        failures.append("torch")
    if not check("CUDA available", lambda: str(__import__("torch").cuda.is_available())):
        pass  # Not a hard failure
    if not check("torchvision", lambda: __import__("torchvision").__version__):
        failures.append("torchvision")
    if not check("transformers", lambda: __import__("transformers").__version__):
        failures.append("transformers")
    if not check("omegaconf", lambda: __import__("omegaconf").__version__):
        failures.append("omegaconf")
    if not check("einops", lambda: __import__("einops").__version__):
        failures.append("einops")
    if not check("cv2", lambda: __import__("cv2").__version__):
        failures.append("cv2")
    if not check("lmdb", lambda: __import__("lmdb").__version__):
        failures.append("lmdb")

    # --- Optional dependencies ---
    print("\n[2] Optional dependencies:")
    check("paddleocr", lambda: __import__("paddleocr") and "ok")
    check("realesrgan", lambda: __import__("realesrgan") and "ok")
    check("wandb", lambda: __import__("wandb").__version__)

    # --- Model initialization ---
    print("\n[3] Model initialization:")

    def init_unet():
        import torch
        sys.path.insert(0, ".")
        from models.unet import UNet
        model = UNet(
            in_channels=6, out_channels=3, base_channels=16,
            channel_mult=(1, 2, 4, 8, 8), num_res_blocks=1,
            attention_levels=(3, 4), text_context_dim=512,
        )
        x = torch.randn(2, 6, 48, 480)
        t = torch.randint(0, 1000, (2,))
        ctx = torch.randn(2, 8, 512)
        y = model(x, t, ctx)
        return f"UNet OK, output shape: {y.shape}"

    if not check("UNet (small)", init_unet):
        failures.append("UNet")

    def init_diffusion():
        import torch
        sys.path.insert(0, ".")
        from models.diffusion import GaussianDiffusion
        diff = GaussianDiffusion(timesteps=100)
        x0 = torch.randn(2, 3, 48, 480)
        t = torch.randint(0, 100, (2,))
        xt, noise = diff.q_sample(x0, t)
        return f"DDPM OK, xt shape: {xt.shape}"

    if not check("GaussianDiffusion", init_diffusion):
        failures.append("GaussianDiffusion")

    def init_byt5():
        sys.path.insert(0, ".")
        from models.byt5_encoder import ByT5TextEncoder
        import torch
        enc = ByT5TextEncoder(num_layers=2, max_length=32)
        ids = torch.zeros(2, 32, dtype=torch.long)
        mask = torch.ones(2, 32, dtype=torch.long)
        out = enc(ids, mask)
        return f"ByT5 OK, output shape: {out.shape}"

    print("  NOTE: ByT5 test requires downloading model (~580MB). Skipping in fast mode.")
    print("        Run with --full to test ByT5 initialization.")
    if "--full" in sys.argv:
        if not check("ByT5TextEncoder", init_byt5):
            failures.append("ByT5")

    # --- Config load ---
    print("\n[4] Config:")
    def load_cfg():
        from omegaconf import OmegaConf
        cfg = OmegaConf.load("configs/textzoom_small.yaml")
        return f"Loaded, SR factor={cfg.data.sr_factor}"

    if not check("configs/textzoom_small.yaml", load_cfg):
        failures.append("config")

    # --- Summary ---
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILURES ({len(failures)}): {', '.join(failures)}")
        print("Fix the above issues before training.")
    else:
        print("All checks passed!")
        print("\nNext steps:")
        print("  1. Download TextZoom: python scripts/download_textzoom.py")
        print("  2. Pre-compute OCR:   python scripts/prepare_annotations.py")
        print("  3. Start training:    torchrun --nproc_per_node=NUM_GPUS train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
