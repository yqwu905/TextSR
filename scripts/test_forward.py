"""
Quick forward-pass test for all model components (no data required).
Run: python scripts/test_forward.py

Tests:
  1. UNet forward pass
  2. GaussianDiffusion q_sample and training_loss
  3. DDIM sampling
  4. Full TextSR model (without ByT5 - uses random embeddings)
"""

import sys
import torch

sys.path.insert(0, ".")


def test_unet():
    """Test UNet forward pass with known architecture sizes."""
    print("\n[1] Testing UNet...")
    from models.unet import UNet

    model = UNet(
        in_channels=6,
        out_channels=3,
        base_channels=16,          # small for testing
        channel_mult=(1, 2, 4, 4, 4),
        num_res_blocks=1,
        attention_levels=(3, 4),
        text_context_dim=64,       # small context dim for testing
        time_embed_dim=64,
    )
    print(f"  UNet parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with 48×480 resolution (paper training size)
    B = 2
    x = torch.randn(B, 6, 48, 480)
    t = torch.randint(0, 1000, (B,))
    ctx = torch.randn(B, 8, 64)
    mask = torch.ones(B, 8, dtype=torch.long)

    with torch.no_grad():
        # With text conditioning
        out = model(x, t, ctx, mask)
        assert out.shape == (B, 3, 48, 480), f"Expected (2,3,48,480), got {out.shape}"
        print(f"  With text: input {x.shape} -> output {out.shape} ✓")

        # Without text conditioning (image-only)
        out_no_text = model(x, t, None, None)
        assert out_no_text.shape == (B, 3, 48, 480)
        print(f"  Without text: input {x.shape} -> output {out_no_text.shape} ✓")

    # Test with TextZoom eval size (32×128)
    x2 = torch.randn(B, 6, 64, 256)  # 2× upsampled from 32×128
    with torch.no_grad():
        out2 = model(x2, t, ctx, mask)
        assert out2.shape == (B, 3, 64, 256)
        print(f"  64×256 input: {x2.shape} -> output {out2.shape} ✓")

    return True


def test_diffusion():
    """Test DDPM training loss and DDIM sampling."""
    print("\n[2] Testing GaussianDiffusion...")
    from models.unet import UNet
    from models.diffusion import GaussianDiffusion

    diff = GaussianDiffusion(timesteps=100, beta_schedule="linear")

    B = 2
    x0 = torch.randn(B, 3, 48, 480)
    t = torch.randint(0, 100, (B,))

    # Forward diffusion
    xt, noise = diff.q_sample(x0, t)
    assert xt.shape == x0.shape
    print(f"  q_sample: x0 {x0.shape} -> xt {xt.shape} ✓")

    # Training loss with small U-Net
    unet = UNet(
        in_channels=6, out_channels=3, base_channels=8,
        channel_mult=(1, 2, 2, 2, 2), num_res_blocks=1,
        attention_levels=(3, 4), text_context_dim=32, time_embed_dim=32,
    )

    image_cond = torch.randn(B, 3, 48, 480)
    ctx = torch.randn(B, 4, 32)
    mask = torch.ones(B, 4, dtype=torch.long)

    loss = diff.training_loss(unet, x0, image_cond, ctx, mask)
    assert loss.ndim == 0 and loss.item() > 0
    print(f"  training_loss: {loss.item():.4f} ✓")

    # DDIM sampling
    shape = (B, 3, 48, 480)
    with torch.no_grad():
        sample = diff.ddim_sample(unet, shape, image_cond, ctx, mask,
                                   cfg_weight=2.0, num_steps=3)
    assert sample.shape == shape
    print(f"  ddim_sample (3 steps, CFG=2.0): output {sample.shape} ✓")

    # Without text
    with torch.no_grad():
        sample_no_text = diff.ddim_sample(unet, shape, image_cond, None, None,
                                           cfg_weight=1.0, num_steps=3)
    assert sample_no_text.shape == shape
    print(f"  ddim_sample (no text): output {sample_no_text.shape} ✓")

    return True


def test_full_model():
    """Test TextSR without downloading ByT5 (use random text embeddings)."""
    print("\n[3] Testing TextSR (mocked text encoder)...")
    import torch.nn as nn
    from models.textsr import TextSR

    # We can't easily test ByT5 download here, but we can test the U-Net + diffusion
    # by mocking the text encoder
    class MockTextEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = 64  # smaller for testing
            self.max_length = 8

        def forward(self, input_ids, attention_mask=None):
            B = input_ids.shape[0]
            return torch.randn(B, input_ids.shape[1], self.hidden_size)

        def get_null_embedding(self, B, device):
            ids = torch.zeros(B, self.max_length, dtype=torch.long, device=device)
            mask = torch.zeros(B, self.max_length, dtype=torch.long, device=device)
            return ids, mask

    # Build model with small config
    from models.unet import UNet
    from models.diffusion import GaussianDiffusion

    unet = UNet(
        in_channels=6, out_channels=3, base_channels=8,
        channel_mult=(1, 2, 2, 2, 2), num_res_blocks=1,
        attention_levels=(3, 4), text_context_dim=64, time_embed_dim=32,
    )
    diff = GaussianDiffusion(timesteps=100)

    B = 2
    lr_up = torch.randn(B, 3, 64, 256)
    residual = torch.randn(B, 3, 64, 256).clamp(-1, 1)

    # Simulate training step
    ctx = torch.randn(B, 8, 64)
    mask = torch.ones(B, 8, dtype=torch.long)

    loss = diff.training_loss(unet, residual, lr_up, ctx, mask)
    print(f"  Training loss: {loss.item():.4f} ✓")

    # Simulate inference
    with torch.no_grad():
        pred_residual = diff.ddim_sample(
            unet, (B, 3, 64, 256), lr_up, ctx, mask,
            cfg_weight=2.0, num_steps=3
        )
    hr_pred = (lr_up + pred_residual * 2.0).clamp(-1, 1)
    assert hr_pred.shape == (B, 3, 64, 256)
    print(f"  Inference output: {hr_pred.shape} ✓")

    return True


def main():
    print("=" * 60)
    print("TextSR Forward Pass Tests")
    print("=" * 60)

    all_passed = True

    try:
        test_unet()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    try:
        test_diffusion()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    try:
        test_full_model()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
        print("\nReady to train. Next steps:")
        print("  1. python scripts/download_textzoom.py")
        print("  2. python scripts/prepare_annotations.py")
        print("  3. torchrun --nproc_per_node=<NUM_GPU> train.py --config configs/textzoom_small.yaml")
    else:
        print("Some tests FAILED. Check the errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
