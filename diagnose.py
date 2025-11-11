#!/usr/bin/env python3
"""
Diagnostic script to debug why cross loss doesn't decrease.

This script will:
1. Load a trained checkpoint (or train for a few steps)
2. Analyze the forward pass in detail
3. Check gradient flow
4. Identify the bottleneck
"""

import torch
import json
from pathlib import Path
from data_loader import build_c2c_dataloaders
from train import build_models, build_optimizer
from model import Contrast2ContrastTrainer

def diagnose_training(config_path="config.json", num_steps=10):
    """Run diagnostic checks on training."""

    print("="*80)
    print("DIAGNOSIS: Cross Loss Not Decreasing")
    print("="*80)

    # Load config
    with open(config_path) as f:
        cfg = json.load(f)

    device = torch.device(cfg["device"])

    # Build data loaders
    print("\n[1/6] Loading data...")
    dataloaders = build_c2c_dataloaders(cfg["data"])
    train_loader = dataloaders["train"]
    sample_batch = next(iter(train_loader))

    # Build models
    print("[2/6] Building models...")
    encoder, decoder_a, decoder_b = build_models(cfg["model"], sample_batch, cfg["trainer"])
    encoder.to(device)
    decoder_a.to(device)
    decoder_b.to(device)

    # Build optimizer
    params = list(encoder.parameters()) + list(decoder_a.parameters()) + list(decoder_b.parameters())
    optimizer = build_optimizer(cfg["optimizer"]["type"], params, cfg["optimizer"])

    # Build trainer
    trainer = Contrast2ContrastTrainer(
        encoder=encoder,
        decoder_a=decoder_a,
        decoder_b=decoder_b,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        config=cfg["trainer"],
        wandb_run=None,
    )

    # Run a few training steps with detailed logging
    print(f"[3/6] Running {num_steps} training steps with diagnostics...\n")

    encoder.train()
    decoder_a.train()
    decoder_b.train()

    for step, batch in enumerate(train_loader):
        if step >= num_steps:
            break

        print(f"\n{'─'*80}")
        print(f"Step {step+1}/{num_steps}")
        print(f"{'─'*80}")

        # Prepare batch
        x_a, x_b, scale_a, scale_b = trainer._prepare_batch(batch)
        x_a_noisy, x_b_noisy = trainer._apply_augmentations(x_a, x_b)

        print(f"✓ Batch prepared:")
        print(f"  x_a shape: {x_a.shape}, dtype: {x_a.dtype}")
        print(f"  x_b shape: {x_b.shape}, dtype: {x_b.dtype}")
        print(f"  scale_a: {scale_a.mean().item():.4f}")
        print(f"  scale_b: {scale_b.mean().item():.4f}")

        # Forward pass
        enc_a = trainer._run_encoder(x_a_noisy, None, return_full=True)
        enc_b = trainer._run_encoder(x_b_noisy, None, return_full=True)
        z_a, skips_a, identity_a = trainer._split_encoder_output(enc_a)
        z_b, skips_b, identity_b = trainer._split_encoder_output(enc_b)

        print(f"\n✓ Encoded:")
        print(f"  z_a: shape={z_a.shape}, mean={z_a.abs().mean().item():.4f}, std={z_a.abs().std().item():.4f}")
        print(f"  z_b: shape={z_b.shape}, mean={z_b.abs().mean().item():.4f}, std={z_b.abs().std().item():.4f}")
        print(f"  z_a ≈ z_b distance: {(z_a - z_b).abs().mean().item():.6f}")
        if skips_a:
            print(f"  skips_a: {len(skips_a)} skip connections")

        # Same-domain reconstruction
        x_a_recon = trainer._run_decoder(decoder_a, z_a, skips=skips_a, identity=identity_a)
        x_b_recon = trainer._run_decoder(decoder_b, z_b, skips=skips_b, identity=identity_b)

        print(f"\n✓ Same-domain reconstruction:")
        print(f"  x_a_recon: shape={x_a_recon.shape}, mean={x_a_recon.abs().mean().item():.4f}")
        print(f"  recon error A: {(x_a_recon - x_a).abs().mean().item():.6f}")
        print(f"  recon error B: {(x_b_recon - x_b).abs().mean().item():.6f}")

        # Cross-domain reconstruction
        cross_skips_for_a, cross_identity_for_a = trainer._prepare_cross_skips(
            skips_a, skips_b, identity_a, identity_b, target_domain="a"
        )
        cross_skips_for_b, cross_identity_for_b = trainer._prepare_cross_skips(
            skips_b, skips_a, identity_b, identity_a, target_domain="b"
        )

        print(f"\n✓ Cross-domain strategy: {trainer.cross_domain_strategy}")
        print(f"  cross_skips_for_a: {cross_skips_for_a}")
        print(f"  cross_identity_for_a: {cross_identity_for_a}")

        x_a_from_b = trainer._run_decoder(decoder_a, z_b, skips=cross_skips_for_a, identity=cross_identity_for_a)
        x_b_from_a = trainer._run_decoder(decoder_b, z_a, skips=cross_skips_for_b, identity=cross_identity_for_b)

        print(f"\n✓ Cross-domain reconstruction:")
        print(f"  x_a_from_b: shape={x_a_from_b.shape}, mean={x_a_from_b.abs().mean().item():.4f}")
        print(f"  x_b_from_a: shape={x_b_from_a.shape}, mean={x_b_from_a.abs().mean().item():.4f}")

        # Compute losses
        from model.utils import compute_l1_loss

        l_content = compute_l1_loss(z_a, z_b)
        l_recon = compute_l1_loss(x_a_recon, x_a) + compute_l1_loss(x_b_recon, x_b)

        # Cross loss with scaling
        x_a_target_scaled = x_a * scale_a
        x_b_target_scaled = x_b * scale_b
        x_a_from_b_scaled = x_a_from_b * scale_a
        x_b_from_a_scaled = x_b_from_a * scale_b

        l_cross = compute_l1_loss(x_a_from_b_scaled, x_a_target_scaled) + compute_l1_loss(
            x_b_from_a_scaled, x_b_target_scaled
        )

        print(f"\n✓ Losses:")
        print(f"  Content loss: {l_content.item():.6f}")
        print(f"  Recon loss:   {l_recon.item():.6f}")
        print(f"  Cross loss:   {l_cross.item():.6f} ← KEY METRIC")

        # Check cross-domain reconstruction quality
        cross_error_a = (x_a_from_b - x_a).abs().mean().item()
        cross_error_b = (x_b_from_a - x_b).abs().mean().item()
        print(f"\n✓ Cross reconstruction errors (unscaled):")
        print(f"  |decoder_a(z_b) - x_a|: {cross_error_a:.6f}")
        print(f"  |decoder_b(z_a) - x_b|: {cross_error_b:.6f}")

        # Compute gradients
        total_loss = (
            trainer.state.loss_weights["content"] * l_content +
            trainer.state.loss_weights["recon"] * l_recon +
            trainer.state.loss_weights["cross"] * l_cross
        )

        optimizer.zero_grad()
        total_loss.backward()

        # Check gradient magnitudes
        encoder_grad = torch.cat([p.grad.flatten() for p in encoder.parameters() if p.grad is not None])
        decoder_a_grad = torch.cat([p.grad.flatten() for p in decoder_a.parameters() if p.grad is not None])
        decoder_b_grad = torch.cat([p.grad.flatten() for p in decoder_b.parameters() if p.grad is not None])

        print(f"\n✓ Gradient magnitudes:")
        print(f"  Encoder:   {encoder_grad.abs().mean().item():.8f}")
        print(f"  Decoder A: {decoder_a_grad.abs().mean().item():.8f}")
        print(f"  Decoder B: {decoder_b_grad.abs().mean().item():.8f}")

        optimizer.step()

    print(f"\n{'='*80}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*80}\n")

    # Summary
    print("KEY FINDINGS:")
    print("1. Check if z_a ≈ z_b (content loss should be small)")
    print("2. Check if same-domain recon works (recon loss decreasing)")
    print("3. Check if cross-domain recon improves (cross loss should decrease)")
    print("4. Check gradient flow (all modules should have gradients)")
    print("\nIf cross loss doesn't decrease:")
    print("  - Decoder might be overfitting to skip connections")
    print("  - Latent might encode domain-specific noise")
    print("  - Loss weights might be imbalanced")

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    diagnose_training(config_path, num_steps)
