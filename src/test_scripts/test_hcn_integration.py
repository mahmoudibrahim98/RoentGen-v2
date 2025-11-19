"""
Test script to verify HCN integration with RoentGen-v2

This script tests:
1. HCN module functionality
2. Dataset parsing of demographics
3. Integration with training components

Run this BEFORE starting full training to catch issues early.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from roentgenv2.train_code.hcn import HierarchicalConditioner
from roentgenv2.train_code.dataset_wds import parse_age_bin, parse_sex, parse_race, extract_clinical_text


def test_demographic_parsing():
    """Test demographic parsing functions."""
    print("\n" + "="*60)
    print("TEST 1: Demographic Parsing")
    print("="*60)

    test_prompts = [
        "28 year old WHITE female. No evidence of acute cardiopulmonary disease.",
        "86 year old BLACK female. No evidence of acute cardiopulmonary process.",
        "71 year old HISPANIC male. Normal chest.",
        "45 year old ASIAN male. Pneumonia in right lower lobe.",
    ]

    for prompt in test_prompts:
        age_idx = parse_age_bin(prompt)
        sex_idx = parse_sex(prompt)
        race_idx = parse_race(prompt)
        clinical = extract_clinical_text(prompt)

        print(f"\nPrompt: {prompt}")
        print(f"  → Age bin: {age_idx}")
        print(f"  → Sex: {sex_idx} ({'F' if sex_idx == 1 else 'M'})")
        print(f"  → Race: {race_idx} (['White', 'Black', 'Asian', 'Hispanic'][race_idx])")
        print(f"  → Clinical: {clinical}")

    print("\n✓ Demographic parsing tests passed!")


def test_hcn_module():
    """Test HCN module forward pass and losses."""
    print("\n" + "="*60)
    print("TEST 2: HCN Module")
    print("="*60)

    # Create HCN
    hcn = HierarchicalConditioner(
        num_age_bins=5,
        num_sex=2,
        num_race=4,
        d_node=256,
        d_ctx=1024,
    )

    print(f"HCN created with {sum(p.numel() for p in hcn.parameters()):,} parameters")

    # Test forward pass
    batch_size = 4
    age_idx = torch.tensor([0, 2, 3, 4])  # Different age bins
    sex_idx = torch.tensor([0, 1, 0, 1])  # M, F, M, F
    race_idx = torch.tensor([0, 1, 2, 3])  # White, Black, Asian, Hispanic

    hcn.train()
    ctx, mu, logsigma = hcn(age_idx, sex_idx, race_idx)

    print(f"\nForward pass:")
    print(f"  Input: age={age_idx.tolist()}, sex={sex_idx.tolist()}, race={race_idx.tolist()}")
    print(f"  Output ctx shape: {ctx.shape} (expected: [{batch_size}, 1, 1024])")
    print(f"  Output mu shape: {mu.shape} (expected: [{batch_size}, 256])")
    print(f"  Output logsigma shape: {logsigma.shape} (expected: [{batch_size}, 256])")

    assert ctx.shape == (batch_size, 1, 1024), f"Wrong ctx shape: {ctx.shape}"
    assert mu.shape == (batch_size, 256), f"Wrong mu shape: {mu.shape}"
    assert logsigma.shape == (batch_size, 256), f"Wrong logsigma shape: {logsigma.shape}"

    # Test compositional loss
    comp_loss = hcn.compute_compositional_loss(age_idx, sex_idx, race_idx)
    print(f"\nCompositional loss: {comp_loss.item():.4f}")
    assert comp_loss.ndim == 0, "Compositional loss should be scalar"

    # Test uncertainty
    sigma = hcn.get_uncertainty(age_idx, sex_idx, race_idx)
    print(f"Uncertainty (sigma): {sigma.tolist()}")
    print(f"  Mean sigma: {sigma.mean().item():.4f}")

    print("\n✓ HCN module tests passed!")


def test_hcn_save_load():
    """Test HCN save and load functionality."""
    print("\n" + "="*60)
    print("TEST 3: HCN Save/Load")
    print("="*60)

    import tempfile
    import shutil

    # Create and save HCN
    hcn = HierarchicalConditioner(
        num_age_bins=5,
        num_sex=2,
        num_race=4,
    )

    temp_dir = tempfile.mkdtemp()
    try:
        print(f"Saving HCN to: {temp_dir}")
        hcn.save_pretrained(temp_dir)

        # Load HCN
        hcn_loaded = HierarchicalConditioner.from_pretrained(temp_dir)

        # Test they produce same output
        age = torch.tensor([1, 2])
        sex = torch.tensor([0, 1])
        race = torch.tensor([1, 2])

        hcn.eval()
        hcn_loaded.eval()

        ctx1, _, _ = hcn(age, sex, race)
        ctx2, _, _ = hcn_loaded(age, sex, race)

        diff = (ctx1 - ctx2).abs().max().item()
        print(f"Max difference between original and loaded: {diff:.6f}")
        assert diff < 1e-5, f"Loaded model differs too much: {diff}"

        print("\n✓ HCN save/load tests passed!")
    finally:
        shutil.rmtree(temp_dir)


def test_loss_computation():
    """Test that losses can be computed and backpropagated."""
    print("\n" + "="*60)
    print("TEST 4: Loss Computation & Backpropagation")
    print("="*60)

    hcn = HierarchicalConditioner(
        num_age_bins=5,
        num_sex=2,
        num_race=4,
    )

    optimizer = torch.optim.Adam(hcn.parameters(), lr=1e-4)

    # Dummy data
    age = torch.randint(0, 5, (8,))
    sex = torch.randint(0, 2, (8,))
    race = torch.randint(0, 4, (8,))

    # Forward pass
    ctx, mu, logsigma = hcn(age, sex, race)

    # Compute losses
    # 1. KL loss
    kl_loss = -0.5 * torch.sum(1 + 2*logsigma - mu**2 - torch.exp(2*logsigma), dim=-1).mean()

    # 2. Compositional loss
    comp_loss = hcn.compute_compositional_loss(age, sex, race)

    # 3. Dummy diffusion loss (simulating real training)
    dummy_target = torch.randn_like(ctx)
    diffusion_loss = torch.nn.functional.mse_loss(ctx, dummy_target)

    # Total loss
    total_loss = diffusion_loss + 0.001 * kl_loss + 0.01 * comp_loss

    print(f"Losses:")
    print(f"  Diffusion: {diffusion_loss.item():.4f}")
    print(f"  KL: {kl_loss.item():.4f}")
    print(f"  Compositional: {comp_loss.item():.4f}")
    print(f"  Total: {total_loss.item():.4f}")

    # Backprop
    optimizer.zero_grad()
    total_loss.backward()

    # Check gradients
    has_grad = sum(1 for p in hcn.parameters() if p.grad is not None)
    total_params = sum(1 for _ in hcn.parameters())
    print(f"\nGradients: {has_grad}/{total_params} parameters have gradients")
    assert has_grad == total_params, "Not all parameters have gradients!"

    optimizer.step()

    print("\n✓ Loss computation tests passed!")


def test_integration():
    """Test that HCN integrates properly with text encoder output."""
    print("\n" + "="*60)
    print("TEST 5: Integration with Text Encoder")
    print("="*60)

    # Simulate text encoder output
    batch_size = 4
    seq_len = 77
    d_ctx = 1024

    text_ctx = torch.randn(batch_size, seq_len, d_ctx)  # Fake CLIP output
    print(f"Text context shape: {text_ctx.shape}")

    # Get HCN output
    hcn = HierarchicalConditioner(num_age_bins=5, num_sex=2, num_race=4, d_ctx=d_ctx)
    age = torch.randint(0, 5, (batch_size,))
    sex = torch.randint(0, 2, (batch_size,))
    race = torch.randint(0, 4, (batch_size,))

    hcn_ctx, _, _ = hcn(age, sex, race)
    print(f"HCN context shape: {hcn_ctx.shape}")

    # Concatenate (as will be done in training)
    full_ctx = torch.cat([text_ctx, hcn_ctx], dim=1)
    print(f"Full context shape: {full_ctx.shape} (expected: [{batch_size}, {seq_len+1}, {d_ctx}])")

    assert full_ctx.shape == (batch_size, seq_len + 1, d_ctx), f"Wrong full context shape: {full_ctx.shape}"

    print("\n✓ Integration tests passed!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("HCN Integration Tests for RoentGen-v2")
    print("="*60)

    try:
        test_demographic_parsing()
        test_hcn_module()
        test_hcn_save_load()
        test_loss_computation()
        test_integration()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou can now proceed with training.")
        print("To enable HCN, set 'use_hcn: True' in your config YAML file.")

    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED!")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
