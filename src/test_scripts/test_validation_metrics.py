"""
Test script for validation metrics implementation.

This script performs basic sanity checks to ensure all metrics can be computed.
It uses synthetic random data to verify the pipeline works end-to-end.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from roentgenv2.train_code.validation_metrics import (
    TextPromptAlignmentMetrics,
    RealSyntheticSimilarityMetrics,
    IntraPromptDiversityMetrics,
    ValidationMetricsRunner
)


def test_text_alignment_metrics():
    """Test text prompt alignment metrics."""
    print("\n" + "="*60)
    print("Testing Text Prompt Alignment Metrics")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = TextPromptAlignmentMetrics(device=device)

    # Create dummy data on the correct device
    batch_size = 10
    synthetic_images = torch.randn(batch_size, 1, 512, 512).to(device)

    # Disease labels (5 diseases)
    disease_labels = torch.randint(0, 2, (batch_size, 5), dtype=torch.float32).to(device)

    # Sex labels (0=M, 1=F)
    sex_labels = torch.randint(0, 2, (batch_size,), dtype=torch.long).to(device)

    # Race labels (0-3)
    race_labels = torch.randint(0, 4, (batch_size,), dtype=torch.long).to(device)

    # Age labels (18-90 years)
    age_labels = (torch.rand(batch_size) * 72 + 18).to(device)

    print("\n1. Testing disease AUROC computation...")
    try:
        aurocs = metrics.compute_disease_auroc(synthetic_images, disease_labels)
        print("   ✅ Disease AUROC computed successfully")
        print(f"   Mean AUROC: {aurocs.get('mean_auroc', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n2. Testing race accuracy computation...")
    try:
        race_acc = metrics.compute_race_accuracy(synthetic_images, race_labels)
        print("   ✅ Race accuracy computed successfully")
        print(f"   Race Accuracy: {race_acc:.4f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n3. Testing age RMSE computation...")
    try:
        age_rmse = metrics.compute_age_rmse(synthetic_images, age_labels)
        print("   ✅ Age RMSE computed successfully")
        print(f"   Age RMSE: {age_rmse:.2f} years")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n4. Testing sex accuracy computation (requires sex model)...")
    try:
        # This will warn if sex model not loaded, which is expected
        sex_acc = metrics.compute_sex_accuracy(synthetic_images, sex_labels)
        if np.isnan(sex_acc):
            print("   ⚠️  Sex model not loaded (expected)")
        else:
            print("   ✅ Sex accuracy computed successfully")
            print(f"   Sex Accuracy: {sex_acc:.4f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")


def test_similarity_metrics():
    """Test real-synthetic similarity metrics."""
    print("\n" + "="*60)
    print("Testing Real-Synthetic Similarity Metrics")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = RealSyntheticSimilarityMetrics(device=device)

    # Create dummy data on the correct device
    batch_size = 10
    real_images = torch.randn(batch_size, 1, 512, 512).to(device)
    synthetic_images = torch.randn(batch_size, 1, 512, 512).to(device)

    print("\n1. Testing FID computation...")
    try:
        fid = metrics.compute_fid(real_images, synthetic_images, batch_size=5)
        print("   ✅ FID computed successfully")
        print(f"   FID Score: {fid:.2f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n2. Testing MS-SSIM computation...")
    try:
        ms_ssim = metrics.compute_ms_ssim(real_images, synthetic_images)
        print("   ✅ MS-SSIM computed successfully")
        print(f"   MS-SSIM: {ms_ssim:.4f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n3. Testing BioViL similarity computation...")
    try:
        biovil_sim = metrics.compute_biovil_similarity(real_images, synthetic_images)
        if np.isnan(biovil_sim):
            print("   ⚠️  BioViL model not available (optional)")
        else:
            print("   ✅ BioViL similarity computed successfully")
            print(f"   BioViL Similarity: {biovil_sim:.4f}")
    except Exception as e:
        print(f"   ⚠️  BioViL error (optional): {e}")


def test_diversity_metrics():
    """Test intra-prompt diversity metrics."""
    print("\n" + "="*60)
    print("Testing Intra-Prompt Diversity Metrics")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = IntraPromptDiversityMetrics(device=device)

    # Create dummy data: 5 prompts, 4 images each
    num_prompts = 5
    images_per_prompt = 4
    images_list = [
        torch.randn(images_per_prompt, 1, 512, 512).to(device)
        for _ in range(num_prompts)
    ]

    print("\n1. Testing intra-prompt MS-SSIM...")
    try:
        intra_ms_ssim = metrics.compute_intra_prompt_ms_ssim(images_list)
        print("   ✅ Intra-prompt MS-SSIM computed successfully")
        print(f"   Intra-Prompt MS-SSIM: {intra_ms_ssim:.4f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n2. Testing intra-prompt BioViL similarity...")
    try:
        intra_biovil = metrics.compute_intra_prompt_biovil_similarity(images_list)
        if np.isnan(intra_biovil):
            print("   ⚠️  BioViL model not available (optional)")
        else:
            print("   ✅ Intra-prompt BioViL similarity computed successfully")
            print(f"   Intra-Prompt BioViL Similarity: {intra_biovil:.4f}")
    except Exception as e:
        print(f"   ⚠️  BioViL error (optional): {e}")


def test_validation_runner(sex_model_path: str):
    """Test the complete validation metrics runner."""
    print("\n" + "="*60)
    print("Testing Complete Validation Metrics Runner")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = ValidationMetricsRunner(device=device, sex_model_path=sex_model_path)

    # Create dummy data on the correct device
    batch_size = 10
    real_images = torch.randn(batch_size, 1, 512, 512).to(device)
    synthetic_images = torch.randn(batch_size, 1, 512, 512).to(device)

    disease_labels = torch.randint(0, 2, (batch_size, 5), dtype=torch.float32).to(device)
    sex_labels = torch.randint(0, 2, (batch_size,), dtype=torch.long).to(device)
    race_labels = torch.randint(0, 4, (batch_size,), dtype=torch.long).to(device)
    age_labels = (torch.rand(batch_size) * 72 + 18).to(device)

    # Create images per prompt for diversity metrics
    images_per_prompt = [
        torch.randn(4, 1, 512, 512).to(device)
        for _ in range(batch_size)
    ]

    print("\nComputing all metrics...")
    try:
        metrics = runner.compute_all_metrics(
            real_images=real_images,
            synthetic_images=synthetic_images,
            disease_labels=disease_labels,
            sex_labels=sex_labels,
            race_labels=race_labels,
            age_labels=age_labels,
            images_per_prompt=images_per_prompt
        )

        print("   ✅ All metrics computed successfully\n")
        runner.print_metrics_summary(metrics)

        return metrics

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("VALIDATION METRICS TEST SUITE")
    print("="*60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Run individual component tests
    test_text_alignment_metrics()
    test_similarity_metrics()
    test_diversity_metrics()

    # Run integrated test
    metrics = test_validation_runner(sex_model_path="/home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/pretrained_models/sex/resnet-all/epoch=13-step=7125.ckpt")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    if metrics is not None:
        print("✅ All core metrics are functional!")
        print("\nNote:")
        print("- Some metrics may show warnings (BioViL, sex model)")
        print("- These are optional and will return NaN if unavailable")
        print("- The validation pipeline can still run without them")
    else:
        print("❌ Some tests failed. Check error messages above.")

    print("\nNext steps:")
    print("1. Install missing dependencies if needed:")
    print("   pip install -r requirements.txt")
    print("\n2. Download optional models:")
    print("   - Sex model checkpoint (for sex accuracy)")
    print("   - BioViL model (for medical image embeddings)")
    print("\n3. Run validation on actual checkpoint:")
    print("   python roentgenv2/train_code/run_validation.py \\")
    print("       --checkpoint_path <checkpoint> \\")
    print("       --validation_csv configs/validation_example.csv \\")
    print("       --output_dir validation_results")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
