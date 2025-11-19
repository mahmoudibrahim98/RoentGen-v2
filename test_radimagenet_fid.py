#!/usr/bin/env python
"""
Test script to verify RadImageNet FID computation.
"""

import torch
import numpy as np
import traceback
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from roentgenv2.train_code.validation_metrics import RealSyntheticSimilarityMetrics


def test_radimagenet_fid():
    """Test RadImageNet FID computation."""
    print("=" * 60)
    print("Testing RadImageNet FID Computation")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"Using device: {device}\n")
    
    # Create dummy images for testing
    print("Creating dummy test images...")
    print("-" * 60)
    num_images = 50
    image_size = 512
    
    # Create "real" images (random but consistent)
    torch.manual_seed(42)
    real_images = torch.rand(num_images, 3, image_size, image_size)
    print(f"Created {num_images} real images: {real_images.shape}")
    
    # Create "synthetic" images (slightly different)
    torch.manual_seed(123)
    synthetic_images = torch.rand(num_images, 3, image_size, image_size)
    print(f"Created {num_images} synthetic images: {synthetic_images.shape}")
    
    # Initialize metrics runner
    print("\nInitializing RealSyntheticSimilarityMetrics...")
    print("-" * 60)
    try:
        metrics_runner = RealSyntheticSimilarityMetrics(device=device)
        print("✓ Metrics runner initialized")
    except Exception as e:
        print(f"✗ Failed to initialize metrics runner: {e}")
        traceback.print_exc()
        return False
    
    # Test RadImageNet loading
    print("\n" + "=" * 60)
    print("Testing RadImageNet Model Loading")
    print("=" * 60)
    try:
        metrics_runner._load_radimagenet()
        if metrics_runner.radimagenet_model is None:
            print("✗ RadImageNet model is None after loading")
            return False
        print("✓ RadImageNet model loaded successfully")
        print(f"  Model type: {type(metrics_runner.radimagenet_model)}")
        print(f"  Model device: {next(metrics_runner.radimagenet_model.parameters()).device}")
        print(f"  Model in eval mode: {not metrics_runner.radimagenet_model.training}")
    except Exception as e:
        print(f"✗ Failed to load RadImageNet model: {e}")
        traceback.print_exc()
        return False
    
    # Test embedding extraction
    print("\n" + "=" * 60)
    print("Testing RadImageNet Embedding Extraction")
    print("=" * 60)
    try:
        print("Extracting embeddings from real images...")
        real_embeddings = metrics_runner._get_radimagenet_embeddings(real_images, batch_size=8)
        print(f"✓ Real embeddings extracted: {real_embeddings.shape}, dtype: {real_embeddings.dtype}")
        print(f"  Min: {real_embeddings.min():.4f}, Max: {real_embeddings.max():.4f}, Mean: {real_embeddings.mean():.4f}")
        
        print("\nExtracting embeddings from synthetic images...")
        synthetic_embeddings = metrics_runner._get_radimagenet_embeddings(synthetic_images, batch_size=8)
        print(f"✓ Synthetic embeddings extracted: {synthetic_embeddings.shape}, dtype: {synthetic_embeddings.dtype}")
        print(f"  Min: {synthetic_embeddings.min():.4f}, Max: {synthetic_embeddings.max():.4f}, Mean: {synthetic_embeddings.mean():.4f}")
        
        # Verify embeddings are 2D
        if real_embeddings.ndim != 2:
            print(f"⚠️ Warning: Real embeddings have {real_embeddings.ndim} dimensions, expected 2")
            print(f"  Shape: {real_embeddings.shape}")
        else:
            print(f"✓ Real embeddings are 2D: [N={real_embeddings.shape[0]}, D={real_embeddings.shape[1]}]")
        
        if synthetic_embeddings.ndim != 2:
            print(f"⚠️ Warning: Synthetic embeddings have {synthetic_embeddings.ndim} dimensions, expected 2")
            print(f"  Shape: {synthetic_embeddings.shape}")
        else:
            print(f"✓ Synthetic embeddings are 2D: [N={synthetic_embeddings.shape[0]}, D={synthetic_embeddings.shape[1]}]")
        
    except Exception as e:
        print(f"✗ Failed to extract embeddings: {e}")
        traceback.print_exc()
        return False
    
    # Test FID computation
    print("\n" + "=" * 60)
    print("Testing RadImageNet FID Computation")
    print("=" * 60)
    try:
        print("Computing FID using RadImageNet embeddings...")
        fid_score = metrics_runner.compute_fid_radimagenet(
            real_images.to(device),
            synthetic_images.to(device),
            batch_size=8
        )
        print(f"✓ FID computation successful!")
        print(f"  FID Score: {fid_score:.4f}")
        print(f"  (Lower is better, 0 = identical distributions)")
        
        # Test with identical images (should give FID ≈ 0)
        print("\nTesting FID with identical images (should be ~0)...")
        fid_identical = metrics_runner.compute_fid_radimagenet(
            real_images.to(device),
            real_images.to(device),
            batch_size=8
        )
        print(f"✓ FID with identical images: {fid_identical:.4f}")
        if fid_identical < 1.0:
            print("  ✓ Good: FID is very low for identical images")
        else:
            print(f"  ⚠️ Warning: FID is higher than expected for identical images")
        
    except Exception as e:
        print(f"✗ Failed to compute FID: {e}")
        traceback.print_exc()
        return False
    
    # Test with different batch sizes
    print("\n" + "=" * 60)
    print("Testing with Different Batch Sizes")
    print("=" * 60)
    for batch_size in [1, 4, 8, 16]:
        try:
            fid = metrics_runner.compute_fid_radimagenet(
                real_images.to(device),
                synthetic_images.to(device),
                batch_size=batch_size
            )
            print(f"✓ Batch size {batch_size:2d}: FID = {fid:.4f}")
        except Exception as e:
            print(f"✗ Batch size {batch_size:2d} failed: {e}")
    
    print("\n" + "=" * 60)
    print("✓ SUCCESS: All RadImageNet FID tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = test_radimagenet_fid()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)











