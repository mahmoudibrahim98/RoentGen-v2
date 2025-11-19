#!/usr/bin/env python
"""
Test script to diagnose RadImageNet loading issues.
"""

import torch
import traceback
import sys

def test_radimagenet_load():
    """Test loading RadImageNet ResNet50 from torch.hub."""
    print("=" * 60)
    print("Testing RadImageNet ResNet50 Loading")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"Using device: {device}\n")
    
    # Try loading RadImageNet
    print("Attempting to load RadImageNet ResNet50 from torch.hub...")
    print("-" * 60)
    
    try:
        model = torch.hub.load(
            "Warvito/radimagenet-models", 
            model="radimagenet_resnet50", 
            trust_repo=True, 
            verbose=True
        )
        print("✓ Model loaded successfully from torch.hub")
        
        # Check model structure
        print("\nModel structure:")
        print("-" * 60)
        print(f"Model type: {type(model)}")
        print(f"Model attributes: {dir(model)[:20]}...")  # First 20 attributes
        
        # Check for FC layer
        if hasattr(model, 'fc'):
            print(f"✓ Has 'fc' attribute: {model.fc}")
        elif hasattr(model, 'classifier'):
            print(f"✓ Has 'classifier' attribute: {model.classifier}")
        else:
            print("⚠️ No 'fc' or 'classifier' attribute found")
            print(f"Available attributes with 'fc' or 'class' in name:")
            attrs = [attr for attr in dir(model) if 'fc' in attr.lower() or 'class' in attr.lower()]
            for attr in attrs:
                print(f"  - {attr}: {getattr(model, attr)}")
        
        # Try to remove FC layer
        print("\nAttempting to remove FC layer...")
        print("-" * 60)
        if hasattr(model, 'fc'):
            model.fc = torch.nn.Identity()
            print("✓ Removed 'fc' layer")
        elif hasattr(model, 'classifier'):
            model.classifier = torch.nn.Identity()
            print("✓ Removed 'classifier' layer")
        else:
            print("⚠️ Could not find FC layer to remove")
        
        # Try moving to device
        print(f"\nAttempting to move model to {device}...")
        print("-" * 60)
        model = model.to(device)
        print(f"✓ Model moved to {device}")
        
        # Try setting to eval mode
        print("\nSetting model to eval mode...")
        print("-" * 60)
        model.eval()
        print("✓ Model set to eval mode")
        
        # Test forward pass with dummy input
        print("\nTesting forward pass with dummy input...")
        print("-" * 60)
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output dtype: {output.dtype}")
        
        print("\n" + "=" * 60)
        print("✓ SUCCESS: RadImageNet loaded and tested successfully!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ FAILED: Error loading RadImageNet")
        print("=" * 60)
        print(f"\nError type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"\nFull traceback:")
        print("-" * 60)
        traceback.print_exc()
        print("-" * 60)
        
        # Try fallback
        print("\n" + "=" * 60)
        print("Attempting fallback to ImageNet-pretrained ResNet50...")
        print("=" * 60)
        try:
            from torchvision.models import resnet50
            fallback_model = resnet50(pretrained=True)
            fallback_model.fc = torch.nn.Identity()
            fallback_model = fallback_model.to(device)
            fallback_model.eval()
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                output = fallback_model(dummy_input)
            
            print("✓ Fallback to ImageNet ResNet50 successful")
            print(f"  Output shape: {output.shape}")
            return False
        except Exception as fallback_error:
            print(f"✗ Fallback also failed: {fallback_error}")
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = test_radimagenet_load()
    sys.exit(0 if success else 1)











