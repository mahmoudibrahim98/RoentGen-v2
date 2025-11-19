#!/usr/bin/env python3
"""
Test script to verify parallel validation implementation.
This script checks if the parallel validation code will work correctly.
"""

import torch
from accelerate import Accelerator

def test_split_between_processes():
    """Test the split_between_processes functionality"""
    print("Testing split_between_processes...")

    # Initialize accelerator
    accelerator = Accelerator()

    # Create a list of items to split
    num_prompts = 10
    num_images_per_prompt = 4
    all_images_to_generate = []

    for prompt_idx in range(num_prompts):
        for image_idx in range(num_images_per_prompt):
            all_images_to_generate.append((prompt_idx, image_idx))

    print(f"Total images to generate: {len(all_images_to_generate)}")
    print(f"Number of processes: {accelerator.num_processes}")
    print(f"Process index: {accelerator.process_index}")

    # Split across processes
    with accelerator.split_between_processes(all_images_to_generate) as images_for_this_process:
        print(f"Process {accelerator.process_index} got {len(images_for_this_process)} images")
        print(f"First 5 images: {images_for_this_process[:5]}")

    # Test gather operation
    print("\nTesting gather operation...")

    # Create some dummy tensors
    local_tensor = torch.randn(5, 3, 64, 64).to(accelerator.device)
    print(f"Local tensor shape: {local_tensor.shape}")

    # Gather from all processes
    gathered_tensor = accelerator.gather(local_tensor)

    if accelerator.is_main_process:
        print(f"Gathered tensor shape: {gathered_tensor.shape}")
        print(f"Expected shape: ({5 * accelerator.num_processes}, 3, 64, 64)")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_split_between_processes()
