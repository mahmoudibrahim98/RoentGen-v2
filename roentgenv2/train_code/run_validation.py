"""
Validation script for RoentGen-v2 model checkpoints.

This script performs inference on the MIMIC-CXR validation set and computes
comprehensive quality metrics to guide checkpoint selection.

Usage:
    python run_validation.py --checkpoint_path <path> --validation_csv <path>
"""

import argparse
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import json
from datetime import datetime

from diffusers import StableDiffusionPipeline, DDPMScheduler
from validation_metrics import ValidationMetricsRunner
from dataset_wds import parse_age_bin, parse_sex, parse_race, extract_clinical_text


def load_validation_data(csv_path: str, real_images_dir: str = None) -> pd.DataFrame:
    """
    Load validation dataset from CSV.

    Args:
        csv_path: Path to validation CSV with columns: prompt, disease_labels, sex, race, age
        real_images_dir: Optional directory containing real images

    Returns:
        DataFrame with validation data
    """
    df = pd.read_csv(csv_path)

    # Verify required columns
    required_cols = ["prompt", "sex", "race", "age"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Parse demographic attributes from prompts if not already present
    if "age_bin" not in df.columns:
        df["age_bin"] = df["prompt"].apply(parse_age_bin)
    if "sex_idx" not in df.columns:
        df["sex_idx"] = df["prompt"].apply(parse_sex)
    if "race_idx" not in df.columns:
        df["race_idx"] = df["prompt"].apply(parse_race)

    # Add real image paths if directory provided
    if real_images_dir:
        if "image_id" in df.columns:
            df["real_image_path"] = df["image_id"].apply(
                lambda x: os.path.join(real_images_dir, f"{x}.jpg")
            )

    return df


def generate_synthetic_images(
    pipeline: StableDiffusionPipeline,
    prompts: List[str],
    num_images_per_prompt: int = 4,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    device: str = "cuda"
) -> List[List[torch.Tensor]]:
    """
    Generate synthetic images for validation prompts.

    Args:
        pipeline: Stable Diffusion pipeline with RoentGen-v2 checkpoint
        prompts: List of text prompts
        num_images_per_prompt: Number of images to generate per prompt (default: 4)
        guidance_scale: Classifier-free guidance scale
        num_inference_steps: Number of denoising steps
        device: Device to run on

    Returns:
        List of lists, where each inner list contains tensors for images from same prompt
    """
    all_images = []

    for prompt in tqdm(prompts, desc="Generating images"):
        images_for_prompt = []

        for seed in range(num_images_per_prompt):
            # Set random seed for reproducibility
            generator = torch.Generator(device=device).manual_seed(seed)

            # Generate image
            output = pipeline(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="pt"  # Return as PyTorch tensor
            )

            image = output.images[0]  # [C, H, W]
            images_for_prompt.append(image)

        all_images.append(images_for_prompt)

    return all_images


def load_real_images(image_paths: List[str]) -> torch.Tensor:
    """
    Load real images from paths.

    Args:
        image_paths: List of paths to real images

    Returns:
        Tensor of real images [N, C, H, W]
    """
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    images = []
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Real image not found: {path}")

        img = Image.open(path).convert("L")  # Convert to grayscale
        img_tensor = transform(img)
        images.append(img_tensor)

    return torch.stack(images)


def prepare_disease_labels(df: pd.DataFrame, disease_names: List[str]) -> torch.Tensor:
    """
    Extract disease labels from DataFrame.

    Args:
        df: DataFrame with disease label columns
        disease_names: List of disease names to extract

    Returns:
        Tensor of disease labels [N, num_diseases]
    """
    labels = []
    for disease in disease_names:
        if disease.lower() in df.columns:
            labels.append(df[disease.lower()].values)
        else:
            # If not present, assume all negative
            labels.append(np.zeros(len(df)))

    return torch.tensor(np.stack(labels, axis=1), dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Validate RoentGen-v2 checkpoint")

    # Model and data paths
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--validation_csv",
        type=str,
        required=True,
        help="Path to validation CSV file"
    )
    parser.add_argument(
        "--real_images_dir",
        type=str,
        default=None,
        help="Directory containing real validation images"
    )
    parser.add_argument(
        "--sex_model_path",
        type=str,
        default=None,
        help="Path to sex prediction model checkpoint"
    )

    # Generation parameters
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=4,
        help="Number of synthetic images to generate per prompt"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="validation_results",
        help="Directory to save validation results"
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save generated images to output directory"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load validation data
    print(f"Loading validation data from {args.validation_csv}")
    val_df = load_validation_data(args.validation_csv, args.real_images_dir)
    print(f"Loaded {len(val_df)} validation samples")

    # Load model checkpoint
    print(f"Loading model from {args.checkpoint_path}")
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.checkpoint_path,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
    )
    pipeline = pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=True)

    # Generate synthetic images
    print(f"Generating {args.num_images_per_prompt} synthetic images per prompt...")
    synthetic_images_per_prompt = generate_synthetic_images(
        pipeline=pipeline,
        prompts=val_df["prompt"].tolist(),
        num_images_per_prompt=args.num_images_per_prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        device=args.device
    )

    # Take first generated image per prompt for main metrics
    synthetic_images = torch.stack([images[0] for images in synthetic_images_per_prompt])

    # Optionally save generated images
    if args.save_images:
        print("Saving generated images...")
        images_dir = os.path.join(args.output_dir, "synthetic_images")
        os.makedirs(images_dir, exist_ok=True)

        from torchvision.utils import save_image
        for i, images in enumerate(synthetic_images_per_prompt):
            for j, img in enumerate(images):
                save_path = os.path.join(images_dir, f"sample_{i:04d}_seed_{j}.png")
                save_image(img, save_path)

    # Load real images if available
    real_images = None
    if args.real_images_dir and "real_image_path" in val_df.columns:
        print("Loading real images...")
        real_images = load_real_images(val_df["real_image_path"].tolist())
        real_images = real_images.to(args.device)

    # Prepare labels
    disease_names = ["Atelectasis", "Cardiomegaly", "Edema", "Pneumothorax", "Pleural Effusion"]
    disease_labels = prepare_disease_labels(val_df, disease_names)

    sex_labels = torch.tensor(val_df["sex_idx"].values, dtype=torch.long)
    race_labels = torch.tensor(val_df["race_idx"].values, dtype=torch.long)
    age_labels = torch.tensor(val_df["age"].values, dtype=torch.float32)

    # Initialize metrics runner
    print("Initializing validation metrics...")
    metrics_runner = ValidationMetricsRunner(
        device=args.device,
        sex_model_path=args.sex_model_path
    )

    # Compute metrics
    print("\nComputing validation metrics...")
    print("="*60)

    metrics = {}

    # 1. Text Prompt Alignment (always computed)
    print("\n1. Computing text prompt alignment metrics...")
    synthetic_images_gpu = synthetic_images.to(args.device)

    disease_aurocs = metrics_runner.text_alignment.compute_disease_auroc(
        synthetic_images_gpu, disease_labels.to(args.device)
    )
    metrics.update(disease_aurocs)

    if args.sex_model_path:
        metrics["sex_accuracy"] = metrics_runner.text_alignment.compute_sex_accuracy(
            synthetic_images_gpu, sex_labels.to(args.device)
        )

    metrics["race_accuracy"] = metrics_runner.text_alignment.compute_race_accuracy(
        synthetic_images_gpu, race_labels.to(args.device)
    )

    metrics["age_rmse"] = metrics_runner.text_alignment.compute_age_rmse(
        synthetic_images_gpu, age_labels.to(args.device)
    )

    # 2. Real-Synthetic Similarity (if real images available)
    if real_images is not None:
        print("\n2. Computing real-synthetic similarity metrics...")

        metrics["fid"] = metrics_runner.similarity.compute_fid(
            real_images, synthetic_images_gpu
        )

        metrics["biovil_similarity"] = metrics_runner.similarity.compute_biovil_similarity(
            real_images, synthetic_images_gpu
        )

        metrics["ms_ssim"] = metrics_runner.similarity.compute_ms_ssim(
            real_images, synthetic_images_gpu
        )
    else:
        print("\n2. Skipping real-synthetic similarity (no real images provided)")

    # 3. Intra-Prompt Diversity
    print("\n3. Computing intra-prompt diversity metrics...")

    # Convert to list of tensors on GPU
    images_per_prompt_gpu = [
        torch.stack(images).to(args.device)
        for images in synthetic_images_per_prompt
    ]

    metrics["intra_prompt_ms_ssim"] = metrics_runner.diversity.compute_intra_prompt_ms_ssim(
        images_per_prompt_gpu
    )

    metrics["intra_prompt_biovil_similarity"] = metrics_runner.diversity.compute_intra_prompt_biovil_similarity(
        images_per_prompt_gpu
    )

    # Print summary
    metrics_runner.print_metrics_summary(metrics)

    # Save metrics to JSON
    metrics_file = os.path.join(args.output_dir, "validation_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {metrics_file}")

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"validation_results_{timestamp}.json")

    results = {
        "checkpoint_path": args.checkpoint_path,
        "validation_csv": args.validation_csv,
        "num_samples": len(val_df),
        "num_images_per_prompt": args.num_images_per_prompt,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "metrics": metrics,
        "timestamp": timestamp
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Detailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
