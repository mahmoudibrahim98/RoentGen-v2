#!/usr/bin/env python
"""
Calculate validation metrics on REAL data only.

This script computes all validation metrics using only real validation data.
For metrics that require real vs synthetic comparison (FID, BioViL, MS-SSIM),
the real data is split into two halves, with one half treated as "synthetic".

This is useful for:
- Establishing baseline metrics (best possible scores)
- Understanding metric ranges
- Validating the validation dataset quality

Usage:
    python roentgenv2/train_code/calculate_metrics_on_real_data.py \
        --validation_csv /path/to/validation/data \
        --output_dir real_data_metrics \
        --num_samples 200
"""

import os
import sys
import json
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import argparse
import yaml
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from validation_metrics import ValidationMetricsRunner
from dataset_wds import RGFineTuningImageDirectoryDataset, RGFineTuningWebDataset
from transformers import CLIPTokenizer

logger = logging.getLogger(__name__)


def load_config(file_path):
    """Load YAML config file."""
    with open(file_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as e:
            print("Error loading YAML:", e)
            return None


def load_real_validation_data(
    validation_csv: str,
    validation_images_dir: Optional[str] = None,
    use_wds_dataset: bool = False,
    num_samples: Optional[int] = None,
    tokenizer = None,
    use_hcn: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[str]]:
    """
    Load real validation data.
    
    Returns:
        images: Tensor of images [N, C, H, W]
        labels: Dictionary with keys: disease, sex, race, age
        prompts: List of prompt strings
    """
    logger.info("Loading real validation data...")
    
    if use_wds_dataset:
        # WebDataset format
        import glob
        if os.path.isdir(validation_csv):
            tar_files = sorted(glob.glob(os.path.join(validation_csv, "*.tar")))
            if not tar_files:
                raise ValueError(f"No tar files found in {validation_csv}")
            val_urls = tar_files
        else:
            val_urls = [validation_csv] if isinstance(validation_csv, str) else validation_csv
        
        dataset = RGFineTuningWebDataset(
            url_list=val_urls,
            tokenizer=tokenizer,
            use_hcn=use_hcn,
            include_text=True,
        )
    else:
        # Directory dataset
        if validation_images_dir is None:
            validation_images_dir = validation_csv
        
        dataset = RGFineTuningImageDirectoryDataset(
            image_dir_path=validation_images_dir,
            text_dir_path=validation_csv,
            tokenizer=tokenizer,
            use_hcn=use_hcn,
            include_text=True,
        )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )
    
    # Collect all data
    all_images = []
    all_labels = {
        "disease": [],
        "sex": [],
        "race": [],
        "age": [],
    }
    all_prompts = []
    
    logger.info("Collecting validation samples...")
    num_collected = 0
    
    for batch in tqdm(dataloader, desc="Loading data"):
        if num_samples is not None and num_collected >= num_samples:
            break
        
        # Extract image
        image = batch["pixel_values"][0]  # [C, H, W]
        all_images.append(image)
        
        # Extract labels
        if "disease_labels" in batch:
            all_labels["disease"].append(batch["disease_labels"][0])
        elif "disease" in batch:
            # Try alternative key name
            all_labels["disease"].append(batch["disease"][0])
        if "sex_idx" in batch:
            all_labels["sex"].append(batch["sex_idx"][0])
        if "race_idx" in batch:
            all_labels["race"].append(batch["race_idx"][0])
        if "age" in batch:
            all_labels["age"].append(batch["age"][0])
        
        # Extract prompt
        prompt = batch.get("text", [""])[0] if isinstance(batch.get("text"), list) else batch.get("text", "")
        all_prompts.append(prompt)
        
        num_collected += 1
    
    # Stack images (keep on CPU to save GPU memory)
    images = torch.stack(all_images)  # [N, C, H, W] on CPU
    
    # Stack labels (keep on CPU)
    stacked_labels = {}
    for key in all_labels:
        if all_labels[key]:
            stacked_labels[key] = torch.stack(all_labels[key])
        else:
            stacked_labels[key] = None
    
    logger.info(f"Loaded {len(all_images)} real validation samples")
    logger.info(f"Images shape: {images.shape}, dtype: {images.dtype}, device: {images.device}")
    
    # Diagnostic: Check image value range
    logger.info(f"Image value range: min={images.min():.4f}, max={images.max():.4f}, mean={images.mean():.4f}, std={images.std():.4f}")
    
    # Log which labels are available
    logger.info("Available labels:")
    for key in stacked_labels:
        if stacked_labels[key] is not None:
            logger.info(f"  {key}: shape {stacked_labels[key].shape}, dtype {stacked_labels[key].dtype}")
        else:
            logger.info(f"  {key}: None (not available)")
    
    return images, stacked_labels, all_prompts


def split_data_in_half(
    images: torch.Tensor,
    labels: Dict[str, torch.Tensor],
    prompts: List[str],
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[str], List[str]]:
    """
    Split data into two halves for similarity metrics.
    
    Returns:
        real_half_images, synth_half_images
        real_half_labels, synth_half_labels
        real_half_prompts, synth_half_prompts
    """
    n = len(images)
    n_half = n // 2
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    indices = np.random.permutation(n)
    
    # Split indices
    real_indices = indices[:n_half]
    synth_indices = indices[n_half:]
    
    # Split images
    real_half_images = images[real_indices]
    synth_half_images = images[synth_indices]
    
    # Split labels
    real_half_labels = {}
    synth_half_labels = {}
    for key in labels:
        if labels[key] is not None:
            real_half_labels[key] = labels[key][real_indices]
            synth_half_labels[key] = labels[key][synth_indices]
        else:
            real_half_labels[key] = None
            synth_half_labels[key] = None
    
    # Split prompts
    real_half_prompts = [prompts[i] for i in real_indices]
    synth_half_prompts = [prompts[i] for i in synth_indices]
    
    logger.info(f"Split {n} samples into two halves: {n_half} (real) + {n - n_half} (synthetic)")
    
    return (
        real_half_images, synth_half_images,
        real_half_labels, synth_half_labels,
        real_half_prompts, synth_half_prompts
    )


def compute_text_alignment_metrics(
    metrics_runner: ValidationMetricsRunner,
    images: torch.Tensor,
    labels: Dict[str, torch.Tensor],
    device: str = "cuda",
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Compute text alignment metrics on real data using methods from validation_metrics.py.
    
    This function now delegates to the enhanced methods in validation_metrics.py which handle:
    - Batching internally to avoid OOM
    - Disease AUROC mapping and filtering
    - Sex accuracy with softmax and filtering
    - Race accuracy with model-to-dataset mapping and filtering
    - Age RMSE computation
    """
    logger.info(f"Computing text alignment metrics (batch_size={batch_size})...")
    logger.info("  Using enhanced methods from validation_metrics.py")
    
    metrics = {}
    
    # Move images to device (methods will handle batching internally)
    images_gpu = images.to(device)
    
    # Disease AUROC - use enhanced method from validation_metrics.py
    if labels.get("disease") is not None:
        logger.info("  Computing disease AUROC...")
        logger.info(f"    Disease labels shape: {labels['disease'].shape}")
        try:
            disease_labels_gpu = labels["disease"].to(device)
            disease_aurocs = metrics_runner.text_alignment.compute_disease_auroc(
                images_gpu, disease_labels_gpu, batch_size=batch_size
            )
            
            # Update metrics with individual disease AUROCs
            for disease_name, auroc in disease_aurocs.items():
                if disease_name == "mean_auroc":
                    metrics["mean_auroc"] = float(auroc) if not np.isnan(auroc) else np.nan
                else:
                    metrics[disease_name] = float(auroc) if not np.isnan(auroc) else np.nan
            
            logger.info(f"    Mean AUROC: {disease_aurocs.get('mean_auroc', np.nan):.4f}")
            for disease_name, auroc in disease_aurocs.items():
                if disease_name != "mean_auroc" and not np.isnan(auroc):
                    logger.info(f"    {disease_name}: {auroc:.4f}")
        except Exception as e:
            logger.warning(f"  Disease AUROC computation failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    else:
        logger.warning("  Disease labels not available - skipping disease AUROC computation")
    
    # Sex Accuracy - use enhanced method from validation_metrics.py
    if labels.get("sex") is not None and metrics_runner.text_alignment.sex_model is not None:
        logger.info("  Computing sex accuracy...")
        try:
            sex_labels_gpu = labels["sex"].to(device)
            sex_acc = metrics_runner.text_alignment.compute_sex_accuracy(
                images_gpu, sex_labels_gpu, batch_size=batch_size
            )
            if not np.isnan(sex_acc):
                metrics["sex_accuracy"] = float(sex_acc)
                logger.info(f"    Sex accuracy: {sex_acc:.4f}")
            else:
                logger.warning("    Sex accuracy computation returned NaN")
        except Exception as e:
            logger.warning(f"  Sex accuracy computation failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    else:
        if labels.get("sex") is None:
            logger.warning("  Sex labels not available - skipping sex accuracy computation")
        elif metrics_runner.text_alignment.sex_model is None:
            logger.warning("  Sex model not loaded - skipping sex accuracy computation")
    
    # Race Accuracy - use enhanced method from validation_metrics.py
    if labels.get("race") is not None:
        logger.info("  Computing race accuracy...")
        try:
            race_labels_gpu = labels["race"].to(device)
            race_acc = metrics_runner.text_alignment.compute_race_accuracy(
                images_gpu, race_labels_gpu, batch_size=batch_size
            )
            if not np.isnan(race_acc):
                metrics["race_accuracy"] = float(race_acc)
                logger.info(f"    Race accuracy: {race_acc:.4f}")
            else:
                logger.warning("    Race accuracy computation returned NaN")
        except Exception as e:
            logger.warning(f"  Race accuracy computation failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    else:
        logger.warning("  Race labels not available - skipping race accuracy computation")
    
    # Age RMSE - use enhanced method from validation_metrics.py
    if labels.get("age") is not None:
        logger.info("  Computing age RMSE...")
        try:
            age_labels_gpu = labels["age"].to(device)
            age_rmse = metrics_runner.text_alignment.compute_age_rmse(
                images_gpu, age_labels_gpu, batch_size=batch_size
            )
            metrics["age_rmse"] = float(age_rmse)
            logger.info(f"    Age RMSE: {age_rmse:.4f}")
        except Exception as e:
            logger.warning(f"  Age RMSE computation failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    else:
        logger.warning("  Age labels not available - skipping age RMSE computation")
    
    # Clean up GPU memory
    del images_gpu
    if 'disease_labels_gpu' in locals():
        del disease_labels_gpu
    if 'sex_labels_gpu' in locals():
        del sex_labels_gpu
    if 'race_labels_gpu' in locals():
        del race_labels_gpu
    if 'age_labels_gpu' in locals():
        del age_labels_gpu
    torch.cuda.empty_cache()
    
    return metrics


def compute_similarity_metrics(
    metrics_runner: ValidationMetricsRunner,
    real_images: torch.Tensor,
    synth_images: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 16
) -> Dict[str, float]:
    """Compute similarity metrics between two halves of real data in batches."""
    logger.info(f"Computing similarity metrics (real vs real split, batch_size={batch_size})...")
    
    metrics = {}
    num_images = len(real_images)
    
    # FID (Inception v3) - uses batch processing internally, but we need to ensure images stay on CPU
    logger.info("  Computing FID (Inception v3)...")
    try:
        # FID computation loads images in batches internally
        metrics["fid"] = metrics_runner.similarity.compute_fid(
            real_images, synth_images, batch_size=batch_size
        )
    except Exception as e:
        logger.warning(f"  FID (Inception v3) computation failed: {e}")
        metrics["fid"] = np.nan
    
    # FID (RadImageNet ResNet50) - medical image specific FID
    logger.info("  Computing FID (RadImageNet ResNet50)...")
    try:
        # RadImageNet FID computation loads images in batches internally
        metrics["fid_radimagenet"] = metrics_runner.similarity.compute_fid_radimagenet(
            real_images, synth_images, batch_size=batch_size
        )
    except Exception as e:
        logger.warning(f"  FID (RadImageNet ResNet50) computation failed: {e}")
        metrics["fid_radimagenet"] = np.nan
    
    # BioViL Similarity - process in batches
    logger.info("  Computing BioViL similarity...")
    try:
        biovil_similarities = []
        
        for i in range(0, num_images, batch_size):
            end_i = min(i + batch_size, num_images)
            batch_real = real_images[i:end_i].to(device)
            batch_synth = synth_images[i:end_i].to(device)
            
            # Compute similarity for this batch
            similarity = metrics_runner.similarity.compute_biovil_similarity(
                batch_real, batch_synth
            )
            if not np.isnan(similarity):
                biovil_similarities.append(similarity)
            
            # Clear GPU cache
            del batch_real, batch_synth
            torch.cuda.empty_cache()
        
        if biovil_similarities:
            metrics["biovil_similarity"] = np.mean(biovil_similarities)
        else:
            metrics["biovil_similarity"] = np.nan
    except Exception as e:
        logger.warning(f"  BioViL similarity computation failed: {e}")
        metrics["biovil_similarity"] = np.nan
    
    # MS-SSIM - process in batches
    logger.info("  Computing MS-SSIM...")
    try:
        msssim_values = []
        
        for i in range(0, num_images, batch_size):
            end_i = min(i + batch_size, num_images)
            batch_real = real_images[i:end_i].to(device)
            batch_synth = synth_images[i:end_i].to(device)
            
            # Compute MS-SSIM for this batch
            msssim = metrics_runner.similarity.compute_ms_ssim(
                batch_real, batch_synth
            )
            if not np.isnan(msssim):
                msssim_values.append(msssim)
            
            # Clear GPU cache
            del batch_real, batch_synth
            torch.cuda.empty_cache()
        
        if msssim_values:
            metrics["ms_ssim"] = np.mean(msssim_values)
        else:
            metrics["ms_ssim"] = np.nan
    except Exception as e:
        logger.warning(f"  MS-SSIM computation failed: {e}")
        metrics["ms_ssim"] = np.nan
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Calculate validation metrics on real data only",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file (optional)
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to config YAML file (optional)"
    )
    
    # Data paths
    parser.add_argument(
        "--validation_csv",
        type=str,
        required=True,
        help="Path to validation CSV or directory"
    )
    parser.add_argument(
        "--validation_images_dir",
        type=str,
        default=None,
        help="Path to validation images directory (for directory dataset)"
    )
    parser.add_argument(
        "--use_wds_dataset",
        action="store_true",
        help="Use WebDataset format"
    )
    
    # Model configuration
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Pretrained model path (for tokenizer)"
    )
    parser.add_argument(
        "--validation_sex_model_path",
        type=str,
        default=None,
        help="Path to sex prediction model checkpoint (optional)"
    )
    
    # Data parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use (-1 for all)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for metrics computation (default: 16)"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="real_data_metrics",
        help="Output directory for results"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config_file:
        config = load_config(args.config_file) or {}
    
    # Override with CLI args
    if args.validation_csv:
        config["validation_csv"] = args.validation_csv
    if args.validation_images_dir:
        config["validation_images_dir"] = args.validation_images_dir
    if args.use_wds_dataset:
        config["use_wds_dataset"] = True
    if args.validation_sex_model_path:
        config["validation_sex_model_path"] = args.validation_sex_model_path
    if args.batch_size is not None:
        config["validation_metrics_batch_size"] = args.batch_size
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    
    # Verify device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer (needed for dataset loading)
    logger.info("Loading tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    
    # Load real validation data
    images, labels, prompts = load_real_validation_data(
        validation_csv=config["validation_csv"],
        validation_images_dir=config.get("validation_images_dir"),
        use_wds_dataset=config.get("use_wds_dataset", False),
        num_samples=args.num_samples,
        tokenizer=tokenizer,
        use_hcn=config.get("use_hcn", False),
    )
    
    # Initialize metrics runner
    logger.info("Initializing validation metrics runner...")
    sex_model_path = config.get("validation_sex_model_path")
    if sex_model_path:
        logger.info(f"Sex model path: {sex_model_path}")
        if not os.path.exists(sex_model_path):
            logger.warning(f"Sex model checkpoint not found at {sex_model_path}")
        else:
            logger.info(f"Sex model checkpoint found (size: {os.path.getsize(sex_model_path) / 1024**2:.2f} MB)")
    else:
        logger.warning("No sex model path provided - sex accuracy will be skipped")
    
    metrics_runner = ValidationMetricsRunner(
        device=args.device,
        sex_model_path=sex_model_path,
    )
    
    # Verify models are loaded
    logger.info("Verifying models are loaded:")
    logger.info(f"  Disease model: {metrics_runner.text_alignment.disease_model is not None}")
    logger.info(f"  Race model: {metrics_runner.text_alignment.race_model is not None}")
    logger.info(f"  Age model: {metrics_runner.text_alignment.age_model is not None}")
    logger.info(f"  Sex model: {metrics_runner.text_alignment.sex_model is not None}")
    if metrics_runner.text_alignment.sex_model is not None:
        logger.info(f"  Sex model device: {next(metrics_runner.text_alignment.sex_model.parameters()).device}")
        logger.info(f"  Sex model training mode: {metrics_runner.text_alignment.sex_model.training}")
    
    # Get batch size from config or use default
    batch_size = config.get("validation_metrics_batch_size", 16)
    logger.info(f"Using batch size: {batch_size} for metrics computation")
    
    # Compute text alignment metrics on ALL real data
    logger.info("="*60)
    logger.info("TEXT ALIGNMENT METRICS (on all real data)")
    logger.info("="*60)
    text_metrics = compute_text_alignment_metrics(
        metrics_runner, images, labels, device=args.device, batch_size=batch_size
    )
    
    # Split data in half for similarity metrics
    logger.info("="*60)
    logger.info("SIMILARITY METRICS (real vs real split)")
    logger.info("="*60)
    (
        real_half_images, synth_half_images,
        real_half_labels, synth_half_labels,
        real_half_prompts, synth_half_prompts
    ) = split_data_in_half(images, labels, prompts)
    
    # Compute similarity metrics
    similarity_metrics = compute_similarity_metrics(
        metrics_runner, real_half_images, synth_half_images, device=args.device, batch_size=batch_size
    )
    
    # Combine all metrics
    all_metrics = {**text_metrics, **similarity_metrics}
    
    # Print summary
    logger.info("="*60)
    logger.info("METRICS SUMMARY")
    logger.info("="*60)
    logger.info("\nText Alignment Metrics:")
    logger.info("  NOTE: These metrics measure pretrained classifier performance on real images.")
    logger.info("  Low scores may indicate:")
    logger.info("    - Domain shift between classifier training data and validation set")
    logger.info("    - Classifier models need retraining/fine-tuning")
    logger.info("    - Dataset-specific issues (label quality, image preprocessing)")
    logger.info("")
    for key, value in text_metrics.items():
        if isinstance(value, float) and not np.isnan(value):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info("\nSimilarity Metrics (real vs real split):")
    logger.info("  NOTE: These represent 'best case' scores (real vs real comparison).")
    logger.info("  When comparing real vs synthetic, expect:")
    logger.info("    - Higher FID (worse)")
    logger.info("    - Lower BioViL similarity (worse)")
    logger.info("    - Lower MS-SSIM (worse)")
    logger.info("")
    for key, value in similarity_metrics.items():
        if isinstance(value, float) and not np.isnan(value):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"real_data_metrics_{timestamp}.json"
    
    results = {
        "timestamp": timestamp,
        "num_samples": len(images),
        "num_samples_per_half": len(real_half_images),
        "config": {
            "validation_csv": config["validation_csv"],
            "validation_images_dir": config.get("validation_images_dir"),
            "use_wds_dataset": config.get("use_wds_dataset", False),
            "num_samples": args.num_samples,
        },
        "metrics": all_metrics,
        "note": "Similarity metrics computed by splitting real data into two halves"
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

