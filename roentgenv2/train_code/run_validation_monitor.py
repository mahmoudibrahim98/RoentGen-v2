#!/usr/bin/env python
"""
Independent validation monitoring script.

This script monitors the output directory for new checkpoints and automatically
runs validation on them. It can be run in parallel with training.

Usage:
    accelerate launch roentgenv2/train_code/run_validation_monitor.py --config_file configs/your_config.yaml
"""

import os
import sys
import time
import json
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
from typing import Optional, Dict, List
import argparse
import yaml
from dataclasses import dataclass

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
try:
    from huggingface_hub.errors import RepositoryNotFoundError
except ImportError:
    # Fallback for older versions
    RepositoryNotFoundError = Exception

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from validation_metrics import ValidationMetricsRunner
from dataset_wds import RGFineTuningImageDirectoryDataset, RGFineTuningWebDataset


logger = get_logger(__name__, log_level="INFO")


@dataclass
class ValidationConfig:
    """Configuration for validation monitoring."""
    # Path to training config
    config_file: str

    # Monitoring settings
    check_interval: int = 1800  # Check for new checkpoints every N seconds
    manifest_file: str = "validation_manifest.json"  # Track validated checkpoints

    # Override validation settings from training config if needed
    num_validation_samples: Optional[int] = None
    val_batch_size: Optional[int] = None
    validation_guidance_scale: Optional[float] = None
    validation_num_inference_steps: Optional[int] = None


def load_config(file_path):
    """Load YAML config file."""
    with open(file_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as e:
            print("Error loading YAML:", e)
            return None


class CheckpointMonitor:
    """Monitors output directory for new checkpoints."""

    def __init__(self, output_dir: str, manifest_file: str):
        self.output_dir = Path(output_dir)
        self.manifest_file = self.output_dir / manifest_file
        self.validated_checkpoints = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Dict]:
        """Load manifest of already validated checkpoints."""
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        """Save manifest of validated checkpoints."""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.validated_checkpoints, f, indent=2)

    def get_new_checkpoints(self) -> List[Path]:
        """Get list of new checkpoints that haven't been validated."""
        if not self.output_dir.exists():
            return []

        # Find all checkpoint directories
        checkpoints = []
        for item in self.output_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                # Extract step number
                try:
                    step = int(item.name.split("-")[1])
                    # Check if not already validated or in progress
                    if item.name not in self.validated_checkpoints:
                        # Check if checkpoint is fully written (has expected files)
                        if self._is_checkpoint_complete(item):
                            checkpoints.append((step, item))
                    else:
                        # Check if validation is in progress (not completed)
                        status = self.validated_checkpoints[item.name]
                        if status.get("status") == "in_progress":
                            # Skip checkpoints that are currently being validated
                            continue
                except (ValueError, IndexError):
                    continue

        # Sort by step number
        checkpoints.sort(key=lambda x: x[0])
        return [ckpt[1] for ckpt in checkpoints]

    def _is_checkpoint_complete(self, checkpoint_dir: Path) -> bool:
        """Check if checkpoint has been fully written."""
        # Check for key files that indicate checkpoint is complete
        # Accelerate saves as model.safetensors (or model.bin), not pytorch_model.bin
        has_model = (checkpoint_dir / "model.safetensors").exists() or (checkpoint_dir / "model.bin").exists() or (checkpoint_dir / "pytorch_model.bin").exists()
        has_random_states = (checkpoint_dir / "random_states_0.pkl").exists()
        return has_model and has_random_states

    def mark_in_progress(self, checkpoint_dir: Path):
        """Mark checkpoint as being validated (in progress)."""
        self.validated_checkpoints[checkpoint_dir.name] = {
            "status": "in_progress",
            "started_at": datetime.now().isoformat(),
            "success": None,
            "metrics": None,
        }
        self._save_manifest()

    def mark_validated(self, checkpoint_dir: Path, metrics: Dict, success: bool = True, num_images: Optional[int] = None):
        """Mark checkpoint as validated (completed)."""
        entry = self.validated_checkpoints.get(checkpoint_dir.name, {})
        entry.update({
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "metrics": metrics if success else None,
            "num_images": num_images,
        })
        self.validated_checkpoints[checkpoint_dir.name] = entry
        self._save_manifest()


class StepImageMonitor:
    """Monitors validation_images directory for new step directories with pre-generated images."""

    def __init__(self, images_base_dir: str, manifest_file: str):
        self.images_base_dir = Path(images_base_dir)
        self.manifest_file = self.images_base_dir.parent / manifest_file
        self.validated_checkpoints = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Dict]:
        """Load manifest of already validated checkpoints."""
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        """Save manifest of validated checkpoints."""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.validated_checkpoints, f, indent=2)

    def get_new_steps(self) -> List[tuple]:
        """Get list of new step directories that haven't been validated.
        
        Returns:
            List of tuples (step_number, step_dir_path)
        """
        if not self.images_base_dir.exists():
            return []

        # Find all step directories (validation images use step_ format)
        checkpoints = []
        for item in self.images_base_dir.iterdir():
            if item.is_dir() and item.name.startswith("step_"):
                # Extract step number
                try:
                    step = int(item.name.split("_")[1])
                    # Use checkpoint- format for manifest keys (unified with checkpoint directories)
                    checkpoint_key = f"checkpoint-{step}"
                    
                    # Check if not already validated or in progress
                    if checkpoint_key not in self.validated_checkpoints:
                        # Check if step directory has images (has gpu subdirectories)
                        if self._is_step_complete(item):
                            checkpoints.append((step, item))
                    else:
                        # Check if validation is in progress (not completed)
                        status = self.validated_checkpoints[checkpoint_key]
                        if status.get("status") == "in_progress":
                            # Skip checkpoints that are currently being validated
                            continue
                except (ValueError, IndexError):
                    continue

        # Sort by step number
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints

    def _is_step_complete(self, step_dir: Path) -> bool:
        """Check if step directory has images ready for validation."""
        # Check if there are gpu subdirectories with images
        gpu_dirs = list(step_dir.glob("gpu_*"))
        if not gpu_dirs:
            return False
        
        # Check if at least one gpu directory has synthetic images
        for gpu_dir in gpu_dirs:
            synth_files = list(gpu_dir.glob("synthetic_*.png"))
            if synth_files:
                return True
        return False

    def mark_in_progress(self, step: int):
        """Mark checkpoint as being validated (in progress)."""
        # Use checkpoint- format for manifest keys (unified with checkpoint directories)
        checkpoint_key = f"checkpoint-{step}"
        self.validated_checkpoints[checkpoint_key] = {
            "status": "in_progress",
            "started_at": datetime.now().isoformat(),
            "success": None,
            "metrics": None,
        }
        self._save_manifest()

    def mark_validated(self, step: int, metrics: Dict, success: bool = True, num_images: Optional[int] = None):
        """Mark checkpoint as validated (completed)."""
        # Use checkpoint- format for manifest keys (unified with checkpoint directories)
        checkpoint_key = f"checkpoint-{step}"
        entry = self.validated_checkpoints.get(checkpoint_key, {})
        entry.update({
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "metrics": metrics if success else None,
            "num_images": num_images,
        })
        self.validated_checkpoints[checkpoint_key] = entry
        self._save_manifest()


class ValidationRunner:
    """Runs validation on a checkpoint."""

    def __init__(
        self,
        accelerator: Accelerator,
        args,
        logger,
    ):
        self.accelerator = accelerator
        self.args = args
        self.logger = logger
        self.load_images_from_dir = args.get("load_images_from_dir", False)

        # Load models (frozen during validation)
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.noise_scheduler = None
        self.weight_dtype = None

        # Models that will be prepared once and reused for all checkpoints
        self.unet = None
        self.hcn = None

        # Validation infrastructure
        self.validation_dataloader = None
        self.metrics_runner = None
        self._metrics_runner_initialized = False  # Initialize flag for lazy metrics runner initialization

        # Only initialize models if we're generating images (not loading from directory)
        if not self.load_images_from_dir:
            self._initialize_fixed_models()
            self._initialize_trainable_models()  # Initialize UNet and HCN once
            self._initialize_validation_data()
        else:
            logger.info("Skipping model initialization (loading images from directory)")
            # Metrics runner will be initialized lazily when needed
    def _rank_prefix(self):
        return f"[Rank {self.accelerator.process_index}/{self.accelerator.num_processes}]"

    def _initialize_fixed_models(self):
        """Initialize models that don't change between checkpoints."""
        logger.info("Loading fixed models (tokenizer, text_encoder, vae, scheduler)...")

        # Tokenizer
        # Handle potential 404 error when transformers tries to check for chat templates
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.args["pretrained_model_name_or_path"],
                subfolder="tokenizer",
                revision=self.args.get("revision"),
                use_auth_token=self.args.get("use_auth_token"),
            )
        except (RepositoryNotFoundError, Exception) as e:
            # If there's an error (e.g., 404 for chat templates), try with local_files_only
            if isinstance(e, RepositoryNotFoundError) or "404" in str(e) or "RepositoryNotFoundError" in str(type(e)):
                logger.warning(f"Got repository error when loading tokenizer, trying with local cache: {e}")
                try:
                    self.tokenizer = CLIPTokenizer.from_pretrained(
                        self.args["pretrained_model_name_or_path"],
                        subfolder="tokenizer",
                        revision=self.args.get("revision"),
                        use_auth_token=self.args.get("use_auth_token"),
                        local_files_only=True,
                    )
                except Exception as e2:
                    # If that fails, try without subfolder (fallback)
                    logger.warning(f"Local files only failed, trying alternative loading: {e2}")
                    self.tokenizer = CLIPTokenizer.from_pretrained(
                        self.args["pretrained_model_name_or_path"],
                        revision=self.args.get("revision"),
                        use_auth_token=self.args.get("use_auth_token"),
                    )
            else:
                raise

        # Text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.args["pretrained_model_name_or_path"],
            subfolder="text_encoder",
            revision=self.args.get("revision"),
            use_auth_token=self.args.get("use_auth_token"),
        )
        
        # VAE (frozen, always fp32)
        self.vae = AutoencoderKL.from_pretrained(
            self.args["pretrained_model_name_or_path"],
            subfolder="vae",
            revision=self.args.get("revision"),
            use_auth_token=self.args.get("use_auth_token"),
        )
        self.vae.eval()
        self.vae.requires_grad_(False)

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.args["pretrained_model_name_or_path"],
            subfolder="scheduler",
        )

        # Don't prepare text_encoder here if it's being trained - it will be loaded from checkpoint
        # Don't prepare VAE at all - it should stay frozen and not be part of checkpoint loading
        if not self.args.get("train_text_encoder", False):
            self.text_encoder.eval()
            self.text_encoder.requires_grad_(False)
            self.text_encoder = self.accelerator.prepare(self.text_encoder)
        
        # Move VAE to device manually (don't use prepare() to avoid checkpoint loading issues)
        self.vae = self.vae.to(self.accelerator.device)

        # Set weight dtype
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
        else:
            self.weight_dtype = torch.float32

        # VAE always stays in fp32
        self.vae.to(dtype=torch.float32)

        logger.info("✓ Fixed models loaded")

    def _initialize_validation_data(self):
        """Initialize validation dataloader and metrics runner."""
        logger.info("Loading validation data...")

        # Load validation dataset
        if self.args.get("use_wds_dataset", False):
            # WebDataset - use validation data path
            val_path = self.args.get("validation_csv") or self.args.get("validation_images_dir")
            if val_path is None:
                raise ValueError("validation_csv or validation_images_dir must be specified for WebDataset validation")
            
            # If val_path is a directory, construct tar file paths
            import os
            import glob
            if os.path.isdir(val_path):
                # Find all tar files in the directory
                tar_files = sorted(glob.glob(os.path.join(val_path, "*.tar")))
                if not tar_files:
                    raise ValueError(f"No tar files found in validation directory: {val_path}")
                val_urls = tar_files
            else:
                val_urls = [val_path] if isinstance(val_path, str) else val_path
            
            validation_dataset = RGFineTuningWebDataset(
                url_list=val_urls,
                tokenizer=self.tokenizer,
                use_hcn=self.args.get("use_hcn", False),
                include_text=True,
            )
        else:
            # Directory dataset
            validation_dataset = RGFineTuningImageDirectoryDataset(
                image_dir_path=self.args["validation_images_dir"],
                text_dir_path=self.args["validation_csv"],
                tokenizer=self.tokenizer,
                use_hcn=self.args.get("use_hcn", False),
                include_text=True,
            )

        # Create dataloader
        self.validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=1,  # We'll batch during generation
            shuffle=False,
            num_workers=4,
        )

        try:
            dataset_size = len(validation_dataset)
            logger.info(f"✓ Validation dataset loaded ({dataset_size} samples)")
        except:
            logger.info(f"✓ Validation dataset loaded")

        # NOTE: Metrics runner initialization is EXPENSIVE and blocks Rank 0
        # We initialize it lazily on first use to avoid blocking other ranks
        # during ValidationRunner initialization
        self.metrics_runner = None
        self._metrics_runner_initialized = False

    def _initialize_trainable_models(self):
        """Initialize UNet and HCN architectures (called once during __init__)."""
        logger.info("Initializing trainable model architectures (UNet, HCN)...")
        
        # Initialize UNet architecture
        self.unet = UNet2DConditionModel.from_pretrained(
            self.args["pretrained_model_name_or_path"],
            subfolder="unet",
            revision=self.args.get("revision"),
            use_auth_token=self.args.get("use_auth_token"),
        )
        
        # Initialize HCN if enabled
        # CRITICAL: Must initialize HCN on ALL ranks consistently before prepare()
        # DDP requires all ranks to have the same model structure
        self.hcn = None
        
        if self.args.get("use_hcn", False):
            # Initialize HCN on all ranks - if it fails on any rank, we need to know
            # All ranks must execute this code path (no is_main_process check)
            try:
                from hcn import HierarchicalConditioner
                self.hcn = HierarchicalConditioner(
                    num_age_bins=self.args.get("hcn_num_age_bins", 5),
                    num_sex=self.args.get("hcn_num_sex", 2),
                    num_race=self.args.get("hcn_num_race", 4),
                    d_node=self.args.get("hcn_d_node", 256),
                    d_ctx=self.args.get("hcn_d_ctx", 1024),
                    dropout=self.args.get("hcn_dropout", 0.1),
                    use_uncertainty=self.args.get("hcn_use_uncertainty", True),
                )
                if self.accelerator.is_local_main_process:
                    logger.info(f"✓ HCN architecture initialized on all ranks")
            except Exception as e:
                logger.error(f"Rank {self.accelerator.process_index}: Failed to initialize HCN: {e}")
                raise RuntimeError(
                    f"HCN initialization failed on rank {self.accelerator.process_index}. "
                    f"DDP requires HCN to be initialized consistently on all ranks. Error: {e}"
                )
        
        # For validation, we DON'T wrap models in DDP (no gradient sync needed)
        # We'll just move them to the correct device and load checkpoint states manually
        # This avoids the prepare() hang issue in distributed validation
        if self.accelerator.is_local_main_process:
            logger.info("Moving models to device (skipping DDP wrapping for validation)...")
        
        # Move models to accelerator device
        self.unet = self.unet.to(self.accelerator.device)
        if self.hcn is not None:
            self.hcn = self.hcn.to(self.accelerator.device)
        if self.args.get("train_text_encoder", False):
            self.text_encoder = self.text_encoder.to(self.accelerator.device)
        
        # Set to eval mode
        self.unet.eval()
        self.unet.requires_grad_(False)
        if self.hcn is not None:
            self.hcn.eval()
            self.hcn.requires_grad_(False)
        if self.args.get("train_text_encoder", False):
            self.text_encoder.eval()
            self.text_encoder.requires_grad_(False)
        
        if self.accelerator.is_local_main_process:
            logger.info("✓ Trainable models initialized")


    def _ensure_metrics_runner(self):
        """Lazy initialization of metrics runner (on ALL ranks for streaming)."""
        if not self._metrics_runner_initialized:  # ← FIXED! No is_main_process check
            logger.info(f"{self._rank_prefix()} Initializing validation metrics runner...")
            device_str = str(self.accelerator.device)
            self.metrics_runner = ValidationMetricsRunner(
                device=device_str,
                sex_model_path=self.args.get("validation_sex_model_path"),
            )
            self._metrics_runner_initialized = True
            logger.info(f"{self._rank_prefix()} ✓ Metrics runner initialized")
    def load_checkpoint(self, checkpoint_dir: Path):
        """Load checkpoint weights into models manually (without DDP)."""
        logger.info(f"Loading checkpoint from {checkpoint_dir}")

        # Since we're not using prepare() (to avoid DDP hangs), we load checkpoints manually
        # Load model weights from safetensors or pytorch bin files
        try:
            import safetensors.torch
            checkpoint_path = Path(checkpoint_dir)
            
            # Load UNet
            if (checkpoint_path / "model.safetensors").exists():
                logger.info("Loading UNet from model.safetensors...")
                state_dict = safetensors.torch.load_file(checkpoint_path / "model.safetensors")
                self.unet.load_state_dict(state_dict, strict=False)
            elif (checkpoint_path / "pytorch_model.bin").exists():
                logger.info("Loading UNet from pytorch_model.bin...")
                state_dict = torch.load(checkpoint_path / "pytorch_model.bin", map_location=self.accelerator.device)
                self.unet.load_state_dict(state_dict, strict=False)
            
            # Load text encoder if trained
            if self.args.get("train_text_encoder", False):
                if (checkpoint_path / "model_1.safetensors").exists():
                    logger.info("Loading text encoder from model_1.safetensors...")
                    state_dict = safetensors.torch.load_file(checkpoint_path / "model_1.safetensors")
                    self.text_encoder.load_state_dict(state_dict, strict=False)
                elif (checkpoint_path / "pytorch_model_1.bin").exists():
                    logger.info("Loading text encoder from pytorch_model_1.bin...")
                    state_dict = torch.load(checkpoint_path / "pytorch_model_1.bin", map_location=self.accelerator.device)
                    self.text_encoder.load_state_dict(state_dict, strict=False)
            
            # Load HCN if enabled
            if self.hcn is not None:
                if (checkpoint_path / "model_2.safetensors").exists():
                    logger.info("Loading HCN from model_2.safetensors...")
                    state_dict = safetensors.torch.load_file(checkpoint_path / "model_2.safetensors")
                    self.hcn.load_state_dict(state_dict, strict=False)
                elif (checkpoint_path / "pytorch_model_2.bin").exists():
                    logger.info("Loading HCN from pytorch_model_2.bin...")
                    state_dict = torch.load(checkpoint_path / "pytorch_model_2.bin", map_location=self.accelerator.device)
                    self.hcn.load_state_dict(state_dict, strict=False)
            
            logger.info("✓ Checkpoint state loaded")
        except FileNotFoundError as e:
            # Check if the error is about EMA model (model_3) - we don't need it for validation
            if "pytorch_model_3" in str(e) or "model_3" in str(e):
                logger.warning(f"EMA model file not found (this is OK for validation): {e}")
                # Try to load without EMA by manually loading the models we need
                # The main models should already be loaded, but let's try to continue
                logger.info("Attempting to load models manually...")
                try:
                    # Load main models directly from safetensors if available
                    import safetensors.torch
                    checkpoint_path = Path(checkpoint_dir)
                    
                    # Load UNet
                    if (checkpoint_path / "model.safetensors").exists():
                        logger.info("Loading UNet from model.safetensors...")
                        state_dict = safetensors.torch.load_file(checkpoint_path / "model.safetensors")
                        missing, unexpected = self.accelerator.unwrap_model(self.unet).load_state_dict(state_dict, strict=False)
                        if missing:
                            logger.warning(f"Missing keys in UNet: {len(missing)} keys")
                        if unexpected:
                            logger.warning(f"Unexpected keys in UNet: {len(unexpected)} keys")
                    
                    # Load text encoder if trained
                    if self.args.get("train_text_encoder", False) and (checkpoint_path / "model_1.safetensors").exists():
                        logger.info("Loading text encoder from model_1.safetensors...")
                        state_dict = safetensors.torch.load_file(checkpoint_path / "model_1.safetensors")
                        missing, unexpected = self.accelerator.unwrap_model(self.text_encoder).load_state_dict(state_dict, strict=False)
                        if missing:
                            logger.warning(f"Missing keys in text encoder: {len(missing)} keys")
                        if unexpected:
                            logger.warning(f"Unexpected keys in text encoder: {len(unexpected)} keys")
                    
                    # Load HCN if enabled
                    if self.hcn is not None and (checkpoint_path / "model_2.safetensors").exists():
                        logger.info("Loading HCN from model_2.safetensors...")
                        state_dict = safetensors.torch.load_file(checkpoint_path / "model_2.safetensors")
                        missing, unexpected = self.accelerator.unwrap_model(self.hcn).load_state_dict(state_dict, strict=False)
                        if missing:
                            logger.warning(f"Missing keys in HCN: {len(missing)} keys")
                        if unexpected:
                            logger.warning(f"Unexpected keys in HCN: {len(unexpected)} keys")
                    
                    logger.info("✓ Checkpoint models loaded manually (skipped EMA model)")
                except Exception as manual_load_error:
                    logger.error(f"Failed to load models manually: {manual_load_error}")
                    raise e  # Re-raise original error if manual load fails
            else:
                # Re-raise if it's a different FileNotFoundError
                raise

        # Set to eval mode
        self.unet.eval()
        self.unet.requires_grad_(False)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        if self.hcn is not None:
            self.hcn.eval()
            self.hcn.requires_grad_(False)

        logger.info("✓ Checkpoint loaded successfully")


    def _generate_validation_images(self, unet, text_encoder, vae, global_step):
        """Generate validation images (distributed across GPUs)."""
        all_synthetic_images = []
        all_real_images = []
        all_labels = {
            "disease": [],
            "sex": [],
            "race": [],
            "age": [],
            "prompts": [],
        }

        num_images_per_prompt = self.args.get("validation_num_images_per_prompt", 4)
        max_prompts = self.args.get("num_validation_samples", 100)
        generation_batch_size = self.args.get("val_batch_size", 1)

        # Handle -1 as "use all samples"
        use_all_samples = (max_prompts == -1)

        # Collect all prompts first
        all_prompts_data = []
        num_prompts_collected = 0

        for batch in self.validation_dataloader:
            if not use_all_samples and num_prompts_collected >= max_prompts:
                break

            batch_size = batch["pixel_values"].shape[0]
            if use_all_samples:
                samples_to_take = batch_size
            else:
                samples_to_take = min(batch_size, max_prompts - num_prompts_collected)

            for i in range(samples_to_take):
                prompt = batch.get("text", [""])[i] if isinstance(batch.get("text"), list) else batch.get("text", "")
                all_prompts_data.append({
                    "prompt": prompt,
                    "batch": batch,
                    "batch_idx": i,
                })

            num_prompts_collected += samples_to_take

        if self.accelerator.is_local_main_process:
            if use_all_samples:
                logger.info(f"Collected {len(all_prompts_data)} prompts from validation dataset (using all samples)")
            else:
                logger.info(f"Collected {len(all_prompts_data)} prompts from validation dataset (limit: {max_prompts})")
            total_images = len(all_prompts_data) * num_images_per_prompt
            logger.info(f"Will generate {total_images} images total")
            logger.info(f"  - {num_images_per_prompt} image(s) per prompt")
            logger.info(f"  - Distributed across {self.accelerator.num_processes} GPU(s)")

        # Distribute prompts across GPUs
        num_gpus = self.accelerator.num_processes
        gpu_idx = self.accelerator.process_index

        prompts_for_this_gpu = []
        for i, prompt_idx in enumerate(range(len(all_prompts_data))):
            if i % num_gpus == gpu_idx:
                prompts_for_this_gpu.append(prompt_idx)

        # Generate images for assigned prompts
        images_to_generate = []
        for prompt_idx in prompts_for_this_gpu:
            for image_idx in range(num_images_per_prompt):
                images_to_generate.append((prompt_idx, image_idx))

        if self.accelerator.is_local_main_process:
            logger.info(f"GPU {gpu_idx}: Generating {len(images_to_generate)} images ({len(prompts_for_this_gpu)} prompts × {num_images_per_prompt} images)")

        # Generate with progress bar
        with torch.no_grad():
            progress_bar = tqdm(
                total=len(images_to_generate),
                desc=f"Generating images (GPU {gpu_idx})",
                disable=not self.accelerator.is_local_main_process,
            )

            for batch_start in range(0, len(images_to_generate), generation_batch_size):
                batch_end = min(batch_start + generation_batch_size, len(images_to_generate))
                batch_images = images_to_generate[batch_start:batch_end]

                # Generate batch
                synthetic_batch, real_batch, labels_batch = self._generate_image_batch(
                    batch_images, all_prompts_data, unet, text_encoder, vae, global_step, batch_start
                )

                # Store results
                all_synthetic_images.extend(synthetic_batch)
                all_real_images.extend(real_batch)
                for key in labels_batch:
                    all_labels[key].extend(labels_batch[key])

                progress_bar.update(len(batch_images))

            progress_bar.close()

        return all_synthetic_images, all_real_images, all_labels

    def _generate_image_batch(self, batch_images, all_prompts_data, unet, text_encoder, vae, global_step, batch_start_idx):
        """Generate a batch of images."""
        # Prepare prompts
        prompts = []
        batch_data_list = []

        for (prompt_idx, image_idx) in batch_images:
            prompt_data = all_prompts_data[prompt_idx]
            prompts.append(prompt_data["prompt"])
            batch_data_list.append((prompt_data["batch"], prompt_data["batch_idx"]))

        local_batch_size = len(prompts)

        # Tokenize prompts
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.accelerator.device)

        # Get text embeddings
        prompt_embeds = text_encoder(input_ids=text_input_ids, return_dict=False)
        text_embeddings = prompt_embeds[0]

        # Add HCN conditioning if available
        if self.hcn is not None:
            text_embeddings = self._add_hcn_conditioning(text_embeddings, batch_data_list)

        # Create unconditional embeddings
        uncond_inputs = self.tokenizer(
            [""] * local_batch_size,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_inputs.input_ids.to(self.accelerator.device)
        uncond_embeds = text_encoder(input_ids=uncond_input_ids, return_dict=False)
        uncond_embeddings = uncond_embeds[0]

        # Ensure unconditional embeddings match the sequence length of conditional ones.
        # We intentionally DO NOT add HCN conditioning here; instead we append a zero
        # demographic token so classifier-free guidance still works mathematically.
        if self.hcn is not None:
            zero_ctx = torch.zeros(
                (uncond_embeddings.shape[0], 1, uncond_embeddings.shape[-1]),
                device=uncond_embeddings.device,
                dtype=uncond_embeddings.dtype,
            )
            uncond_embeddings = torch.cat([uncond_embeddings, zero_ctx], dim=1)

        # Concatenate for classifier-free guidance
        encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings], dim=0)

        # Prepare latents
        latents_shape = (
            local_batch_size,
            unet.config.in_channels,
            self.args["resolution"] // 8,
            self.args["resolution"] // 8,
        )

        # Generate latents with unique seeds
        generators = [
            torch.Generator(device=self.accelerator.device).manual_seed(
                global_step * 10000 + self.accelerator.process_index * 1000 + batch_start_idx + i
            )
            for i in range(local_batch_size)
        ]
        latents = torch.stack([
            torch.randn(
                (1, unet.config.in_channels, self.args["resolution"] // 8, self.args["resolution"] // 8),
                generator=gen,
                device=self.accelerator.device,
                dtype=text_embeddings.dtype,
            )[0] for gen in generators
        ])

        init_noise_sigma = getattr(self.noise_scheduler, 'init_noise_sigma', 1.0)
        latents = latents * init_noise_sigma

        # Diffusion loop
        self.noise_scheduler.set_timesteps(self.args.get("validation_num_inference_steps", 50))
        timesteps = self.noise_scheduler.timesteps.to(self.accelerator.device)
        self.noise_scheduler.timesteps = timesteps

        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

            timestep_tensor = t.expand(latent_model_input.shape[0]) if t.ndim == 0 else t.repeat(latent_model_input.shape[0] // t.shape[0])

            noise_pred = unet(
                latent_model_input,
                timestep_tensor,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guidance_scale = self.args.get("validation_guidance_scale", 7.5)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents
        latents = 1 / 0.18215 * latents
        latents_for_decode = latents.to(dtype=torch.float32)
        images = vae.decode(latents_for_decode).sample
        images = (images / 2 + 0.5).clamp(0, 1)

        # Store results
        synthetic_images = [images[i].cpu() for i in range(local_batch_size)]
        # Real images from dataset are in [-1, 1] range, convert to [0, 1] for saving (same as synthetic)
        real_images_raw = [batch_data_list[i][0]["pixel_values"][batch_data_list[i][1]].cpu() for i in range(local_batch_size)]
        real_images = [(img / 2 + 0.5).clamp(0, 1) for img in real_images_raw]

        labels = {
            "disease": [],
            "sex": [],
            "race": [],
            "age": [],
            "prompts": prompts,
        }

        for i in range(local_batch_size):
            batch, batch_idx = batch_data_list[i]
            if "disease_labels" in batch:
                labels["disease"].append(batch["disease_labels"][batch_idx].cpu())
            if "sex_idx" in batch:
                labels["sex"].append(batch["sex_idx"][batch_idx].cpu())
            if "race_idx" in batch:
                labels["race"].append(batch["race_idx"][batch_idx].cpu())
            if "age" in batch:
                labels["age"].append(batch["age"][batch_idx].cpu())

        return synthetic_images, real_images, labels

    def _add_hcn_conditioning(self, text_embeddings, batch_data_list):
        """Add HCN conditioning to text embeddings."""
        if self.hcn is None:
            return text_embeddings

        age_indices = []
        sex_indices = []
        race_indices = []

        for (batch_data, batch_idx) in batch_data_list:
            if "age_idx" in batch_data and "sex_idx" in batch_data and "race_idx" in batch_data:
                age_idx = batch_data["age_idx"][batch_idx] if batch_data["age_idx"].dim() > 0 else batch_data["age_idx"]
                sex_idx = batch_data["sex_idx"][batch_idx] if batch_data["sex_idx"].dim() > 0 else batch_data["sex_idx"]
                race_idx = batch_data["race_idx"][batch_idx] if batch_data["race_idx"].dim() > 0 else batch_data["race_idx"]

                age_indices.append(age_idx)
                sex_indices.append(sex_idx)
                race_indices.append(race_idx)

        if age_indices:
            age_indices = torch.stack(age_indices).squeeze().to(self.accelerator.device)
            sex_indices = torch.stack(sex_indices).squeeze().to(self.accelerator.device)
            race_indices = torch.stack(race_indices).squeeze().to(self.accelerator.device)

            if age_indices.dim() == 0:
                age_indices = age_indices.unsqueeze(0)
            if sex_indices.dim() == 0:
                sex_indices = sex_indices.unsqueeze(0)
            if race_indices.dim() == 0:
                race_indices = race_indices.unsqueeze(0)

            hcn_unwrapped = self.accelerator.unwrap_model(self.hcn)
            hcn_unwrapped.eval()
            hcn_ctx, _, _ = hcn_unwrapped(age_indices, sex_indices, race_indices)

            text_embeddings = torch.cat([text_embeddings, hcn_ctx], dim=1)

        return text_embeddings

    def _load_validation_images_from_disk(self, global_step):
        """Load all validation images from disk (saved by all GPUs)."""
        # Always use standard validation_images directory structure
        output_dir = Path(self.args["output_dir"])
        validation_images_dir = output_dir / "validation_images" / f"step_{global_step}"
        return self._load_validation_images_from_custom_dir(validation_images_dir)
    
    def _load_validation_images_from_custom_dir(self, images_dir):
        """Load all validation images from a custom directory.
        
        Args:
            images_dir: Path to directory containing subdirectories like 'gpu_0', 'gpu_1', etc.
                       Each subdirectory should contain synthetic_*.png, real_*.png, and labels.pkl files.
        """
        from PIL import Image
        import torchvision.transforms as transforms
        import pickle
        import glob
        
        images_dir = Path(images_dir)
        
        if not images_dir.exists():
            logger.warning(f"Validation images directory not found: {images_dir}")
            return None, None, None
        
        # Find all GPU directories
        gpu_dirs = sorted(glob.glob(str(images_dir / "gpu_*")))
        
        if not gpu_dirs:
            logger.warning(f"No GPU directories found in {images_dir}")
            return None, None, None
        
        logger.info(f"Loading validation images from {len(gpu_dirs)} GPU directories...")
        
        # Collect all image paths and labels
        synthetic_image_paths = []
        real_image_paths = []
        all_labels_dict = {
            "disease": [],
            "sex": [],
            "race": [],
            "age": [],
        }
        
        # Transform to convert PIL to tensor (images are saved as PNG)
        # IMPORTANT: Ensure consistent resolution - images should be 512x512
        transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),  # Ensure 512x512
            transforms.ToTensor(),  # Converts PIL to tensor [0, 1]
        ])
        
        for gpu_dir in gpu_dirs:
            gpu_dir = Path(gpu_dir)
            
            # Load labels if available
            labels_path = gpu_dir / "labels.pkl"
            if labels_path.exists():
                with open(labels_path, 'rb') as f:
                    gpu_labels = pickle.load(f)
                    if gpu_labels.get("disease"):
                        all_labels_dict["disease"].extend(gpu_labels["disease"])
                    if gpu_labels.get("sex"):
                        all_labels_dict["sex"].extend(gpu_labels["sex"])
                    if gpu_labels.get("race"):
                        all_labels_dict["race"].extend(gpu_labels["race"])
                    if gpu_labels.get("age"):
                        all_labels_dict["age"].extend(gpu_labels["age"])
            
            # Find all synthetic and real images
            synth_files = sorted(gpu_dir.glob("synthetic_*.png"))
            real_files = sorted(gpu_dir.glob("real_*.png"))
            
            synthetic_image_paths.extend(synth_files)
            if real_files:
                real_image_paths.extend(real_files)
        
        logger.info(f"Found {len(synthetic_image_paths)} synthetic images and {len(real_image_paths)} real images")
        
        # Load images in batches to avoid memory issues
        def load_images_batch(image_paths, batch_size=32, crop_black_borders=False):
            """Load images in batches.
            
            Args:
                image_paths: List of image paths
                batch_size: Batch size for loading
                crop_black_borders: If True, crop black borders (for real images with SquarePad artifacts)
            """
            all_images = []
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert("RGB")
                        
                        # Crop black borders if requested (for real images with SquarePad artifacts)
                        if crop_black_borders:
                            import numpy as np
                            img_array = np.array(img)
                            # Find non-black regions (threshold at 5 to handle near-black pixels)
                            non_black = np.any(img_array > 5, axis=2)
                            if non_black.any():
                                rows = np.where(non_black.any(axis=1))[0]
                                cols = np.where(non_black.any(axis=0))[0]
                                if len(rows) > 0 and len(cols) > 0:
                                    top, bottom = rows[0], rows[-1] + 1
                                    left, right = cols[0], cols[-1] + 1
                                    # Crop to content area
                                    img = img.crop((left, top, right, bottom))
                        
                        img_tensor = transform(img)
                        # Convert to [C, H, W] format (already from ToTensor)
                        all_images.append(img_tensor)
                    except Exception as e:
                        logger.warning(f"Failed to load image {path}: {e}")
            return all_images
        
        # Load synthetic images (no border cropping needed - already square)
        synthetic_images = load_images_batch(synthetic_image_paths, crop_black_borders=False)
        # Load real images with black border cropping to remove SquarePad artifacts
        real_images = load_images_batch(real_image_paths, crop_black_borders=True) if real_image_paths else None
        
        # Stack labels if available
        disease_labels = torch.stack(all_labels_dict["disease"]) if all_labels_dict["disease"] else None
        sex_labels = torch.stack(all_labels_dict["sex"]) if all_labels_dict["sex"] else None
        race_labels = torch.stack(all_labels_dict["race"]) if all_labels_dict["race"] else None
        age_labels = torch.stack(all_labels_dict["age"]) if all_labels_dict["age"] else None
        
        return synthetic_images, real_images, {
            "disease": disease_labels,
            "sex": sex_labels,
            "race": race_labels,
            "age": age_labels,
        }


    def _move_metrics_models_to_gpu(self):
        """Move validation metric models to GPU."""
        if hasattr(self.metrics_runner, 'similarity'):
            if hasattr(self.metrics_runner.similarity, 'fid_model') and self.metrics_runner.similarity.fid_model is not None:
                self.metrics_runner.similarity.fid_model.to(self.accelerator.device)
            if hasattr(self.metrics_runner.similarity, 'biovil_model') and self.metrics_runner.similarity.biovil_model is not None:
                self.metrics_runner.similarity.biovil_model.to(self.accelerator.device)
        if hasattr(self.metrics_runner, 'text_alignment'):
            if hasattr(self.metrics_runner.text_alignment, 'disease_model') and self.metrics_runner.text_alignment.disease_model is not None:
                self.metrics_runner.text_alignment.disease_model.to(self.accelerator.device)
            if hasattr(self.metrics_runner.text_alignment, 'sex_model') and self.metrics_runner.text_alignment.sex_model is not None:
                self.metrics_runner.text_alignment.sex_model.to(self.accelerator.device)
            if hasattr(self.metrics_runner.text_alignment, 'race_model') and self.metrics_runner.text_alignment.race_model is not None:
                self.metrics_runner.text_alignment.race_model.to(self.accelerator.device)
            if hasattr(self.metrics_runner.text_alignment, 'age_model') and self.metrics_runner.text_alignment.age_model is not None:
                self.metrics_runner.text_alignment.age_model.to(self.accelerator.device)

    def _save_validation_images(self, all_synthetic_images, all_real_images, all_labels, global_step):
        """Save validation images and labels to disk if requested."""
        if not all_synthetic_images:
            return
        
        try:
            from torchvision.utils import save_image
            import pickle
            
            # Create output directory for validation images
            output_dir = Path(self.args["output_dir"])
            validation_images_dir = output_dir / "validation_images" / f"step_{global_step}"
            validation_images_dir.mkdir(parents=True, exist_ok=True)
            
            # Save images from this GPU
            gpu_idx = self.accelerator.process_index
            gpu_dir = validation_images_dir / f"gpu_{gpu_idx}"
            gpu_dir.mkdir(parents=True, exist_ok=True)
            
            prompts = all_labels.get("prompts", [""] * len(all_synthetic_images))
            
            # Save labels for later loading
            labels_to_save = {
                "disease": all_labels.get("disease", []),
                "sex": all_labels.get("sex", []),
                "race": all_labels.get("race", []),
                "age": all_labels.get("age", []),
                "prompts": prompts,
            }
            
            for i, (synth_img, real_img) in enumerate(zip(all_synthetic_images, all_real_images)):
                # Save synthetic image
                synth_path = gpu_dir / f"synthetic_{i:06d}.png"
                save_image(synth_img, synth_path)
                
                # Save real image if available
                if real_img is not None:
                    real_path = gpu_dir / f"real_{i:06d}.png"
                    save_image(real_img, real_path)
                
                # Save prompt if available
                if i < len(prompts) and prompts[i]:
                    prompt_path = gpu_dir / f"prompt_{i:06d}.txt"
                    with open(prompt_path, 'w') as f:
                        f.write(str(prompts[i]))
            
            # Save labels as pickle file
            labels_path = gpu_dir / "labels.pkl"
            with open(labels_path, 'wb') as f:
                pickle.dump(labels_to_save, f)
            
            logger.info(f"Saved {len(all_synthetic_images)} validation images to {gpu_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to save validation images: {e}")
            import traceback
            logger.warning(traceback.format_exc())

    def _move_metrics_models_to_cpu(self):
        """Move validation metric models to CPU."""
        if hasattr(self.metrics_runner, 'similarity'):
            if hasattr(self.metrics_runner.similarity, 'fid_model') and self.metrics_runner.similarity.fid_model is not None:
                self.metrics_runner.similarity.fid_model.to('cpu')
            if hasattr(self.metrics_runner.similarity, 'biovil_model') and self.metrics_runner.similarity.biovil_model is not None:
                self.metrics_runner.similarity.biovil_model.to('cpu')
        if hasattr(self.metrics_runner, 'text_alignment'):
            if hasattr(self.metrics_runner.text_alignment, 'disease_model') and self.metrics_runner.text_alignment.disease_model is not None:
                self.metrics_runner.text_alignment.disease_model.to('cpu')
            if hasattr(self.metrics_runner.text_alignment, 'sex_model') and self.metrics_runner.text_alignment.sex_model is not None:
                self.metrics_runner.text_alignment.sex_model.to('cpu')
            if hasattr(self.metrics_runner.text_alignment, 'race_model') and self.metrics_runner.text_alignment.race_model is not None:
                self.metrics_runner.text_alignment.race_model.to('cpu')
            if hasattr(self.metrics_runner.text_alignment, 'age_model') and self.metrics_runner.text_alignment.age_model is not None:
                self.metrics_runner.text_alignment.age_model.to('cpu')



    # --- ADD: per-batch update ---------------------------------------------------------

    def _compute_metrics_batched(self, all_synthetic_images, all_real_images, all_labels, global_step):
        """
        Compute ALL metrics on loaded images in batches (main process only).
        No distributed operations - everything happens on one GPU.
        """
        import numpy as np
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        metrics = {}
        device = self.accelerator.device
        num_images = len(all_synthetic_images)
        batch_size = self.args.get("validation_metrics_batch_size", 8)
        
        logger.info(f"Computing metrics on {num_images} images (batch_size={batch_size})")
        
        # Convert to tensors and move to device
        synth_tensor = torch.stack(all_synthetic_images).to(device)
        real_tensor = torch.stack(all_real_images).to(device) if all_real_images else None
        
        # Diagnostic: Log shapes and value ranges before FID computation
        logger.info(f"Image shapes before FID:")
        logger.info(f"  Synthetic: {synth_tensor.shape}, dtype: {synth_tensor.dtype}")
        logger.info(f"  Real: {real_tensor.shape if real_tensor is not None else None}, dtype: {real_tensor.dtype if real_tensor is not None else None}")
        logger.info(f"Image value ranges before FID:")
        logger.info(f"  Synthetic: min={synth_tensor.min():.4f}, max={synth_tensor.max():.4f}, mean={synth_tensor.mean():.4f}, std={synth_tensor.std():.4f}")
        if real_tensor is not None:
            logger.info(f"  Real: min={real_tensor.min():.4f}, max={real_tensor.max():.4f}, mean={real_tensor.mean():.4f}, std={real_tensor.std():.4f}")
        
        # Check for shape mismatches
        if real_tensor is not None:
            if synth_tensor.shape != real_tensor.shape:
                logger.warning(f"⚠️ SHAPE MISMATCH: Synthetic {synth_tensor.shape} vs Real {real_tensor.shape}")
            if synth_tensor.shape[1] != real_tensor.shape[1]:
                logger.warning(f"⚠️ CHANNEL MISMATCH: Synthetic has {synth_tensor.shape[1]} channels, Real has {real_tensor.shape[1]} channels")
            if synth_tensor.shape[2:] != real_tensor.shape[2:]:
                logger.warning(f"⚠️ SPATIAL DIMENSION MISMATCH: Synthetic {synth_tensor.shape[2:]} vs Real {real_tensor.shape[2:]}")
        
        # Extract labels
        disease_labels = all_labels.get("disease")
        sex_labels = all_labels.get("sex") 
        race_labels = all_labels.get("race")
        age_labels = all_labels.get("age")
        
        # Move labels to device if they exist
        if disease_labels is not None:
            disease_labels = disease_labels.to(device)
        if sex_labels is not None:
            sex_labels = sex_labels.to(device)
        if race_labels is not None:
            race_labels = race_labels.to(device)
        if age_labels is not None:
            age_labels = age_labels.to(device)
        
        # ================================================================
        # ACCUMULATORS
        # ================================================================
        # Similarity metrics accumulators (BioViL and MS-SSIM need per-batch averaging)
        biovil_sims = []
        msssim_vals = []
        
        # ================================================================
        # BATCH PROCESSING FOR BIOVIL AND MS-SSIM
        # FID will be computed separately using compute_fid() which handles batching internally
        # Text alignment metrics will be computed using methods from validation_metrics.py
        # ================================================================
        logger.info("Processing batches for BioViL and MS-SSIM computation...")
        
        for i in range(0, num_images, batch_size):
            end_i = min(i + batch_size, num_images)
            batch_synth = synth_tensor[i:end_i]
            batch_real = real_tensor[i:end_i] if real_tensor is not None else None
            
            # ------------------------------------------------------------
            # SIMILARITY METRICS (BioViL, MS-SSIM)
            # FID is computed separately below using compute_fid()
            # ------------------------------------------------------------
            if hasattr(self.metrics_runner, 'similarity') and batch_real is not None:
                try:
                    # BioViL similarity
                    biovil = self.metrics_runner.similarity.compute_biovil_similarity(
                        batch_real, batch_synth
                    )
                    if not np.isnan(biovil):
                        biovil_sims.append(biovil)
                except Exception as e:
                    logger.warning(f"BioViL error: {e}")
                
                try:
                    # MS-SSIM
                    msssim = self.metrics_runner.similarity.compute_ms_ssim(
                        batch_real, batch_synth
                    )
                    if not np.isnan(msssim):
                        msssim_vals.append(msssim)
                except Exception as e:
                    logger.warning(f"MS-SSIM error: {e}")
        
        logger.info("Batch processing complete, aggregating metrics...")
        
        # ================================================================
        # AGGREGATE SIMILARITY METRICS
        # ================================================================
        
        # FID (Inception v3) - use enhanced method from validation_metrics.py (handles batching internally)
        if hasattr(self.metrics_runner, 'similarity') and real_tensor is not None:
            try:
                logger.info("Computing FID (Inception v3) using enhanced method from validation_metrics.py...")
                metrics["val/fid"] = self.metrics_runner.similarity.compute_fid(
                    real_tensor, synth_tensor, batch_size=batch_size
                )
                logger.info(f"FID (Inception v3): {metrics['val/fid']:.4f}")
            except Exception as e:
                logger.warning(f"FID (Inception v3) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # FID (RadImageNet ResNet50) - medical image specific FID
        if hasattr(self.metrics_runner, 'similarity') and real_tensor is not None:
            try:
                logger.info("Computing FID (RadImageNet ResNet50) using enhanced method from validation_metrics.py...")
                metrics["val/fid_radimagenet"] = self.metrics_runner.similarity.compute_fid_radimagenet(
                    real_tensor, synth_tensor, batch_size=batch_size
                )
                logger.info(f"FID (RadImageNet ResNet50): {metrics['val/fid_radimagenet']:.4f}")
            except Exception as e:
                logger.warning(f"FID (RadImageNet ResNet50) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # ================================================================
        # SUBGROUP METRICS (Level 1 & 2: per sex, per ethnicity, per age group, and intersectional)
        # ================================================================
        compute_subgroup_metrics = self.args.get("compute_subgroup_metrics", False)
        
        if not compute_subgroup_metrics:
            logger.info("Subgroup metrics computation is disabled (set compute_subgroup_metrics=true in config to enable)")
        
        if compute_subgroup_metrics and hasattr(self.metrics_runner, 'similarity') and real_tensor is not None:
            try:
                logger.info("Computing FID per subgroup (Level 1: per sex, per ethnicity, per age group)...")
                subgroup_fids = self.metrics_runner.similarity.compute_fid_per_subgroup(
                    real_tensor, synth_tensor,
                    sex_labels=sex_labels,
                    race_labels=race_labels,
                    age_labels=age_labels,
                    batch_size=batch_size,
                    use_radimagenet=False  # Use Inception v3 for Level 1
                )
                
                # Add subgroup FIDs to metrics dictionary
                for group_type, group_results in subgroup_fids.items():
                    for subgroup_name, fid_value in group_results.items():
                        metric_key = f"val/fid_subgroup_{group_type}_{subgroup_name}"
                        metrics[metric_key] = float(fid_value)
                        logger.info(f"  FID ({group_type}: {subgroup_name}): {fid_value:.4f}")
                
                logger.info(f"✓ Level 1 subgroup FID computation complete ({sum(len(v) for v in subgroup_fids.values())} subgroups)")
            except Exception as e:
                logger.warning(f"Subgroup FID (Level 1) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # FID (RadImageNet) per subgroup (Level 1)
        if compute_subgroup_metrics and hasattr(self.metrics_runner, 'similarity') and real_tensor is not None:
            try:
                logger.info("Computing FID (RadImageNet) per subgroup (Level 1)...")
                subgroup_fids_radimagenet = self.metrics_runner.similarity.compute_fid_per_subgroup(
                    real_tensor, synth_tensor,
                    sex_labels=sex_labels,
                    race_labels=race_labels,
                    age_labels=age_labels,
                    batch_size=batch_size,
                    use_radimagenet=True  # Use RadImageNet for Level 1
                )
                
                # Add subgroup FIDs to metrics dictionary
                for group_type, group_results in subgroup_fids_radimagenet.items():
                    for subgroup_name, fid_value in group_results.items():
                        metric_key = f"val/fid_radimagenet_subgroup_{group_type}_{subgroup_name}"
                        metrics[metric_key] = float(fid_value)
                        logger.info(f"  FID RadImageNet ({group_type}: {subgroup_name}): {fid_value:.4f}")
                
                logger.info(f"✓ Level 1 subgroup FID (RadImageNet) computation complete ({sum(len(v) for v in subgroup_fids_radimagenet.values())} subgroups)")
            except Exception as e:
                logger.warning(f"Subgroup FID RadImageNet (Level 1) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # ================================================================
        # INTERSECTIONAL SUBGROUP FID METRICS (Level 2: age group x ethnicity x sex)
        # ================================================================
        if compute_subgroup_metrics and hasattr(self.metrics_runner, 'similarity') and real_tensor is not None:
            try:
                logger.info("Computing FID per intersectional subgroup (Level 2: age group x ethnicity x sex)...")
                intersectional_fids = self.metrics_runner.similarity.compute_fid_per_intersectional_subgroup(
                    real_tensor, synth_tensor,
                    sex_labels=sex_labels,
                    race_labels=race_labels,
                    age_labels=age_labels,
                    batch_size=batch_size,
                    use_radimagenet=False  # Use Inception v3 for Level 2
                )
                
                # Add intersectional FIDs to metrics dictionary
                for subgroup_name, fid_value in intersectional_fids.items():
                    metric_key = f"val/fid_intersectional_{subgroup_name}"
                    metrics[metric_key] = float(fid_value)
                    logger.info(f"  FID (intersectional: {subgroup_name}): {fid_value:.4f}")
                
                logger.info(f"✓ Level 2 intersectional subgroup FID computation complete ({len(intersectional_fids)} subgroups)")
            except Exception as e:
                logger.warning(f"Intersectional subgroup FID (Level 2) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # FID (RadImageNet) per intersectional subgroup (Level 2)
        if compute_subgroup_metrics and hasattr(self.metrics_runner, 'similarity') and real_tensor is not None:
            try:
                logger.info("Computing FID (RadImageNet) per intersectional subgroup (Level 2)...")
                intersectional_fids_radimagenet = self.metrics_runner.similarity.compute_fid_per_intersectional_subgroup(
                    real_tensor, synth_tensor,
                    sex_labels=sex_labels,
                    race_labels=race_labels,
                    age_labels=age_labels,
                    batch_size=batch_size,
                    use_radimagenet=True  # Use RadImageNet for Level 2
                )
                
                # Add intersectional FIDs to metrics dictionary
                for subgroup_name, fid_value in intersectional_fids_radimagenet.items():
                    metric_key = f"val/fid_radimagenet_intersectional_{subgroup_name}"
                    metrics[metric_key] = float(fid_value)
                    logger.info(f"  FID RadImageNet (intersectional: {subgroup_name}): {fid_value:.4f}")
                
                logger.info(f"✓ Level 2 intersectional subgroup FID (RadImageNet) computation complete ({len(intersectional_fids_radimagenet)} subgroups)")
            except Exception as e:
                logger.warning(f"Intersectional subgroup FID RadImageNet (Level 2) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # BioViL Similarity
        if biovil_sims:
            metrics["val/biovil_similarity"] = float(np.mean(biovil_sims))
            logger.info(f"BioViL Similarity: {metrics['val/biovil_similarity']:.4f}")
        
        # MS-SSIM
        if msssim_vals:
            metrics["val/ms_ssim"] = float(np.mean(msssim_vals))
            logger.info(f"MS-SSIM: {metrics['val/ms_ssim']:.4f}")
        
        # ================================================================
        # SUBGROUP MS-SSIM METRICS (Level 1: per sex, per ethnicity, per age group)
        # ================================================================
        if compute_subgroup_metrics and hasattr(self.metrics_runner, 'similarity') and real_tensor is not None:
            try:
                logger.info("Computing MS-SSIM per subgroup (Level 1: per sex, per ethnicity, per age group)...")
                subgroup_ms_ssim = self.metrics_runner.similarity.compute_ms_ssim_per_subgroup(
                    real_tensor, synth_tensor,
                    sex_labels=sex_labels,
                    race_labels=race_labels,
                    age_labels=age_labels
                )
                
                # Add subgroup MS-SSIM to metrics dictionary
                for group_type, group_results in subgroup_ms_ssim.items():
                    for subgroup_name, ms_ssim_value in group_results.items():
                        metric_key = f"val/ms_ssim_subgroup_{group_type}_{subgroup_name}"
                        metrics[metric_key] = float(ms_ssim_value)
                        logger.info(f"  MS-SSIM ({group_type}: {subgroup_name}): {ms_ssim_value:.4f}")
                
                logger.info(f"✓ Level 1 subgroup MS-SSIM computation complete ({sum(len(v) for v in subgroup_ms_ssim.values())} subgroups)")
            except Exception as e:
                logger.warning(f"Subgroup MS-SSIM (Level 1) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # ================================================================
        # INTERSECTIONAL SUBGROUP MS-SSIM METRICS (Level 2: age group x ethnicity x sex)
        # ================================================================
        if compute_subgroup_metrics and hasattr(self.metrics_runner, 'similarity') and real_tensor is not None:
            try:
                logger.info("Computing MS-SSIM per intersectional subgroup (Level 2: age group x ethnicity x sex)...")
                intersectional_ms_ssim = self.metrics_runner.similarity.compute_ms_ssim_per_intersectional_subgroup(
                    real_tensor, synth_tensor,
                    sex_labels=sex_labels,
                    race_labels=race_labels,
                    age_labels=age_labels
                )
                
                # Add intersectional MS-SSIM to metrics dictionary
                for subgroup_name, ms_ssim_value in intersectional_ms_ssim.items():
                    metric_key = f"val/ms_ssim_intersectional_{subgroup_name}"
                    metrics[metric_key] = float(ms_ssim_value)
                    logger.info(f"  MS-SSIM (intersectional: {subgroup_name}): {ms_ssim_value:.4f}")
                
                logger.info(f"✓ Level 2 intersectional subgroup MS-SSIM computation complete ({len(intersectional_ms_ssim)} subgroups)")
            except Exception as e:
                logger.warning(f"Intersectional subgroup MS-SSIM (Level 2) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # ================================================================
        # SUBGROUP BIOVIL SIMILARITY METRICS (Level 1: per sex, per ethnicity, per age group)
        # ================================================================
        if compute_subgroup_metrics and hasattr(self.metrics_runner, 'similarity') and real_tensor is not None:
            try:
                logger.info("Computing BioViL similarity per subgroup (Level 1: per sex, per ethnicity, per age group)...")
                subgroup_biovil = self.metrics_runner.similarity.compute_biovil_similarity_per_subgroup(
                    real_tensor, synth_tensor,
                    sex_labels=sex_labels,
                    race_labels=race_labels,
                    age_labels=age_labels,
                    batch_size=batch_size
                )
                
                # Add subgroup BioViL similarity to metrics dictionary
                for group_type, group_results in subgroup_biovil.items():
                    for subgroup_name, biovil_value in group_results.items():
                        metric_key = f"val/biovil_similarity_subgroup_{group_type}_{subgroup_name}"
                        metrics[metric_key] = float(biovil_value)
                        logger.info(f"  BioViL Similarity ({group_type}: {subgroup_name}): {biovil_value:.4f}")
                
                logger.info(f"✓ Level 1 subgroup BioViL similarity computation complete ({sum(len(v) for v in subgroup_biovil.values())} subgroups)")
            except Exception as e:
                logger.warning(f"Subgroup BioViL similarity (Level 1) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # ================================================================
        # INTERSECTIONAL SUBGROUP BIOVIL SIMILARITY METRICS (Level 2: age group x ethnicity x sex)
        # ================================================================
        if compute_subgroup_metrics and hasattr(self.metrics_runner, 'similarity') and real_tensor is not None:
            try:
                logger.info("Computing BioViL similarity per intersectional subgroup (Level 2: age group x ethnicity x sex)...")
                intersectional_biovil = self.metrics_runner.similarity.compute_biovil_similarity_per_intersectional_subgroup(
                    real_tensor, synth_tensor,
                    sex_labels=sex_labels,
                    race_labels=race_labels,
                    age_labels=age_labels,
                    batch_size=batch_size
                )
                
                # Add intersectional BioViL similarity to metrics dictionary
                for subgroup_name, biovil_value in intersectional_biovil.items():
                    metric_key = f"val/biovil_similarity_intersectional_{subgroup_name}"
                    metrics[metric_key] = float(biovil_value)
                    logger.info(f"  BioViL Similarity (intersectional: {subgroup_name}): {biovil_value:.4f}")
                
                logger.info(f"✓ Level 2 intersectional subgroup BioViL similarity computation complete ({len(intersectional_biovil)} subgroups)")
            except Exception as e:
                logger.warning(f"Intersectional subgroup BioViL similarity (Level 2) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # ================================================================
        # AGGREGATE TEXT ALIGNMENT METRICS
        # Use enhanced methods from validation_metrics.py
        # ================================================================
        
        # Use the enhanced methods from validation_metrics.py for text alignment metrics
        # These methods handle all the filtering and mapping logic internally
        if hasattr(self.metrics_runner, 'text_alignment'):
            ta = self.metrics_runner.text_alignment
            
            # Disease AUROC - use enhanced method with batching
            if disease_labels is not None and synth_tensor is not None:
                try:
                    logger.info("Computing disease AUROC using enhanced method from validation_metrics.py (with batching)...")
                    disease_aurocs = ta.compute_disease_auroc(synth_tensor, disease_labels, batch_size=batch_size)
                    for disease_name, auroc in disease_aurocs.items():
                        if disease_name == "mean_auroc":
                            metrics["val/mean_auroc"] = float(auroc) if not np.isnan(auroc) else np.nan
                        else:
                            metrics[f"val/{disease_name}"] = float(auroc) if not np.isnan(auroc) else np.nan
                    logger.info(f"Disease AUROC computation complete. Mean AUROC: {disease_aurocs.get('mean_auroc', np.nan):.4f}")
                except Exception as e:
                    import traceback
                    logger.error(f"Disease AUROC error: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Sex Accuracy - use enhanced method with batching
            if sex_labels is not None and synth_tensor is not None:
                try:
                    logger.info("Computing sex accuracy using enhanced method from validation_metrics.py (with batching)...")
                    sex_acc = ta.compute_sex_accuracy(synth_tensor, sex_labels, batch_size=batch_size)
                    if not np.isnan(sex_acc):
                        metrics["val/sex_accuracy"] = float(sex_acc)
                        logger.info(f"Sex Accuracy: {sex_acc:.4f}")
                    else:
                        logger.warning("Sex accuracy computation returned NaN")
                except Exception as e:
                    logger.warning(f"Sex accuracy error: {e}")
            
            # Race Accuracy - use enhanced method with batching
            if race_labels is not None and synth_tensor is not None:
                try:
                    logger.info("Computing race accuracy using enhanced method from validation_metrics.py (with batching)...")
                    race_acc = ta.compute_race_accuracy(synth_tensor, race_labels, batch_size=batch_size)
                    if not np.isnan(race_acc):
                        metrics["val/race_accuracy"] = float(race_acc)
                        logger.info(f"Race Accuracy: {race_acc:.4f}")
                    else:
                        logger.warning("Race accuracy computation returned NaN")
                except Exception as e:
                    logger.warning(f"Race accuracy error: {e}")
            
            # Age RMSE - use enhanced method with batching
            if age_labels is not None and synth_tensor is not None:
                try:
                    logger.info("Computing age RMSE using enhanced method from validation_metrics.py (with batching)...")
                    age_rmse = ta.compute_age_rmse(synth_tensor, age_labels, batch_size=batch_size)
                    metrics["val/age_rmse"] = float(age_rmse)
                    logger.info(f"Age RMSE: {age_rmse:.4f}")
                except Exception as e:
                    logger.warning(f"Age RMSE error: {e}")
        
        logger.info("="*60)
        logger.info(f"Total metrics computed: {len(metrics)}")
        logger.info("="*60)
        
        return metrics

    def run_validation(self, global_step: int) -> Dict:
        """
        Hybrid validation approach:
        - Distributed: Image generation (parallelized across all GPUs) - SKIPPED if load_images_from_dir is set
        - Centralized: Metrics computation (only rank 0, no collective ops)
        
        This avoids all distributed synchronization issues while keeping speed benefits.
        
        If load_images_from_dir is set, skips generation and loads pre-generated images instead.
        """
        if self.accelerator.is_main_process:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running validation at step {global_step}")
            if self.load_images_from_dir:
                logger.info(f"Mode: Loading pre-generated images from {self.load_images_from_dir}")
            else:
                logger.info(f"Mode: Generating images")
            logger.info(f"{'='*60}")

        # Initialize metrics runner ONLY on rank 0
        if self.accelerator.is_main_process:
            self._ensure_metrics_runner()
            if self.metrics_runner is not None:
                self._move_metrics_models_to_gpu()

        # Only set models to eval if we're generating (not needed for loading)
        if not self.load_images_from_dir:
            self.unet.eval()
            self.text_encoder.eval()
            if self.hcn is not None:
                self.hcn.eval()

        try:
            # ==============================================================
            # PHASE 1: GENERATE OR SKIP (depending on load_images_from_dir)
            # ==============================================================
            if self.load_images_from_dir:
                # Skip generation - will load from directory instead
                if self.accelerator.is_main_process:
                    logger.info(f"Rank 0: Skipping generation, will load images from {self.load_images_from_dir}")
                else:
                    logger.info(f"[Rank {self.accelerator.process_index}] Skipping generation (loading mode)")
            else:
                # Normal generation flow
                unwrapped_unet = self.accelerator.unwrap_model(self.unet).eval()
                unwrapped_text_encoder = self.accelerator.unwrap_model(self.text_encoder).eval()
                unwrapped_vae = self.accelerator.unwrap_model(self.vae).eval().to(dtype=torch.float32)

                logger.info(f"[Rank {self.accelerator.process_index}] Starting distributed generation")
                
                all_synthetic_images, all_real_images, all_labels = self._generate_validation_images(
                    unwrapped_unet, unwrapped_text_encoder, unwrapped_vae, global_step
                )
                
                logger.info(f"[Rank {self.accelerator.process_index}] Generated {len(all_synthetic_images)} images")

                # ==============================================================
                # PHASE 2: SAVE TO DISK (all ranks save their portion)
                # ==============================================================
                self._save_validation_images(
                    all_synthetic_images, all_real_images, all_labels, global_step
                )
                
                logger.info(f"[Rank {self.accelerator.process_index}] Saved images to disk")

            # ==============================================================
            # PHASE 2.5: VERIFY ALL GPUS COMPLETED (rank 0 checks)
            # ==============================================================
            if self.accelerator.is_main_process:
                # Wait a bit for all GPUs to finish saving
                import time
                time.sleep(2)
                
                # Check if all expected GPU directories exist
                validation_images_dir = Path(self.args["output_dir"]) / "validation_images" / f"step_{global_step}"
                expected_gpu_dirs = [validation_images_dir / f"gpu_{i}" for i in range(self.accelerator.num_processes)]
                existing_gpu_dirs = [d for d in expected_gpu_dirs if d.exists()]
                
                if len(existing_gpu_dirs) < self.accelerator.num_processes:
                    missing_gpus = [i for i in range(self.accelerator.num_processes) 
                                  if not expected_gpu_dirs[i].exists()]
                    logger.warning(f"Only {len(existing_gpu_dirs)}/{self.accelerator.num_processes} GPUs completed generation!")
                    logger.warning(f"Missing GPUs: {missing_gpus}")
                    logger.warning(f"This may indicate GPUs crashed or hung during generation.")
                    logger.warning(f"Proceeding with images from completed GPUs only.")

            # ==============================================================
            # PHASE 3: CENTRALIZED METRICS (rank 0 only, no collectives)
            # ==============================================================
            metrics = {}
            num_images = 0
            
            if self.accelerator.is_main_process:
                logger.info("Rank 0: Loading images from disk for metrics computation")
                
                # Load images - _load_validation_images_from_disk handles both cases correctly
                # It constructs the step-specific path whether using load_images_from_dir or standard location
                loaded_synth, loaded_real, loaded_labels = self._load_validation_images_from_disk(global_step)
                
                if loaded_synth and len(loaded_synth) > 0:
                    num_images = len(loaded_synth)
                    logger.info(f"Rank 0: Loaded {num_images} total images")
                    
                    # Compute metrics on loaded images (batched to avoid OOM)
                    metrics = self._compute_metrics_batched(
                        loaded_synth, loaded_real, loaded_labels, global_step
                    )
                    
                    logger.info("Rank 0: Metrics computation complete")
                    logger.info("="*60)
                    for key, value in metrics.items():
                        logger.info(f"{key}: {value:.4f}")
                    logger.info("="*60)
                else:
                    logger.warning("Rank 0: No images found, skipping metrics")
            else:
                logger.info(f"[Rank {self.accelerator.process_index}] Waiting for rank 0 to finish metrics")

            # ==============================================================
            # PHASE 4: FINAL SYNC & CLEANUP
            # ==============================================================
            logger.info(f"[Rank {self.accelerator.process_index}] Validation complete")

            if self.accelerator.is_main_process and self.metrics_runner is not None:
                self._move_metrics_models_to_cpu()

            torch.cuda.empty_cache()
            
            # Return metrics dict with num_images added for manifest tracking
            # Store num_images in metrics dict so it's accessible to callers
            if self.accelerator.is_main_process:
                metrics["_num_images"] = num_images
            
            return metrics

        except Exception as e:
            logger.error(f"[Rank {self.accelerator.process_index}] Validation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
def main():
    """Main validation monitoring loop."""
    parser = argparse.ArgumentParser(description="Validation monitoring script")
    parser.add_argument("--config_file", type=str, required=True, help="Path to training config file")
    parser.add_argument("--check_interval", type=int, default=60, help="Check for new checkpoints every N seconds")
    parser.add_argument("--manifest_file", type=str, default="validation_manifest.json", help="Manifest file to track validated checkpoints")
    parser.add_argument("--training_run_id", type=str, default=None, help="Training run ID to resume logging (for wandb)")
    parser.add_argument("--training_run_name", type=str, default=None, help="Training run name to resume logging")
    parser.add_argument("--load_images_from_dir", action="store_true", help="Load pre-generated images from validation_images directory instead of generating them. Images should be in output_dir/validation_images/step_X/ subdirectories.")
    args = parser.parse_args()

    # Load training config
    config = load_config(args.config_file)
    if config is None:
        raise ValueError(f"Failed to load config from {args.config_file}")
    
    # Add load_images_from_dir flag to config if provided
    # If load_images_from_dir is set, we monitor step directories instead of checkpoints
    load_images_from_dir = args.load_images_from_dir
    if load_images_from_dir:
        config["load_images_from_dir"] = True
        # Use print before Accelerator is initialized (logger requires Accelerator)
        print(f"Image loading mode: Will monitor and load pre-generated images from validation_images directory")

    # Initialize accelerator
    # Ensure GPU is used if available (validation requires GPU for generation, unless loading from directory)
    accelerator = Accelerator(
        mixed_precision=config.get("mixed_precision", "bf16"),
        log_with=config.get("report_to", "wandb"),
    )
    
    # Now we can use logger after Accelerator is initialized
    if load_images_from_dir:
        logger.info(f"Image loading mode: Will monitor and load pre-generated images from validation_images directory")
    
    # Verify GPU is available (only required if generating images, not if loading from directory)
    # load_images_from_dir is already set above, but get from config to be safe
    if not load_images_from_dir:
        load_images_from_dir = config.get("load_images_from_dir", False)
    if not load_images_from_dir:
        if not torch.cuda.is_available():
            logger.error("CUDA is not available! Validation requires GPU for image generation.")
            logger.error("Please run this script in a SLURM job with GPU allocation or ensure CUDA is available.")
            logger.error("Alternatively, use --load_images_from_dir to load pre-generated images.")
            raise RuntimeError("CUDA not available - validation requires GPU for generation")
        
        if accelerator.device.type == "cpu":
            logger.error(f"Accelerator initialized on CPU (device: {accelerator.device})")
            logger.error("This script requires GPU for image generation. Please check:")
            logger.error("  1. GPU is allocated (e.g., via SLURM: --gres=gpu:1)")
            logger.error("  2. CUDA_VISIBLE_DEVICES is set correctly")
            logger.error("  3. accelerate launch is configured for GPU")
            logger.error("Alternatively, use --load_images_from_dir to load pre-generated images.")
            raise RuntimeError("Accelerator on CPU - GPU required for validation")
        
        logger.info(f"✓ Accelerator initialized on {accelerator.device} (GPU available)")
    else:
        logger.info(f"✓ Accelerator initialized on {accelerator.device}")
        logger.info("  Note: GPU not required since loading pre-generated images")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Try to load training run info to resume same run
    training_run_info_file = Path(config["output_dir"]) / "training_run_info.json"
    training_run_id = args.training_run_id
    training_run_name = args.training_run_name
    training_project = None

    if training_run_info_file.exists():
        try:
            with open(training_run_info_file, 'r') as f:
                run_info = json.load(f)
                if training_run_id is None:
                    training_run_id = run_info.get("run_id")
                if training_run_name is None:
                    training_run_name = run_info.get("run_name")
                training_project = run_info.get("project")  # Get project from training run info
                logger.info(f"Found training run info: run_id={training_run_id}, run_name={training_run_name}, project={training_project}")
        except Exception as e:
            logger.warning(f"Could not load training run info: {e}")

    # Initialize trackers for logging
    tracker_init_kwargs = {}
    if config.get("report_to") == "wandb":
        import wandb
        # Use project from training run info if available, otherwise use logging_dir or default
        wandb_project = training_project if training_project else config.get("logging_dir", "roentgen-validation")
        
        # Prepare wandb init kwargs
        wandb_init_kwargs = {
            "project": wandb_project,
            "name": training_run_name if training_run_name else f"validation-{config['output_dir'].split('/')[-1]}",
            "tags": ["validation", "monitoring"],
        }

        # If we have a training run ID, try to resume it
        if training_run_id:
            try:
                logger.info(f"Attempting to resume wandb run: {training_run_id} in project: {wandb_project}")
                # Resume the run using wandb directly - MUST use the correct project from training
                wandb.init(
                    id=training_run_id,
                    resume="allow",
                    project=wandb_project,  # Use the project from training run info
                    name=training_run_name,  # Keep the same run name
                    tags=wandb_init_kwargs.get("tags", []),
                )
                logger.info(f"✓ Successfully resumed wandb run {training_run_id} in project {wandb_project}")
                
                # Define validation metrics to allow out-of-order logging
                # This allows us to log validation metrics at any step, even if training has progressed further
                # Using summary="last" allows metrics to be updated at any step
                validation_metric_names = [
                    "val/fid", "val/fid_radimagenet", "val/biovil_similarity", "val/ms_ssim", "val/mean_auroc",
                    "val/sex_accuracy", "val/race_accuracy", "val/age_rmse",
                    "val/Atelectasis", "val/Consolidation", "val/Infiltration", 
                    "val/Pneumothorax", "val/Edema", "val/Emphysema", "val/Fibrosis",
                    "val/Effusion", "val/Pneumonia", "val/Pleural_Thickening",
                    "val/Cardiomegaly", "val/Nodule", "val/Mass", "val/Hernia",
                    "val/Lung Lesion", "val/Fracture", "val/Lung Opacity",
                    "val/Enlarged Cardiomediastinum"
                ]
                for metric_name in validation_metric_names:
                    # Define metric to allow out-of-order logging
                    # By not specifying step constraints, wandb allows logging at any step
                    # summary="last" keeps the most recent value for summary
                    try:
                        wandb.define_metric(metric_name, step_metric="global_step", summary="last")
                    except:
                        # If step_metric doesn't work, try without it
                        try:
                            wandb.define_metric(metric_name, summary="last")
                        except:
                            # Metric might already be defined, that's OK
                            pass
                logger.info("✓ Defined validation metrics for out-of-order logging")
                
                # Initialize accelerator trackers - when wandb is already initialized,
                # accelerator will detect and use the existing run
                # Pass minimal kwargs to avoid re-initialization
                if wandb.run is not None:
                    # Wandb is already initialized, accelerator will use it
                    tracker_init_kwargs["wandb"] = {}  # Empty dict - accelerator will detect existing run
                else:
                    # Fallback: provide init kwargs
                    tracker_init_kwargs["wandb"] = {
                        "id": training_run_id,
                        "resume": "allow",
                        "project": wandb_project,
                    }
                accelerator.init_trackers(
                    project_name=wandb_project,
                    config=config,
                    init_kwargs=tracker_init_kwargs,
                )
                logger.info("✓ Accelerator trackers initialized with resumed wandb run")
            except Exception as e:
                logger.warning(f"Could not resume training run: {e}")
                logger.info("Creating new wandb run for validation")
                tracker_init_kwargs["wandb"] = wandb_init_kwargs
                accelerator.init_trackers(
                    project_name=wandb_init_kwargs["project"],
                    config=config,
                    init_kwargs=tracker_init_kwargs,
                )
                # Define validation metrics for new run too
                if wandb.run is not None:
                    validation_metric_names = [
                        "val/fid", "val/biovil_similarity", "val/ms_ssim", "val/mean_auroc",
                        "val/sex_accuracy", "val/race_accuracy", "val/age_rmse",
                        "val/Atelectasis", "val/Consolidation", "val/Infiltration", 
                        "val/Pneumothorax", "val/Edema", "val/Emphysema", "val/Fibrosis",
                        "val/Effusion", "val/Pneumonia", "val/Pleural_Thickening",
                        "val/Cardiomegaly", "val/Nodule", "val/Mass", "val/Hernia",
                        "val/Lung Lesion", "val/Fracture", "val/Lung Opacity",
                        "val/Enlarged Cardiomediastinum"
                    ]
                    for metric_name in validation_metric_names:
                        wandb.define_metric(metric_name, summary="last")
        else:
            tracker_init_kwargs["wandb"] = wandb_init_kwargs
            accelerator.init_trackers(
                project_name=wandb_init_kwargs["project"],
                config=config,
                init_kwargs=tracker_init_kwargs,
            )
            # Define validation metrics for new run
            if wandb.run is not None:
                validation_metric_names = [
                    "val/fid", "val/fid_radimagenet", "val/biovil_similarity", "val/ms_ssim", "val/mean_auroc",
                    "val/sex_accuracy", "val/race_accuracy", "val/age_rmse",
                    "val/Atelectasis", "val/Consolidation", "val/Infiltration", 
                    "val/Pneumothorax", "val/Edema", "val/Emphysema", "val/Fibrosis",
                    "val/Effusion", "val/Pneumonia", "val/Pleural_Thickening",
                    "val/Cardiomegaly", "val/Nodule", "val/Mass", "val/Hernia",
                    "val/Lung Lesion", "val/Fracture", "val/Lung Opacity",
                    "val/Enlarged Cardiomediastinum"
                ]
                for metric_name in validation_metric_names:
                    wandb.define_metric(metric_name, summary="last")
    else:
        # For tensorboard or other trackers
        accelerator.init_trackers(
            project_name=config.get("logging_dir", "roentgen-validation"),
            config=config,
        )

    logger.info("✓ Logging trackers initialized")

    # Initialize runner
    runner = ValidationRunner(
        accelerator=accelerator,
        args=config,
        logger=logger,
    )
    
    # NOTE: DO NOT add wait_for_everyone() here - it causes hangs
    # Rank 0 initializes metrics runner which delays it
    # The monitoring loop will naturally synchronize when all ranks
    # call load_checkpoint() and then prepare()

    logger.info("="*60)
    logger.info("Validation monitoring started")
    logger.info(f"Output directory: {config['output_dir']}")
    logger.info(f"Check interval: {args.check_interval}s")
    logger.info("="*60)

    # Main monitoring loop
    try:
        # If loading from directory, monitor step directories instead of checkpoints
        if load_images_from_dir:
            # Initialize step image monitor
            output_dir = Path(config["output_dir"])
            images_base_dir = output_dir / "validation_images"
            
            if not images_base_dir.exists():
                logger.error(f"Validation images directory not found: {images_base_dir}")
                raise ValueError(f"Validation images directory not found: {images_base_dir}")
            
            monitor = StepImageMonitor(
                images_base_dir=str(images_base_dir),
                manifest_file=args.manifest_file,
            )
            
            logger.info("="*60)
            logger.info("Monitoring pre-generated images")
            logger.info(f"Image base directory: {images_base_dir}")
            logger.info(f"Manifest file: {args.manifest_file}")
            logger.info("="*60)
            
            # Monitoring loop for step directories
            while True:
                # Check for new step directories
                new_steps = monitor.get_new_steps()
                
                if new_steps:
                    logger.info(f"Found {len(new_steps)} new step directory(ies): {[f'step_{s[0]}' for s in new_steps]}")
                    
                    for step, step_dir in new_steps:
                        try:
                            logger.info(f"\n{'='*60}")
                            logger.info(f"Validating step directory: {step_dir.name} (step {step})")
                            logger.info(f"{'='*60}")
                            
                            # Mark as in progress EARLY - before starting validation
                            # Only main process updates manifest to avoid race conditions
                            if accelerator.is_main_process:
                                monitor.mark_in_progress(step)
                                logger.info(f"✓ Marked checkpoint as in progress in manifest")
                            
                            # Run validation (loads images from step directory)
                            metrics = runner.run_validation(step)
                            
                            # Log metrics to wandb/tensorboard
                            if metrics and accelerator.is_main_process:
                                try:
                                    # Extract num_images before filtering metrics for logging
                                    num_images = metrics.get("_num_images", 0)
                                    # Filter out _num_images from metrics before logging (it's just for tracking)
                                    metrics_for_logging = {k: v for k, v in metrics.items() if k != "_num_images"}
                                    
                                    # For wandb, use direct logging to handle out-of-order steps
                                    if config.get("report_to") == "wandb":
                                        import wandb
                                        # Ensure metrics are defined (fallback in case they weren't defined during init)
                                        if wandb.run is not None:
                                            for metric_name in metrics_for_logging.keys():
                                                try:
                                                    # Try to define metric if not already defined
                                                    # Use step_metric to allow independent step tracking
                                                    wandb.define_metric(metric_name, step_metric="global_step", summary="last")
                                                except:
                                                    try:
                                                        # Fallback: define without step_metric
                                                        wandb.define_metric(metric_name, summary="last")
                                                    except:
                                                        # Metric might already be defined, that's OK
                                                        pass
                                        # Check current wandb step to see if we can log at the requested step
                                        # wandb.run.step tracks the last logged step
                                        current_wandb_step = wandb.run.step if (wandb.run and hasattr(wandb.run, 'step')) else 0
                                        
                                        log_dict = dict(metrics_for_logging)
                                        log_dict["validation_step"] = step  # Always include checkpoint step as metric
                                        
                                        if step >= current_wandb_step:
                                            # Safe to log with step parameter - step is current or future
                                            wandb.log(log_dict, step=step)
                                        else:
                                            # Step is in the past - log without step to avoid data being ignored
                                            # The validation_step metric preserves which checkpoint this corresponds to
                                            wandb.log(log_dict)
                                            new_step = wandb.run.step if (wandb.run and hasattr(wandb.run, 'step')) else 'unknown'
                                            logger.info(f"  Note: Logged at wandb step {new_step}, checkpoint step {step} preserved in 'validation_step' metric")
                                    else:
                                        # For tensorboard or other trackers, normal logging
                                        accelerator.log(metrics_for_logging, step=step)
                                    
                                    logger.info(f"✓ Metrics logged to {config.get('report_to', 'tracker')} at step {step}")
                                except Exception as e:
                                    logger.warning(f"Failed to log metrics: {e}")
                                    import traceback
                                    logger.warning(traceback.format_exc())
                            
                            # Mark as validated - ONLY on main process to ensure metrics are saved correctly
                            # Non-main processes have empty metrics dict, so they would overwrite with {}
                            if accelerator.is_main_process:
                                num_images = metrics.get("_num_images", 0) if metrics else 0
                                # Remove _num_images from metrics before saving (it's just for tracking)
                                metrics_for_manifest = {k: v for k, v in metrics.items() if k != "_num_images"} if metrics else {}
                                monitor.mark_validated(step, metrics_for_manifest, success=True, num_images=num_images)
                            else:
                                # Non-main processes just wait - main process will update manifest
                                logger.info(f"[Rank {accelerator.process_index}] Skipping manifest update (main process handles it)")
                            
                            logger.info(f"✓ Step {step} validated successfully")
                            
                        except Exception as e:
                            logger.error(f"Failed to validate step {step}: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            # Mark as failed - ONLY on main process
                            if accelerator.is_main_process:
                                monitor.mark_validated(step, {}, success=False, num_images=0)
                else:
                    if accelerator.is_local_main_process:
                        logger.info(f"No new step directories found. Waiting {args.check_interval}s...")
                
                # Wait before next check
                time.sleep(args.check_interval)
        
        # Normal monitoring loop for checkpoint validation
        monitor = CheckpointMonitor(
            output_dir=config["output_dir"],
            manifest_file=args.manifest_file,
        )
        
        while True:
            # Check for new checkpoints
            new_checkpoints = monitor.get_new_checkpoints()

            if new_checkpoints:
                logger.info(f"Found {len(new_checkpoints)} new checkpoint(s): {[c.name for c in new_checkpoints]}")

                for checkpoint_dir in new_checkpoints:
                    try:
                        # Extract step number
                        step = int(checkpoint_dir.name.split("-")[1])

                        logger.info(f"\n{'='*60}")
                        logger.info(f"Validating checkpoint: {checkpoint_dir.name} (step {step})")
                        logger.info(f"{'='*60}")

                        # Mark as in progress EARLY - before starting validation
                        # Only main process updates manifest to avoid race conditions
                        if accelerator.is_main_process:
                            monitor.mark_in_progress(checkpoint_dir)
                            logger.info(f"✓ Marked checkpoint as in progress in manifest")
                        
                        # Ensure all processes see the manifest update before proceeding
                        # accelerator.wait_for_everyone()

                        # Load checkpoint
                        runner.load_checkpoint(checkpoint_dir)

                        # Run validation
                        metrics = runner.run_validation(step)

                        # Log metrics to wandb/tensorboard
                        if metrics and accelerator.is_main_process:
                            try:
                                # Extract num_images before filtering metrics for logging
                                num_images = metrics.get("_num_images", 0)
                                # Filter out _num_images from metrics before logging (it's just for tracking)
                                metrics_for_logging = {k: v for k, v in metrics.items() if k != "_num_images"}
                                
                                # For wandb, use direct logging to handle out-of-order steps
                                if config.get("report_to") == "wandb":
                                    import wandb
                                    # Ensure metrics are defined (fallback in case they weren't defined during init)
                                    if wandb.run is not None:
                                        for metric_name in metrics_for_logging.keys():
                                            try:
                                                # Try to define metric if not already defined
                                                # Use step_metric to allow independent step tracking
                                                wandb.define_metric(metric_name, step_metric="global_step", summary="last")
                                            except:
                                                try:
                                                    # Fallback: define without step_metric
                                                    wandb.define_metric(metric_name, summary="last")
                                                except:
                                                    # Metric might already be defined, that's OK
                                                    pass
                                    # Check current wandb step to see if we can log at the requested step
                                    # wandb.run.step tracks the last logged step
                                    current_wandb_step = wandb.run.step if (wandb.run and hasattr(wandb.run, 'step')) else 0
                                    
                                    log_dict = dict(metrics_for_logging)
                                    log_dict["validation_step"] = step  # Always include checkpoint step as metric
                                    
                                    if step >= current_wandb_step:
                                        # Safe to log with step parameter - step is current or future
                                        wandb.log(log_dict, step=step)
                                    else:
                                        # Step is in the past - log without step to avoid data being ignored
                                        # The validation_step metric preserves which checkpoint this corresponds to
                                        wandb.log(log_dict)
                                        new_step = wandb.run.step if (wandb.run and hasattr(wandb.run, 'step')) else 'unknown'
                                        logger.info(f"  Note: Logged at wandb step {new_step}, checkpoint step {step} preserved in 'validation_step' metric")
                                else:
                                    # For tensorboard or other trackers, normal logging
                                    accelerator.log(metrics_for_logging, step=step)
                                
                                logger.info(f"✓ Metrics logged to {config.get('report_to', 'tracker')} at step {step}")
                            except Exception as e:
                                logger.warning(f"Failed to log metrics: {e}")
                                import traceback
                                logger.warning(traceback.format_exc())

                        # Mark as validated - ONLY on main process to ensure metrics are saved correctly
                        # Non-main processes have empty metrics dict, so they would overwrite with {}
                        if accelerator.is_main_process:
                            num_images = metrics.get("_num_images", 0) if metrics else 0
                            # Remove _num_images from metrics before saving (it's just for tracking)
                            metrics_for_manifest = {k: v for k, v in metrics.items() if k != "_num_images"} if metrics else {}
                            monitor.mark_validated(checkpoint_dir, metrics_for_manifest, success=True, num_images=num_images)
                        else:
                            # Non-main processes just wait - main process will update manifest
                            logger.info(f"[Rank {accelerator.process_index}] Skipping manifest update (main process handles it)")

                        logger.info(f"✓ Checkpoint {checkpoint_dir.name} validated successfully")

                    except Exception as e:
                        logger.error(f"Failed to validate checkpoint {checkpoint_dir.name}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        # Mark as failed - ONLY on main process
                        if accelerator.is_main_process:
                            monitor.mark_validated(checkpoint_dir, {}, success=False, num_images=0)
            else:
                if accelerator.is_local_main_process:
                    logger.info(f"No new checkpoints found. Waiting {args.check_interval}s...")

            # Wait before next check
            time.sleep(args.check_interval)

    except KeyboardInterrupt:
        logger.info("\nValidation monitoring stopped by user")
    except Exception as e:
        logger.error(f"Validation monitoring failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # End tracking
        if accelerator.is_main_process:
            try:
                accelerator.end_training()
                logger.info("✓ Tracking ended")
            except Exception as e:
                logger.warning(f"Failed to end tracking: {e}")


if __name__ == "__main__":
    main()
