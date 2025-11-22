from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict
import yaml
import argparse


@dataclass
class BaseConfig:
    """All the configurable parameters for the training script."""

    # Path to pretrained model or model identifier from huggingface.co/models.
    pretrained_model_name_or_path: str = None
    # Revision of pretrained model identifier from huggingface.co/models.
    revision: str = None
    # Path to text_encoder model or model identifier from huggingface.co/models.
    pretrained_text_encoder_name_or_path: str = None
    # Hugging face authentication token
    use_auth_token: str = None
    embedding_method: str = "last_hidden_state"
    use_attention_mask: bool = False
    # Cache directory when loading models
    cache_dir: str = None
    # Whether to initialize the unet with random weights
    random_unet: bool = False
    enforce_tokenizer_max_sentence_length: int = None

    # Whether to load dataset from wds instead of image/prompt directories
    use_wds_dataset: bool = False
    url_root: str = None
    # A folder containing the training data of images.
    image_dir: str = None
    image_type: str = "pt"
    # A folder containing the training data of prompts.
    prompt_dir: str = None
    # A file filtering based on file names.
    data_filter_file: str = None
    data_filter_split_token: str = "\n"
    loss_weights_file: str = None
    loss_weights_split_token: str = "\n"

    inference_prompt_file: str = None
    inference_prompt_split_token: str = "\n"
    inference_prompt_number_per_prompt: int = 4
    inference_prompt_output_file: str = None

    # save only the weights that were modified instead of the entire pipeline
    save_only_modified_weights: bool = False
    do_not_save_weights: bool = False
    # The output directory where the model predictions and checkpoints will be written.
    output_dir: str = None

    # Seed number for reproducible training
    seed: int = 10
    # The resolution for input images, all the images in the train/validation
    # dataset will be resized to this
    resolution: int = 512
    # Whether to center crop images before resizing to resolution
    center_crop: bool = False

    # Whether to train the text encoder
    train_text_encoder: bool = False
    # Batch size (per device) for the training dataloader.
    train_batch_size: int = 4
    # For debugging purposes or quicker training, truncate the number
    # of training examples to this value if set
    max_train_samples: int = None
    num_train_epochs: int = 100
    # Total number of training steps to perform.
    # If provided, overrides num_train_epochs.
    max_train_steps: int = None
    # Number of updates steps to accumulate before performing a backward/update pass.
    gradient_accumulation_steps: int = 1
    # Whether or not to use gradient checkpointing to save memory
    # at the expense of slower backward pass.
    gradient_checkpointing: bool = False

    # Initial learning rate (after the potential warmup period) to use.
    learning_rate: float = 5e-06
    # Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
    scale_lr: bool = False
    # Choose between ["linear", "cosine", "cosine_with_restarts",
    # "polynomial", "constant", "constant_with_warmup"]
    lr_scheduler: str = "constant"
    # Number of steps for the warmup in the lr scheduler.
    lr_warmup_steps: int = 500
    # Whether or not to use 8-bit Adam from bitsandbytes.
    use_8bit_adam: bool = False
    # Whether to use EMA model for the unet.
    use_ema: bool = False

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-02
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0

    logging_dir: str = "logs"
    report_to: str = "wandb"

    # Choose between fp16 and bf16 (requires Nvidia Ampere GPU)
    mixed_precision: str = "no"
    # For distributed training: local_rank
    local_rank: int = 1

    # Save a checkpoint of the training state every X updates.
    # Checkpoints can be used for resuming training via `--resume_from_checkpoint`.
    checkpointing_steps: int = 500
    # Max number of checkpoints to store.
    checkpoints_total_limit: int = None
    # Whether training should be resumed from a previous checkpoint. Use a path saved by
    # `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.
    resume_from_checkpoint: str = None

    # ====== HCN (Hierarchical Conditioner Network) Parameters ======
    # Whether to use HCN for compositional demographic embeddings
    use_hcn: bool = False
    # Hidden dimension for HCN embeddings
    hcn_d_node: int = 256
    # Output dimension for HCN (should match text encoder output, e.g., 1024 for SD 2.1)
    hcn_d_ctx: int = 1024
    # Dropout probability in HCN MLPs
    hcn_dropout: float = 0.1
    # Whether to use uncertainty quantification in HCN
    hcn_use_uncertainty: bool = True
    # Number of age bins for categorization
    hcn_num_age_bins: int = 5
    # Number of sex categories (typically 2: M/F)
    hcn_num_sex: int = 2
    # Number of race/ethnicity categories
    hcn_num_race: int = 4
    # Weight for KL divergence loss (uncertainty regularization)
    hcn_kl_weight: float = 0.001
    # Number of steps to anneal KL weight from 0 to target value
    hcn_kl_anneal_steps: int = 10000
    # Weight for compositional consistency loss
    hcn_comp_weight: float = 0.01
    # Weight for auxiliary loss
    hcn_aux_weight: float = 0.1
    
    # ====== Demographic Encoder (V4) Parameters ======
    # Whether to use DemographicEncoder (lightweight embeddings-based conditioning)
    use_demographic_encoder: bool = False
    # Hidden dimension for demographic embeddings
    demo_d_hidden: int = 256
    # Output dimension (should match text encoder output, e.g., 1024 for SD 2.1)
    demo_d_output: int = 1024
    # Dropout probability in demographic encoder MLP
    demo_dropout: float = 0.1
    # Number of age bins for categorization
    demo_num_age_bins: int = 5
    # Number of sex categories (typically 2: M/F)
    demo_num_sex: int = 2
    # Number of race/ethnicity categories
    demo_num_race: int = 4
    # Weight for auxiliary demographic classification losses
    demo_aux_weight: float = 1.0
    # Whether to use demographic dropout strategy (50% of batches remove demographics from text)
    demo_use_dropout: bool = False
    # Dropout probability for removing demographics from text prompt
    demo_text_dropout_prob: float = 0.5
    # Step at which to start demographic dropout (allows warm-up period)
    demo_dropout_start_step: int = 0
    # Path to pretrained demographic encoder (optional)
    demographic_encoder_pretrained_path: str = None
    
    # ====== FairDiffusion Parameters ======
    use_fairdiffusion: bool = False
    fairdiffusion_input_perturbation: float = 0.0
    fairdiffusion_time_window: int = 250
    fairdiffusion_exploitation_rate: float = 0.7
    fairdiffusion_sigma_init: float = 1.0
    fairdiffusion_sigma_min: float = 0.0
    fairdiffusion_sigma_max: float = 1.0
    fairdiffusion_min_instance_weight: float = 0.1
    fairdiffusion_ucb_beta: float = 0.1
    fairdiffusion_attribute_fields: List[str] = field(
        default_factory=lambda: ["race_idx", "sex_idx", "age_idx"]
    )
    fairdiffusion_attribute_cardinalities: Dict[str, int] = field(default_factory=dict)
    # ====== Validation Parameters ======
    # Whether to run validation during training
    run_validation: bool = False
    # Run validation every N steps
    validation_steps: int = 2500
    # Optional offsets applied to each validation interval (e.g., [-1500, 0])
    validation_schedule_offsets: Optional[List[int]] = None
    # Optional minimum global step before applying schedule
    validation_schedule_min_step: Optional[int] = None
    # Number of validation samples to use (-1 for all)
    num_validation_samples: int = 100
    # Batch size for validation generation (number of images generated in parallel)
    val_batch_size: int = 1
    # Path to validation CSV file
    validation_csv: str = None
    # Path to real validation images directory
    validation_images_dir: str = None
    # Path to sex model checkpoint for validation
    validation_sex_model_path: str = None
    # Number of images to generate per validation prompt
    validation_num_images_per_prompt: int = 4
    # Guidance scale for validation generation
    validation_guidance_scale: float = 7.5
    # Number of inference steps for validation
    validation_num_inference_steps: int = 50
    # Whether to save validation images
    validation_save_images: bool = False
    # Batch size for loading and processing validation images when computing metrics (to avoid OOM)
    validation_metrics_batch_size: int = 32
    # Whether to compute subgroup-specific metrics (FID, MS-SSIM, BioViL per sex/race/age and intersectional subgroups)
    # This can significantly increase validation time, so it's disabled by default
    compute_subgroup_metrics: bool = False

    def get_config(self):
        return self.__dict__


def load_config(file_path):
    with open(file_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as e:
            print("Error loading YAML:", e)
            return None


def get_args_from_config():
    parser = argparse.ArgumentParser(
        description="Training script to fine-tune unet of the stable diffusion model."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        required=True,
        help="Experiment config file.",
    )
    args = parser.parse_args()
    config = load_config(args.config_file)
    my_args = BaseConfig(**config)
    return my_args
