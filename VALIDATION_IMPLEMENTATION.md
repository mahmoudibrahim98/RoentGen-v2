# Validation Metrics Implementation Summary

This document summarizes the validation metrics implementation for RoentGen-v2 based on the paper's methodology.

## Implementation Status: ✅ Complete

All three categories of validation metrics have been fully implemented as described in the paper.

## Files Created

### 1. Core Metrics Module
**File**: `roentgenv2/train_code/validation_metrics.py`

Implements four main classes:

#### `TextPromptAlignmentMetrics`
Evaluates how accurately RoentGen-v2 adheres to provided text prompts.

**Features**:
- Disease classification using XRV DenseNet-121 model
- AUROC computation for 5 target diseases:
  - Atelectasis
  - Cardiomegaly
  - Edema
  - Pneumothorax
  - Pleural Effusion
- Sex classification accuracy (custom model from Lancet paper)
- Race classification accuracy (XRV model)
- Age prediction RMSE in years (XRV model)
- Automatic image preprocessing (512x512 → 224x224 with area interpolation)

#### `RealSyntheticSimilarityMetrics`
Evaluates similarity between real and synthetic images.

**Features**:
- **FID (Fréchet Inception Distance)**: Uses Inception v3 embeddings
  - Measures distribution similarity
  - Lower scores indicate better quality
  - 0 = perfect match
- **BioViL Cosine Similarity**: Medical image encoder from Microsoft
  - Pairwise cosine similarity between real/synthetic pairs
  - 1.0 = identical embeddings
  - Uses state-of-the-art CXR interpretation model
- **MS-SSIM (Multi-Scale Structural Similarity)**:
  - Structural similarity across multiple scales
  - Range: 0 (no similarity) to 1 (identical)
  - Target range: 0.25-0.75 (avoiding overfitting)

#### `IntraPromptDiversityMetrics`
Quantifies diversity among multiple images from the same prompt.

**Features**:
- **MS-SSIM Diversity**: Average pairwise MS-SSIM within prompt
  - Lower scores = more diverse (desirable)
  - 1.0 = model collapse
- **BioViL Embedding Similarity**: Semantic diversity
  - Pairwise cosine similarity of BioViL embeddings
  - Lower scores = more diverse (desirable)
  - Captures semantic content variability

#### `ValidationMetricsRunner`
Unified interface for computing all metrics.

**Features**:
- Single method to compute all metrics
- Formatted summary printing
- Automatic handling of missing data
- GPU/CPU support

### 2. Validation Runner Script
**File**: `roentgenv2/train_code/run_validation.py`

End-to-end validation script for checkpoint evaluation.

**Features**:
- Load model checkpoints
- Generate synthetic images with multiple random seeds
- Compute all validation metrics
- Save results to JSON
- Optional image saving
- Comprehensive CLI interface

**Command-line Arguments**:
```bash
--checkpoint_path       # Model checkpoint (required)
--validation_csv        # Validation data (required)
--real_images_dir       # Real images directory (optional)
--sex_model_path        # Sex model checkpoint (optional)
--num_images_per_prompt # Images per prompt (default: 4)
--guidance_scale        # CFG scale (default: 7.5)
--num_inference_steps   # Denoising steps (default: 50)
--output_dir            # Results directory
--save_images           # Save generated images
--device                # cuda/cpu
```

### 3. Documentation
**File**: `VALIDATION_GUIDE.md`

Comprehensive user guide covering:
- Installation instructions
- Quick start examples
- Advanced usage
- Output interpretation
- Metric thresholds and guidelines
- Programmatic API usage
- Training integration examples
- Troubleshooting

### 4. Example Configuration
**File**: `configs/validation_example.csv`

Sample validation CSV with 10 diverse cases covering:
- All demographic groups (sex, race, age ranges)
- Multiple disease combinations
- Realistic clinical findings

### 5. Updated Requirements
**File**: `requirements.txt`

Added dependencies:
- `pytorch-msssim==1.0.0` - Multi-Scale SSIM
- `scipy>=1.10.0` - Matrix operations for FID
- `scikit-learn>=1.3.0` - Classification metrics
- `pillow>=10.0.0` - Image loading
- `tqdm>=4.65.0` - Progress bars

## Implementation Details

### Metric Computation Flow

```
1. Load Model Checkpoint
   ↓
2. Load Validation Data (CSV + optional real images)
   ↓
3. Generate Synthetic Images
   - For each prompt: generate N images with different seeds
   - Default: 4 images per prompt
   ↓
4. Compute Text Prompt Alignment
   - Preprocess to 224x224 (area interpolation)
   - XRV disease model → AUROC
   - XRV race model → Accuracy
   - XRV age model → RMSE
   - Sex model → Accuracy (if provided)
   ↓
5. Compute Real-Synthetic Similarity (if real images available)
   - Inception v3 → FID score
   - BioViL → Cosine similarity
   - MS-SSIM → Structural similarity
   ↓
6. Compute Intra-Prompt Diversity
   - Pairwise MS-SSIM within prompts
   - Pairwise BioViL similarity within prompts
   ↓
7. Save Results
   - JSON metrics file
   - Detailed results with config
   - Optional: synthetic images
```

### Paper Alignment

| Paper Requirement | Implementation | Location |
|-------------------|----------------|----------|
| XRV DenseNet-121 for diseases | ✅ Implemented | `TextPromptAlignmentMetrics.__init__`:46 |
| AUROC for 5 diseases | ✅ Implemented | `compute_disease_auroc`:104 |
| Sex prediction model | ✅ Implemented | `load_sex_model`:76 |
| Race classification (XRV) | ✅ Implemented | `compute_race_accuracy`:177 |
| Age prediction (XRV) | ✅ Implemented | `compute_age_rmse`:196 |
| Area interpolation (512→224) | ✅ Implemented | `preprocess_images`:87 |
| FID with Inception v3 | ✅ Implemented | `compute_fid`:269 |
| BioViL embeddings | ✅ Implemented | `compute_biovil_similarity`:323 |
| MS-SSIM | ✅ Implemented | `compute_ms_ssim`:372 |
| 4 images per prompt | ✅ Implemented | `run_validation.py` default |
| Intra-prompt MS-SSIM | ✅ Implemented | `compute_intra_prompt_ms_ssim`:439 |
| Intra-prompt BioViL similarity | ✅ Implemented | `compute_intra_prompt_biovil_similarity`:475 |

## Usage Examples

### Basic Validation
```bash
python roentgenv2/train_code/run_validation.py \
    --checkpoint_path ./checkpoints/checkpoint-5000 \
    --validation_csv ./configs/validation_example.csv \
    --output_dir ./validation_results
```

### Full Validation with Real Images
```bash
python roentgenv2/train_code/run_validation.py \
    --checkpoint_path ./checkpoints/checkpoint-5000 \
    --validation_csv ./data/mimic_cxr_validation.csv \
    --real_images_dir ./data/mimic_cxr_images \
    --sex_model_path ./models/sex_model.pth \
    --num_images_per_prompt 4 \
    --save_images \
    --output_dir ./validation_results
```

### Programmatic Usage
```python
from roentgenv2.train_code.validation_metrics import ValidationMetricsRunner

metrics_runner = ValidationMetricsRunner(device="cuda")

metrics = metrics_runner.compute_all_metrics(
    real_images=real_imgs,
    synthetic_images=synth_imgs,
    disease_labels=diseases,
    sex_labels=sex,
    race_labels=race,
    age_labels=age
)

metrics_runner.print_metrics_summary(metrics)
```

## Expected Output

```
============================================================
VALIDATION METRICS SUMMARY
============================================================

1. TEXT PROMPT ALIGNMENT
------------------------------------------------------------
  Disease Classification (AUROC):
    Atelectasis        : 0.7845
    Cardiomegaly       : 0.8123
    Edema              : 0.7654
    Pneumothorax       : 0.8234
    Pleural Effusion   : 0.7912
    Mean AUROC         : 0.7954

  Demographic Attributes:
    Sex Accuracy:   0.8950
    Race Accuracy:  0.7234
    Age RMSE:       8.45 years

2. REAL-SYNTHETIC IMAGE SIMILARITY
------------------------------------------------------------
  FID Score:               45.23 (lower is better)
  BioViL Cosine Similarity: 0.7856 (1.0 = perfect)
  MS-SSIM:                 0.4523 (1.0 = identical)

3. INTRA-PROMPT DIVERSITY
------------------------------------------------------------
  Intra-Prompt MS-SSIM:     0.3421 (lower = more diverse)
  Intra-Prompt BioViL Sim:  0.5234 (lower = more diverse)

============================================================
```

## Model Selection Guidelines

Based on the computed metrics, select the optimal checkpoint:

### High Priority Metrics (Must Pass)
1. **Disease AUROC** > 0.75 (mean across diseases)
2. **MS-SSIM** > 0.25 (ensures realistic CXRs)
3. **Intra-Prompt MS-SSIM** < 0.8 (avoids mode collapse)

### Medium Priority Metrics (Should Pass)
1. **Sex Accuracy** > 0.85
2. **FID** < 100
3. **BioViL Similarity** > 0.6

### Nice-to-Have Metrics
1. **Race Accuracy** > 0.70
2. **Age RMSE** < 10 years
3. **FID** < 50

### Balanced Selection
Select checkpoint with:
- Best disease AUROC (primary clinical utility)
- Acceptable demographic fairness (sex/race accuracy)
- Good diversity (low intra-prompt similarity)
- Reasonable FID (not necessarily lowest, as may overfit)

## Integration with Training Loop

The metrics can be integrated into the training loop at `roentgenv2/train_code/train_loop.py:214` where the TODO comment currently exists:

```python
# At line 214 in train_loop.py
if global_step % args.validation_steps == 0:
    from validation_metrics import ValidationMetricsRunner

    metrics_runner = ValidationMetricsRunner(device=accelerator.device)

    # Run validation
    val_metrics = validate_epoch(
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        tokenizer=tokenizer,
        hcn=hcn,
        val_dataloader=val_dataloader,
        metrics_runner=metrics_runner,
        accelerator=accelerator
    )

    # Log to wandb
    accelerator.log(val_metrics, step=global_step)

    # Save checkpoint if best
    if val_metrics["mean_auroc"] > best_auroc:
        best_auroc = val_metrics["mean_auroc"]
        save_checkpoint(args.output_dir, "best_model")
```

## Dependencies

All required libraries:
- `torchxrayvision` - XRV models (already installed)
- `pytorch-msssim` - MS-SSIM computation (NEW)
- `scipy` - FID covariance matrix (NEW)
- `scikit-learn` - AUROC, accuracy metrics (NEW)
- `transformers` - BioViL model (already installed)
- `pillow` - Image I/O (NEW)
- `tqdm` - Progress tracking (NEW)

## Testing

To verify the implementation works:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test with example data
python roentgenv2/train_code/run_validation.py \
    --checkpoint_path <your_checkpoint> \
    --validation_csv ./configs/validation_example.csv \
    --output_dir ./test_validation

# 3. Check output files
ls -la ./test_validation/
# Expected: validation_metrics.json, validation_results_*.json
```

## Known Limitations

1. **BioViL Model**: May require HuggingFace authentication for some versions
   - Fallback: Metrics will return NaN if model unavailable
   - Alternative: Use FID and MS-SSIM only

2. **Memory Usage**: Processing large validation sets may require chunking
   - Solution: Process in batches in `run_validation.py`

3. **Sex Model**: Requires separate checkpoint download
   - Optional: Can run validation without sex accuracy metric

## Future Enhancements

Potential additions:
- [ ] LPIPS (Learned Perceptual Image Patch Similarity)
- [ ] Demographic fairness metrics (statistical parity, equalized odds)
- [ ] Uncertainty calibration metrics for HCN
- [ ] Clinical reader study integration
- [ ] Automated checkpoint selection based on composite score

## References

1. **TorchXRayVision**: Cohen et al., "TorchXRayVision: A library of chest X-ray datasets and models"
2. **BioViL**: Boecking et al., "Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing"
3. **Sex Prediction Model**: Larrazabal et al., "Sex classification from chest X-ray images using deep learning"
4. **MS-SSIM**: Wang et al., "Multiscale structural similarity for image quality assessment"
5. **FID**: Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
