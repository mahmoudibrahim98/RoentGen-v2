# RoentGen-v2 Validation Metrics Guide

This guide explains how to use the validation metrics implementation to evaluate RoentGen-v2 model checkpoints.

## Overview

The validation system implements three categories of metrics as described in the paper:

1. **Text Prompt Alignment**: Evaluates how accurately the model follows text prompts
2. **Real-Synthetic Image Similarity**: Measures fidelity to real medical images
3. **Intra-Prompt Diversity**: Quantifies variability among images from the same prompt

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Note: For BioViL embeddings, you may need to install additional dependencies:

```bash
pip install transformers[vision]
```

## Quick Start

### 1. Prepare Validation Data

Create a CSV file with your validation set containing the following columns:

- `prompt`: Text prompt (e.g., "45 year old White Male. Pneumothorax in left lung")
- `sex`: Patient sex (M/F)
- `race`: Patient race (White/Black/Asian/Hispanic)
- `age`: Patient age in years
- `atelectasis`: Binary label (0/1) - optional
- `cardiomegaly`: Binary label (0/1) - optional
- `edema`: Binary label (0/1) - optional
- `pneumothorax`: Binary label (0/1) - optional
- `pleural effusion`: Binary label (0/1) - optional
- `image_id`: Unique identifier for matching real images - optional
- `real_image_path`: Path to corresponding real image - optional

Example CSV:

```csv
prompt,sex,race,age,atelectasis,cardiomegaly,edema,pneumothorax,pleural effusion,image_id
"45 year old White Male. Pneumothorax in left lung",M,White,45,0,0,0,1,0,img001
"62 year old Black Female. Cardiomegaly with pleural effusion",F,Black,62,0,1,0,0,1,img002
```

### 2. Run Validation

Basic usage (text alignment only):

```bash
python roentgenv2/train_code/run_validation.py \
    --checkpoint_path ./checkpoints/checkpoint-5000 \
    --validation_csv ./data/validation.csv \
    --output_dir ./validation_results
```

Full validation with real image comparison:

```bash
python roentgenv2/train_code/run_validation.py \
    --checkpoint_path ./checkpoints/checkpoint-5000 \
    --validation_csv ./data/validation.csv \
    --real_images_dir ./data/validation_images \
    --sex_model_path ./models/sex_model.pth \
    --output_dir ./validation_results \
    --save_images
```

### 3. Advanced Options

```bash
python roentgenv2/train_code/run_validation.py \
    --checkpoint_path ./checkpoints/checkpoint-5000 \
    --validation_csv ./data/validation.csv \
    --real_images_dir ./data/validation_images \
    --sex_model_path ./models/sex_model.pth \
    --num_images_per_prompt 4 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --output_dir ./validation_results \
    --save_images \
    --device cuda
```

**Parameters**:
- `--checkpoint_path`: Path to model checkpoint directory (required)
- `--validation_csv`: Path to validation CSV file (required)
- `--real_images_dir`: Directory containing real validation images
- `--sex_model_path`: Path to sex prediction model checkpoint
- `--num_images_per_prompt`: Number of synthetic images per prompt (default: 4)
- `--guidance_scale`: Classifier-free guidance scale (default: 7.5)
- `--num_inference_steps`: Number of denoising steps (default: 50)
- `--output_dir`: Directory to save results (default: validation_results)
- `--save_images`: Save generated images
- `--device`: Device to run on (cuda/cpu)

## Output

The validation script generates:

1. **validation_metrics.json**: Summary of all computed metrics
2. **validation_results_TIMESTAMP.json**: Detailed results with configuration
3. **synthetic_images/** (if --save_images): Generated images organized by prompt

### Example Output

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

## Metric Interpretation

### Text Prompt Alignment

**Disease AUROC** (higher is better):
- > 0.8: Excellent alignment
- 0.7-0.8: Good alignment
- 0.6-0.7: Fair alignment
- < 0.6: Poor alignment

**Sex/Race Accuracy** (higher is better):
- > 0.9: Excellent
- 0.8-0.9: Good
- 0.7-0.8: Fair
- < 0.7: Poor

**Age RMSE** (lower is better):
- < 5 years: Excellent
- 5-10 years: Good
- 10-15 years: Fair
- > 15 years: Poor

### Real-Synthetic Similarity

**FID Score** (lower is better):
- < 30: Excellent quality
- 30-50: Good quality
- 50-100: Fair quality
- > 100: Poor quality

**BioViL Cosine Similarity** (higher is better):
- > 0.8: Excellent similarity
- 0.7-0.8: Good similarity
- 0.6-0.7: Fair similarity
- < 0.6: Poor similarity

**MS-SSIM** (target: 0.25-0.75):
- 1.0: Overfitting (identical to training data)
- 0.5-0.8: Good similarity
- 0.25-0.5: Fair similarity
- < 0.25: Poor similarity (not realistic CXRs)

### Intra-Prompt Diversity

**MS-SSIM & BioViL Similarity** (lower is better):
- 1.0: Model collapse (all images identical)
- 0.7-0.9: Low diversity
- 0.4-0.7: Good diversity
- < 0.4: High diversity

## Programmatic Usage

You can also use the validation metrics directly in your code:

```python
from roentgenv2.train_code.validation_metrics import ValidationMetricsRunner
import torch

# Initialize metrics runner
metrics_runner = ValidationMetricsRunner(
    device="cuda",
    sex_model_path="./models/sex_model.pth"
)

# Prepare your data
real_images = torch.randn(100, 1, 512, 512)  # [N, C, H, W]
synthetic_images = torch.randn(100, 1, 512, 512)
disease_labels = torch.randint(0, 2, (100, 5))  # [N, 5]
sex_labels = torch.randint(0, 2, (100,))
race_labels = torch.randint(0, 4, (100,))
age_labels = torch.rand(100) * 80 + 18

# Compute all metrics
metrics = metrics_runner.compute_all_metrics(
    real_images=real_images,
    synthetic_images=synthetic_images,
    disease_labels=disease_labels,
    sex_labels=sex_labels,
    race_labels=race_labels,
    age_labels=age_labels
)

# Print summary
metrics_runner.print_metrics_summary(metrics)
```

### Individual Metric Computation

```python
from roentgenv2.train_code.validation_metrics import (
    TextPromptAlignmentMetrics,
    RealSyntheticSimilarityMetrics,
    IntraPromptDiversityMetrics
)

# Text alignment metrics
text_metrics = TextPromptAlignmentMetrics(device="cuda")
text_metrics.load_sex_model("./models/sex_model.pth")

aurocs = text_metrics.compute_disease_auroc(synthetic_images, disease_labels)
sex_acc = text_metrics.compute_sex_accuracy(synthetic_images, sex_labels)
race_acc = text_metrics.compute_race_accuracy(synthetic_images, race_labels)
age_rmse = text_metrics.compute_age_rmse(synthetic_images, age_labels)

# Similarity metrics
sim_metrics = RealSyntheticSimilarityMetrics(device="cuda")

fid = sim_metrics.compute_fid(real_images, synthetic_images)
biovil_sim = sim_metrics.compute_biovil_similarity(real_images, synthetic_images)
ms_ssim = sim_metrics.compute_ms_ssim(real_images, synthetic_images)

# Diversity metrics
div_metrics = IntraPromptDiversityMetrics(device="cuda")

# List of [K, C, H, W] tensors (K images per prompt)
images_per_prompt = [torch.randn(4, 1, 512, 512) for _ in range(100)]

intra_ms_ssim = div_metrics.compute_intra_prompt_ms_ssim(images_per_prompt)
intra_biovil = div_metrics.compute_intra_prompt_biovil_similarity(images_per_prompt)
```

## Integration with Training

To integrate validation into your training loop:

```python
from roentgenv2.train_code.validation_metrics import ValidationMetricsRunner

# Initialize once
metrics_runner = ValidationMetricsRunner(device="cuda")

# During training, at checkpoint intervals
@torch.no_grad()
def validate_epoch(model, val_dataloader, metrics_runner):
    model.eval()

    all_real_images = []
    all_synthetic_images = []
    all_labels = {"disease": [], "sex": [], "race": [], "age": []}

    for batch in val_dataloader:
        # Generate synthetic images
        synthetic = model.generate(batch["prompt"])

        all_real_images.append(batch["image"])
        all_synthetic_images.append(synthetic)
        all_labels["disease"].append(batch["disease_labels"])
        all_labels["sex"].append(batch["sex"])
        all_labels["race"].append(batch["race"])
        all_labels["age"].append(batch["age"])

    # Concatenate all batches
    real_images = torch.cat(all_real_images)
    synthetic_images = torch.cat(all_synthetic_images)

    # Compute metrics
    metrics = metrics_runner.compute_all_metrics(
        real_images=real_images,
        synthetic_images=synthetic_images,
        disease_labels=torch.cat(all_labels["disease"]),
        sex_labels=torch.cat(all_labels["sex"]),
        race_labels=torch.cat(all_labels["race"]),
        age_labels=torch.cat(all_labels["age"])
    )

    model.train()
    return metrics
```

## Troubleshooting

### BioViL Model Not Loading

If you encounter issues loading the BioViL model:

```python
# Option 1: Use different model name
from transformers import AutoModel
model = AutoModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

# Option 2: Disable BioViL metrics
# The code will automatically skip BioViL metrics and return NaN
```

### Out of Memory Errors

If you encounter OOM errors:

1. Reduce batch size in metrics computation
2. Use CPU for some metrics: `device="cpu"`
3. Process validation set in chunks
4. Reduce `num_images_per_prompt`

### Missing Real Images

If real images are not available:
- The script will skip real-synthetic similarity metrics
- You will only get text alignment and diversity metrics
- This is useful for rapid checkpoint evaluation

## Citation

If you use these validation metrics, please cite the RoentGen-v2 paper:

```bibtex
@article{roentgenv2,
  title={RoentGen-v2: Demographically Fair Chest X-ray Generation via Hierarchical Conditioning},
  author={...},
  journal={...},
  year={2025}
}
```

## References

1. Torch X-ray Vision: https://github.com/mlmed/torchxrayvision
2. BioViL: https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized
3. Sex prediction model: https://www.thelancet.com/journals/ebiom/article/PIIS2352-3964(23)00032-4/fulltext
