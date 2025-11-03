# HCN Integration - Implementation Guide

## Status: 90% Complete

### ✅ Completed Components

1. **HCN Module** (`hcn.py`) - Fully implemented
2. **Dataset Modifications** (`dataset_wds.py`) - Demographic parsing added
3. **Config Files** (`config.py`, `train_config_demo.yaml`) - HCN parameters added
4. **Model Loading** (`models.py`) - `load_hcn` function added
5. **Main Training Script** (`train.py`) - HCN initialization and preparation complete
6. **Test Script** (`test_hcn_integration.py`) - Comprehensive testing suite

### 🚧 Remaining Modifications

Two critical files need updates:

1. **train_loop.py** - Integrate HCN conditioning and losses
2. **pipeline.py** - Save/load HCN with pipeline

---

## CRITICAL MODIFICATION 1: train_loop.py

### Location of Changes

Find the `train_loop` function definition and modify it to:

1. Add `hcn=None` parameter to function signature
2. In the training loop, after getting text embeddings, add HCN conditioning
3. Add KL and compositional losses
4. Include HCN parameters in gradient clipping

### Code Changes Needed

#### Step 1: Modify function signature

```python
# FIND (around line 8):
def train_loop(
    logger,
    args,
    initial_global_step,
    first_epoch,
    accelerator,
    train_dataloader,
    unet,
    text_encoder,
    vae,
    noise_scheduler,
    weight_dtype,
    optimizer,
    lr_scheduler,
    ema_unet,
):

# REPLACE WITH:
def train_loop(
    logger,
    args,
    initial_global_step,
    first_epoch,
    accelerator,
    train_dataloader,
    unet,
    text_encoder,
    vae,
    noise_scheduler,
    weight_dtype,
    optimizer,
    lr_scheduler,
    ema_unet,
    hcn=None,  # ADD THIS PARAMETER
):
```

#### Step 2: Set HCN to training mode

```python
# FIND (around line 32-37):
    unet.train()
    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info("Epoch {}, global step {}".format(epoch, global_step))

        if args.train_text_encoder:
            text_encoder.train()

# REPLACE WITH:
    unet.train()
    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info("Epoch {}, global step {}".format(epoch, global_step))

        if args.train_text_encoder:
            text_encoder.train()

        # ADD THIS:
        if hcn is not None:
            hcn.train()
```

#### Step 3: Add HCN conditioning (MOST CRITICAL)

```python
# FIND (around line 70-81):
                # Get the text embedding for conditioning
                text_input_ids = batch["input_ids"]
                if args.use_attention_mask:
                    attention_mask = batch["attention_mask"]
                else:
                    attention_mask = None

                prompt_embeds = text_encoder(
                    input_ids=text_input_ids,
                    attention_mask=attention_mask,
                    return_dict=False,
                )
                encoder_hidden_states = prompt_embeds[0]

# REPLACE WITH:
                # Get the text embedding for conditioning
                text_input_ids = batch["input_ids"]
                if args.use_attention_mask:
                    attention_mask = batch["attention_mask"]
                else:
                    attention_mask = None

                prompt_embeds = text_encoder(
                    input_ids=text_input_ids,
                    attention_mask=attention_mask,
                    return_dict=False,
                )
                encoder_hidden_states = prompt_embeds[0]  # [B, 77, d_ctx]

                # === ADD HCN CONDITIONING ===
                kl_loss = None
                comp_loss = None
                if hcn is not None:
                    # Get HCN demographic context
                    hcn_ctx, mu, logsigma = hcn(
                        batch["age_idx"],
                        batch["sex_idx"],
                        batch["race_idx"],
                    )  # hcn_ctx: [B, 1, d_ctx]

                    # Concatenate text and demographic contexts
                    encoder_hidden_states = torch.cat(
                        [encoder_hidden_states, hcn_ctx], dim=1
                    )  # [B, 78, d_ctx]

                    # Compute KL divergence (uncertainty regularization)
                    kl_loss = -0.5 * torch.sum(
                        1 + 2 * logsigma - mu ** 2 - torch.exp(2 * logsigma),
                        dim=-1
                    ).mean()

                    # Compute compositional consistency loss
                    comp_loss = hcn.compute_compositional_loss(
                        batch["age_idx"],
                        batch["sex_idx"],
                        batch["race_idx"],
                    )
```

#### Step 4: Add HCN losses to total loss

```python
# FIND (around line 102-106):
                # Compute instance loss
                loss = F.mse_loss(
                    noise_pred.float(), target.float(), reduction="none"
                ).mean([1, 2, 3])
                loss_weights = batch["loss_weights"].to(dtype=weight_dtype)
                loss = (loss * loss_weights).sum() / loss_weights.sum()

# REPLACE WITH:
                # Compute instance loss
                loss = F.mse_loss(
                    noise_pred.float(), target.float(), reduction="none"
                ).mean([1, 2, 3])
                loss_weights = batch["loss_weights"].to(dtype=weight_dtype)
                loss = (loss * loss_weights).sum() / loss_weights.sum()

                # === ADD HCN LOSSES ===
                if kl_loss is not None:
                    # Anneal KL weight from 0 to target value
                    kl_weight = min(1.0, global_step / args.hcn_kl_anneal_steps) * args.hcn_kl_weight
                    loss = loss + kl_weight * kl_loss

                if comp_loss is not None:
                    loss = loss + args.hcn_comp_weight * comp_loss
```

#### Step 5: Add HCN to gradient clipping

```python
# FIND (around line 109-115):
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

# REPLACE WITH:
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    # ADD HCN PARAMETERS:
                    if hcn is not None:
                        params_to_clip = itertools.chain(params_to_clip, hcn.parameters())

                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
```

#### Step 6: Add HCN losses to logging

```python
# FIND (around line 168-170):
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

# REPLACE WITH:
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            # ADD HCN LOSS LOGGING:
            if kl_loss is not None:
                logs["kl_loss"] = kl_loss.detach().item()
                logs["kl_weight"] = kl_weight
            if comp_loss is not None:
                logs["comp_loss"] = comp_loss.detach().item()

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
```

---

## CRITICAL MODIFICATION 2: pipeline.py

### Code Changes Needed

#### Modify create_and_save_pipeline function

```python
# FIND the function signature (around line 10-20):
def create_and_save_pipeline(
    args,
    accelerator,
    unet,
    text_encoder,
    text_encoder_name,
    tokenizer,
    vae,
    ema_unet,
    kwargs_from_pretrained,
    unet_config_changed,
    unet_config,
):

# REPLACE WITH:
def create_and_save_pipeline(
    args,
    accelerator,
    unet,
    text_encoder,
    text_encoder_name,
    tokenizer,
    vae,
    ema_unet,
    kwargs_from_pretrained,
    unet_config_changed,
    unet_config,
    hcn=None,  # ADD THIS PARAMETER
):
```

#### Add HCN saving at end of function

```python
# FIND (near the end of the function, after pipeline.save_pretrained):
    pipeline.save_pretrained(args.output_dir)

    # ... some logging code ...

# ADD AFTER THIS:

    # === SAVE HCN ===
    if hcn is not None:
        import os
        hcn_unwrapped = accelerator.unwrap_model(hcn)
        hcn_save_path = os.path.join(args.output_dir, "hcn")
        hcn_unwrapped.save_pretrained(hcn_save_path)
        logger.info(f"HCN saved to {hcn_save_path}")
```

---

## Testing Your Implementation

### Step 1: Run Unit Tests

```bash
cd /mnt/c/Users/ibrahimm/RoentGen-v2
python test_hcn_integration.py
```

Expected output:
```
==============================================================
✅ ALL TESTS PASSED!
==============================================================
```

### Step 2: Test Training (Dry Run)

Modify `configs/train_config_demo.yaml`:
```yaml
max_train_steps: 2  # Just 2 steps for testing
use_hcn: True
```

Run training:
```bash
accelerate launch --num_processes=1 --mixed_precision bf16 \
  roentgenv2/train_code/train.py \
  --config_file="./configs/train_config_demo.yaml"
```

Watch for in logs:
```
HCN initialized with 339,464 parameters
HCN mode enabled: parsing demographics from prompts
KL loss: 0.XXXX, KL weight: 0.XXXXXX
Compositional loss: 0.XXXX
```

### Step 3: Verify HCN is Training

After 2 steps, check output directory:
```bash
ls outputs/hcn/
# Should contain: config.json, pytorch_model.bin
```

### Step 4: Full Training

Once dry run works:
```yaml
max_train_steps: 60000
use_hcn: True
```

---

## Troubleshooting

### Error: "batch missing age_idx"

**Cause**: Dataset not parsing demographics
**Fix**: Ensure `use_hcn: True` in config

### Error: "dimension mismatch in cat"

**Cause**: HCN d_ctx doesn't match text encoder
**Fix**: Set `hcn_d_ctx: 1024` (for SD 2.1) or 768 (for SD 1.x)

### Error: "KL loss is nan"

**Cause**: KL weight too high initially
**Fix**: Increase `hcn_kl_anneal_steps` to 10000+

### Warning: "logsigma clamped"

**Normal**: This prevents numerical instability, ignore

---

## Expected Training Behavior

### Loss Curves

- **Diffusion loss**: Should decrease normally (0.1 → 0.05 typically)
- **KL loss**: Starts high (~50), decreases to ~5-10
- **Compositional loss**: Starts high (~0.5), decreases to ~0.1-0.2

### HCN Uncertainty

Rare demographic groups should have **higher sigma** (uncertainty).

To check after training:
```python
from hcn import HierarchicalConditioner

hcn = HierarchicalConditioner.from_pretrained("outputs/hcn")

# Common group (e.g., middle-aged white male)
sigma_common = hcn.get_uncertainty(
    age_idx=torch.tensor([2]),  # 40-60
    sex_idx=torch.tensor([0]),   # Male
    race_idx=torch.tensor([0]),  # White
)

# Rare group (e.g., elderly Asian female)
sigma_rare = hcn.get_uncertainty(
    age_idx=torch.tensor([4]),  # 80+
    sex_idx=torch.tensor([1]),   # Female
    race_idx=torch.tensor([2]),  # Asian
)

print(f"Common group sigma: {sigma_common.item():.4f}")
print(f"Rare group sigma: {sigma_rare.item():.4f}")
# sigma_rare should be > sigma_common
```

---

## Next Steps After Implementation

1. **Baseline Comparison**: Train one model with `use_hcn: False`, one with `use_hcn: True`
2. **Evaluation**: Measure per-group FID scores
3. **Ablation Studies**:
   - HCN without uncertainty (`hcn_use_uncertainty: False`)
   - HCN without compositional loss (`hcn_comp_weight: 0.0`)
4. **Hyperparameter Tuning**:
   - Try different `hcn_d_node` (128, 256, 512)
   - Adjust `hcn_kl_weight` (0.0001 - 0.01)

---

## Summary Checklist

- [x] HCN module created (`hcn.py`)
- [x] Dataset parsing added (`dataset_wds.py`)
- [x] Config files updated
- [x] Model loading function added (`models.py`)
- [x] Main training script modified (`train.py`)
- [x] Test script created
- [ ] **TODO**: Modify `train_loop.py` (follow guide above)
- [ ] **TODO**: Modify `pipeline.py` (follow guide above)
- [ ] **TODO**: Run `test_hcn_integration.py`
- [ ] **TODO**: Run dry-run training (2 steps)
- [ ] **TODO**: Run full training

**Estimated time to complete remaining tasks**: 1-2 hours

Good luck! 🚀
