# HCN Conditioning Debug Log
## Progress: What went wrong with V1 and V2?

| Version | What we did |Demographics in text|Demo Condition|Aux Loss| FID | Demographic | Directory|Steps| Notes |
|---------|-------------|---------|---------|---------|---------|----------|----------|-----|-----|
| **Baseline** | Single text encoder; prompt contains demographics and clinical finding |✅ Yes|In Clip|❌ No| ❌FID ≈ 100 | ✅Perfect ~ 1 |outputs/output/1_baseline|27500|  |
| **HCN v1 (bugged)** | Removed demographics from prompt, relied on HCN token only; No aux loss |❌ No|HCN Token|❌ No| ✅ FID improved to ~82 | ❌ Demographic accuracy collapsed (≈0.5) |outputs/output/0_full_hcn| 32500| |
| **HCN v1.5 (base)** 🟡 | Remove Demographics from text + HCN token + aux losses (0.01);|❌ No|HCN Token|✅ Yes| ⏳ | ⏳ |outputs/output_v1.5/0_hcn_with_dropout_weak_aux_loss | 9000|the current implementation have a wierd tokenization path: In training loop (original_prompt → tokenize → decode → extract_clinical_text() → re-tokenize)	instead of original_prompt → extract_clinical_text() → tokenize; |
| **HCN v1.5 (no aux)** 🟡 |  Remove Demographics from text + HCN token + aux losses (0);|❌ No|HCN Token|✅ Yes| ⏳  | ⏳ |outputs/output_v1.5/1_hcn_with_dropout_no_aux_loss | 15500|the current implementation have a wierd tokenization path: In training loop (original_prompt → tokenize → decode → extract_clinical_text() → re-tokenize)	instead of original_prompt → extract_clinical_text() → tokenize; also, eve if aux_loss_weight is 0, this implemenation has more parameters and a different forward signature than v1; repeated validaiton with fixed validaiton script (only first step), but no improvement.|
| **HCN v1.6 (no aux)** 🟡⏳ |  Remove Demographics from text + HCN token + aux losses (0);|❌ No|HCN Token|✅ Yes| ✅ one step improved to around 80   | ❌ demographic attributes collapsed. |outputs/output_v1.5/1_hcn_with_dropout_no_aux_loss | |aligned implementation with v1 more.|
| **HCN v2 (base)** | Put demographics back in the prompt + HCN token + aux losses; comp_weight = 0.01 |✅ Yes|HCN Token|✅ Yes| ❌FID regressed to baseline | ✅ Demographic accuracy recovered |outputs/output_v2/0_full_hcn| 7500||
| **HCN v2 (strong)** | Put demographics back in the prompt + HCN token + aux losses; comp_weight = 0.05 |✅ Yes|HCN Token|✅ Yes| ❌FID regressed to baseline(maybe a bit better)  | ✅ Demographic accuracy recovered|outputs/output_v2/0a_full_hcn_strong| 12500| |
| **HCN v2 (strongest)** | Put demographics back in the prompt + HCN token + aux losses; comp_weight = 0.1 |✅ Yes|HCN Token|✅ Yes|❌❌FID worse than baseline |❌Demographic accuracy collapsing|outputs/output_v2/0b_full_hcn_strongest| 2500 ||
| **HCN v3** | Dual CLIP encoder,extending CLIP's positional embeddings  |Yes|Dual Clip|❌ No| ❌❌FID worse than baseline | ❌Demographic accuracy collapsed|outputs/output_v3/1_dual_encoder_v3|7500||
| **v4 (baseline)** | dual-encoder architecture, simple demographic encoder (no dropout)  |✅ Yes|Simple Demo Token|✅ Yes| ❌❌❌✅ Demographic accuracy recovered (wrong results) | ❌❌❌FID regressed to baseline (wrong results) |outputs/output_v4/1_demographic_encoder_v4|10000||
| **v4 (full dropout)** 🟡| dual-encoder architecture, simple demographic encoder (full dropout of demograghic text)  |❌ No|Simple Demo Token|✅ Yes| ⏳ | ⏳ |outputs/output_v4/2_demographic_encoder_full_dropout|5000|the current implementation have a wierd tokenization path: In training loop (original_prompt → tokenize → decode → extract_clinical_text() → re-tokenize)	instead of original_prompt → extract_clinical_text() → tokenize; |


Improtant notice: for v4_2(with dropout), even when the training was done with dropout, the valdiaiton was able to recover demog accuracies if the demos was in text .however, if demos not in text, we werent able to recover the demog accuracy.




for **v1.5**, 
```
use_hcn: true
use_demographic_encoder: false
demo_use_dropout: true
demo_text_dropout_prob: 1.0
demo_dropout_start_step: 0
```
Key insight from v1: **when demographics were removed from the prompt, CLIP’s 77-token budget was freed up for clinical text**, so image quality (FID/BioViL) improved. But we lost subgroup conditioning because there was no demographic signal left.


## Background
- **Baseline** training fed the entire prompt (including “65-year-old BLACK FEMALE”) into the text encoder. Demographic predictors therefore saw near-perfect accuracy because the prompt itself was the conditioning signal.
- **HCN v1** removed demographics from the text prompt and relied entirely on a new Hierarchical Conditioner Network (HCN) token. The diffusion loss alone wasn’t strong enough to make the UNet rely on that token, so generated images lost subgroup cues and sex/race/age metrics collapsed, even though FID improved.

## Root cause
1. Text encoder lost explicit demographic words once `use_hcn=True`.
2. HCN losses (`hcn_kl_weight=1e-3`, `hcn_comp_weight=1e-2`) were too small/slowly annealed, so the HCN embedding carried almost no usable information.
3. No auxiliary supervision tied HCN outputs to actual age/sex/race categories.

## Fixes implemented (HCN v2)
1. **Keep demographics in text prompts** regardless of `use_hcn`, so baseline behavior is preserved while HCN provides an additional context token.
2. **Augmented HCN** with auxiliary classifiers that predict age/sex/race directly from the demographic embedding. The training loop logs `hcn_ctx_norm` and optimizes an averaged cross-entropy loss weighted by `hcn_aux_weight`.
3. **Config knobs for ablations**:
   - `hcn_kl_weight` and `hcn_kl_anneal_steps`
   - `hcn_comp_weight`
   - New `hcn_aux_weight`
4. Added `experiments/hcn_v2_ablation_plan.md` to track recommended sweeps (e.g., KL weight 5e-3, anneal 5k as the first run).

## Next steps
- Run the updated configuration (`configs/0_train_full_hcn.yaml` with KL=5e-3, anneal=5k) and monitor new metrics (`aux_loss`, `hcn_ctx_norm`) alongside subgroup accuracies.
- Compare checkpointed results against the baseline to verify that demographic predictors rebound while FID improvements remain.***

## V3 dual text conditioner (clinical + demographic branches)
- Keep the **clinical prompt** identical to v1 (demographics stripped) to retain the FID boost.
- Introduce a **second text-conditioning branch** containing structured demographic strings (`AGE: 60-80; SEX: Female; RACE: Black`) that is tokenized separately.
- Concatenate `[clinical_tokens, demographic_tokens, HCN token?]` before feeding the UNet, so clinical and demographic information no longer compete for the 77-token CLIP budget.
- Dataset changes:
  - Always provide `input_ids` for clinical text and new `demo_input_ids`/`demo_attention_mask` tensors for demographics when `use_dual_text_conditioner=true`.
  - Store `demo_text` in validation batches so the generation script can encode both branches.
- Training/validation changes:
  - `train_loop` encodes clinical and demographic text passes separately and concatenates embeddings.
  - Classifier-free guidance pads the unconditional branch with zero demographic tokens.
  - Validation monitor now tokenizes two prompt lists (`prompts` + `demo_prompts`) to mirror training.
- Config: `configs/0c_train_full_hcn_v3.yaml` enables the feature (`use_dual_text_conditioner: true`) and sets `demographic_prompt_max_length` (default 16 tokens).
- **FAILED**: Extended CLIP positional embeddings broke pre-trained representations. Demographic accuracy collapsed.

## V4 Demographic Encoder (Clean Dual-Encoder Architecture)
**Date**: November 2024  
**Status**: Implementation complete, ready for training

### What V4 Does Differently
- **No CLIP modification**: Keeps CLIP at 77 tokens with original positional embeddings
- **Separate demographic encoder**: Lightweight embeddings + MLP (not text-based)
- **Demographics as categorical indices**: age_idx, sex_idx, race_idx (not tokenized text)
- **Strong forcing mechanisms**: 
  - Auxiliary losses (weight=1.0) force embeddings to be discriminative
  - Optional dropout strategy removes demographics from text 50% of time
- **Concatenate after encoding**: `[clinical_embeds (77, 1024), demo_embeds (1, 1024)]`

### Architecture
```
clinical_text → CLIP (unchanged) → [B, 77, 1024]
(age, sex, race) → DemographicEncoder → [B, 1, 1024]
Concatenate → [B, 78, 1024] → UNet cross-attention
```

### Files Created
- `roentgenv2/train_code/demographic_encoder.py` - Core module (~100K params)
- `configs/v4/1_train_demographic_encoder_v4.yaml` - Base config
- `configs/v4/2_train_demographic_encoder_v4_with_dropout.yaml` - With dropout forcing
- `V4_README.md` - Complete documentation

### Files Modified
- `config.py` - Added V4 parameters
- `train_loop.py` - Demographic encoder conditioning + aux losses
- `train.py` - Load and prepare demographic encoder
- `dataset_wds.py` - Parse demographics for V4
- `pipeline.py` - Save demographic encoder

### Why V4 Should Work
1. ✅ Avoids V3's CLIP modification issue
2. ✅ Avoids V1's weak supervision issue (strong aux losses)
3. ✅ Preserves V1's FID benefit (clean clinical text)
4. ✅ Simple architecture (easy to debug)
5. ✅ Multiple forcing mechanisms (aux losses + optional dropout)

### Expected Results
- **FID**: ~82-85 (matching V1's improvement)
- **Demographic Accuracy**: >0.95 (with aux losses forcing)
- **Training**: Stable, converges in 10-15k steps

### Configs
- Config 1 (recommended first): No dropout, strong aux losses
- Config 2 (if accuracy poor): Adds 50% demographic dropout after 5k warmup steps

### Output Directories
- `output_v4/1_demographic_encoder_v4` - Base config
- `output_v4/2_demographic_encoder_v4_dropout` - With dropout


*Last updated: $(date +"%Y-%m-%d %H:%M:%S")*

