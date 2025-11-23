
# Run Training

```bash
sbatch submit_training.sh
```

# Run Validation

```bash
sbatch submit_validation.sh /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/configs/1_train_baseline.yaml 10 validation_manifest.json 0
```
Run valdiaiton with stop_at_step option :

 sbatch submit_validation.sh CONFIG_PATH CHECK_INTERVAL MANIFEST_FILE LOAD_IMAGES_FLAG STOP_AT_STEP

```bash
sbatch submit_validation_0.sh \
  /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/configs/0_train_full_hcn.yaml \
  10 \
  validation_manifest.json \
  0 \
  37500
```


# Python Metric plotting
```bash
python plot_validation_metrics.py --manifest /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/outputs/output_v1/1_train_full_hcn_rerun/validation_manifest.json --output_dir /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/outputs/output_v1/1_train_full_hcn_rerun/
```

# Comaprison

```bash

python plot_model_comparison.py --step 12500 --output_dir output/model_comparison --validation_data /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/real_data/val_data.csv
```

# create batch script
./create_submit_script.sh {train} or {val}  {exp_name}

```bash

./create_submit_script.sh train v1.5/1_hcn_with_aux_loss_and_dropout

```

# Extract All Results

```bash

python extract_validation_metrics.py
```


sbatch submit_jobs/v4/submit_validation_2_demographic_encoder_full_dropout.sh \
  /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/configs/v4/2_train_demographic_encoder_v4_full_dropout.yaml \
  300 \
  validation_manifest.json \
  0 


python plot_validation_metrics.py --manifest /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/outputs/output_v1.5/1_hcn_with_dropout_no_aux_loss/validation_manifest.json --output_dir /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/outputs/output_v1.5/1_hcn_with_dropout_no_aux_loss/


python plot_model_comparison.py --step 2500 --output_dir output_v2/model_comparison --validation_data /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/real_data/val_data.csv



./create_submit_script.sh train v1.5/1_hcn_with_aux_loss_and_dropout


3_train_demographic_encoder_v4_with_dropout