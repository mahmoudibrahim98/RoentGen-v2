"""
Create WebDataset for validation with all necessary metadata.

This script creates validation tar files that include:
1. Image tensors (.pt_image)
2. Prompt text (.prompt_metadata)  
3. Validation metadata (.validation_metadata) - NEW!
   - Disease labels (5 diseases)
   - Demographics (age, sex, race)
   - Any other metadata needed for validation metrics

The format is compatible with the validation system in train_loop.py.
"""

import os
import pickle
from PIL import Image
from tqdm import tqdm
import numpy as np
import tarfile
import io
import json
from pathlib import Path
import torch
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

# Source data
source_dir = Path("/home/vito/ibrahimm/projects/AI4Health/sourcedata/Chest Xray/physionet.org/files/mimic-cxr-jpg/2.0.0/")
img_dir = source_dir / "files"

# Output directories
real_data_dir = Path("/home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/real_data")

split_to_dir = {
    "train": real_data_dir / "training_data",
    "val": real_data_dir / "val_data",
    "test": real_data_dir / "test_data",
}

# Create directories
for d in split_to_dir.values():
    os.makedirs(d, exist_ok=True)

# Maximum samples per tar file
max_samples_per_tar = 1000

# Disease labels to extract (modify based on your data)
DISEASE_COLUMNS = [
    'Atelectasis', 'Cardiomegaly',
       'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
       'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
       'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices',
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_disease_labels(row, disease_columns):
    """
    Extract disease labels from dataframe row.
    
    Returns:
        list of floats: Binary labels (1.0 for positive, 0.0 for negative, -1.0 for uncertain/missing)
    """
    labels = []
    for disease in disease_columns:
        if disease in row:
            val = row[disease]
            # Handle different formats: 1/0, True/False, 1.0/0.0/-1.0
            if pd.isna(val):
                labels.append(-1.0)
            elif val == 1 or val == 1.0 or val == True:
                labels.append(1.0)
            elif val == 0 or val == 0.0 or val == False:
                labels.append(0.0)
            else:
                labels.append(-1.0)  # Uncertain
        else:
            labels.append(-1.0)  # Missing
    return labels


def extract_demographics(row):
    """
    Extract demographic information from dataframe row.
    
    Returns:
        dict with keys: age, sex, race, sex_idx, race_idx, age_bin
    """
    demographics = {}
    
    # Age
    if 'anchor_age' in row:
        demographics['age'] = float(row['anchor_age']) if not pd.isna(row['anchor_age']) else -1.0
    else:
        demographics['age'] = -1.0
    
    # Age bin (for HCN)
    age = demographics['age']
    if age < 0:
        demographics['age_bin'] = -1
    elif age < 18:
        demographics['age_bin'] = 0
    elif age < 40:
        demographics['age_bin'] = 1
    elif age < 60:
        demographics['age_bin'] = 2
    elif age < 80:
        demographics['age_bin'] = 3
    else:
        demographics['age_bin'] = 4
    
    # Sex
    if 'gender' in row:
        sex = str(row['gender']).upper()
        if sex in ['M', 'MALE']:
            demographics['sex'] = 'M'
            demographics['sex_idx'] = 0
        elif sex in ['F', 'FEMALE']:
            demographics['sex'] = 'F'
            demographics['sex_idx'] = 1
        else:
            demographics['sex'] = 'UNKNOWN'
            demographics['sex_idx'] = -1
    else:
        demographics['sex'] = 'UNKNOWN'
        demographics['sex_idx'] = -1
    
    # Race (modify categories based on your data)
    if 'ethnicity' in row:
        race = str(row['ethnicity']).upper()
        # Map to indices (modify based on your data)
        race_map = {
            'WHITE': 0,
            'BLACK': 1,
            'ASIAN': 2,
            'HISPANIC': 3,
            'OTHER': 3,
            'UNKNOWN': -1
        }
        demographics['race'] = race
        demographics['race_idx'] = race_map.get(race, -1)
    else:
        demographics['race'] = 'UNKNOWN'
        demographics['race_idx'] = -1
    
    return demographics


def create_validation_metadata(row, disease_columns):
    """
    Create complete validation metadata dictionary.
    
    Returns:
        dict containing all metadata needed for validation
    """
    metadata = {}
    
    # Disease labels
    metadata['disease_labels'] = extract_disease_labels(row, disease_columns)
    
    # Demographics
    demographics = extract_demographics(row)
    metadata.update(demographics)
    
    # Additional metadata (optional)
    if 'study_id' in row:
        metadata['study_id'] = str(row['study_id'])
    if 'subject_id' in row:
        metadata['subject_id'] = str(row['subject_id'])
    if 'image_id' in row:
        metadata['image_id'] = str(row['image_id'])
    
    return metadata


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def create_webdataset_with_validation_metadata(PA_data, split_to_create="val"):
    """
    Create WebDataset tar files with validation metadata.
    
    Args:
        PA_data: DataFrame with all the data
        split_to_create: Which split to create ("train", "val", or "test")
    """
    print(f"\n{'='*60}")
    print(f"Creating WebDataset for split: {split_to_create}")
    print(f"{'='*60}\n")
    
    # Filter data for this split
    data_split = PA_data[PA_data["split"] == split_to_create]
    data_split = data_split.reset_index(drop=True)
    
    print(f"Total samples in {split_to_create}: {len(data_split)}")
    
    total_count = 0
    tar_sample_count = 0
    tar_idx = 0
    tar = None
    tar_path = None
    
    # Iterate through samples
    for idx, row in tqdm(data_split.iterrows(), total=len(data_split), desc=f"Processing {split_to_create}"):
        
        # Open new tar if needed
        if tar_sample_count == 0:
            tar_path = split_to_dir[split_to_create] / f"{split_to_create}_{tar_idx}.tar"
            tar = tarfile.open(tar_path, "w")
            print(f"\nOpened new tar: {tar_path}")
        
        # Get image path
        img_path = Path(row['image'])
        if not img_path.exists():
            print(f"Missing image file: {img_path}")
            continue
        
        try:
            # ================================================================
            # 1. PROCESS IMAGE - PyTorch tensor (1, H, W), float32, [0, 1]
            # ================================================================
            with Image.open(img_path) as img:
                img = img.convert("L")  # Grayscale
                arr = np.array(img)
                arr = arr.astype(np.float32) / 255.0  # Normalize to [0, 1]
            
            # Convert to tensor (1, H, W)
            arr_tensor = torch.from_numpy(arr)
            
            # Serialize image tensor
            img_bytes = io.BytesIO()
            pickle.dump(arr_tensor, img_bytes)
            img_bytes.seek(0)
            
            # ================================================================
            # 2. PROCESS PROMPT - UTF-8 text
            # ================================================================
            prompt_bytes = row['sentence'].encode("utf-8")
            prompt_stream = io.BytesIO(prompt_bytes)
            
            # ================================================================
            # 3. CREATE VALIDATION METADATA - JSON
            # ================================================================
            validation_metadata = create_validation_metadata(row, DISEASE_COLUMNS)
            
            # Serialize metadata as JSON
            metadata_json = json.dumps(validation_metadata)
            metadata_bytes = metadata_json.encode("utf-8")
            metadata_stream = io.BytesIO(metadata_bytes)
            
            # ================================================================
            # 4. ADD ALL FILES TO TAR
            # ================================================================
            tar_key = f"{tar_sample_count+1:06d}"
            
            # Add image tensor
            ptinfo = tarfile.TarInfo(f"{tar_key}.pt_image")
            ptinfo.size = img_bytes.getbuffer().nbytes
            img_bytes.seek(0)
            tar.addfile(ptinfo, img_bytes)
            
            # Add prompt
            promptinfo = tarfile.TarInfo(f"{tar_key}.prompt_metadata")
            promptinfo.size = prompt_stream.getbuffer().nbytes
            prompt_stream.seek(0)
            tar.addfile(promptinfo, prompt_stream)
            
            # Add validation metadata (NEW!)
            metainfo = tarfile.TarInfo(f"{tar_key}.validation_metadata")
            metainfo.size = metadata_stream.getbuffer().nbytes
            metadata_stream.seek(0)
            tar.addfile(metainfo, metadata_stream)
            
            tar_sample_count += 1
            total_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # ================================================================
        # 5. CLOSE TAR IF FULL
        # ================================================================
        if tar_sample_count >= max_samples_per_tar:
            tar.close()
            
            # Write size file
            size_txt_path = split_to_dir[split_to_create] / f"{split_to_create}_{tar_idx}_size.txt"
            with open(size_txt_path, "w") as f:
                f.write(str(tar_sample_count) + "\n")
            
            print(f"Closed {tar_path} with {tar_sample_count} samples")
            
            tar_sample_count = 0
            tar_idx += 1
    
    # ================================================================
    # 6. CLOSE FINAL TAR
    # ================================================================
    if tar_sample_count > 0 and tar is not None:
        tar.close()
        
        size_txt_path = split_to_dir[split_to_create] / f"{split_to_create}_{tar_idx}_size.txt"
        with open(size_txt_path, "w") as f:
            f.write(str(tar_sample_count) + "\n")
        
        print(f"Closed {tar_path} with {tar_sample_count} samples")
    
    print(f"\n{'='*60}")
    print(f"Split {split_to_create}: {total_count} samples in {tar_idx+1} tar file(s)")
    print(f"{'='*60}\n")
    
    # ================================================================
    # 7. VERIFY TAR FILES
    # ================================================================
    print(f"\nVerifying tar files for {split_to_create}:")
    splitdir = split_to_dir[split_to_create]
    size_files = sorted(splitdir.glob(f"{split_to_create}_*_size.txt"))
    total_in_split = 0
    
    for sizefile in size_files:
        try:
            with open(sizefile) as f:
                count = int(f.read().strip())
                total_in_split += count
                print(f"  {sizefile.name}: {count} samples")
        except Exception as e:
            print(f"  {sizefile.name}: Error reading: {e}")
    
    print(f"\nTotal samples verified: {total_in_split}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Load your PA_data DataFrame
    # Example: PA_data = pd.read_csv("your_data.csv")
    
    # For demonstration, assuming PA_data exists
    # PA_data should have columns:
    # - 'split': 'train', 'val', or 'test'
    # - 'image': path to image file
    # - 'sentence': prompt text
    # - 'Atelectasis', 'Cardiomegaly', etc.: disease labels
    # - 'age', 'sex', 'race': demographics
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║  Validation Dataset Creator for RoentGen-v2                        ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    This script creates WebDataset tar files with complete validation metadata.
    
    Required columns in PA_data:
    - split: 'train', 'val', or 'test'
    - image: path to image file
    - sentence: prompt text
    - Atelectasis, Cardiomegaly, Edema, Pneumothorax, Pleural Effusion: disease labels
    - age, sex, race: demographic attributes
    
    Output format per sample:
    - {key}.pt_image: PyTorch tensor (1, H, W), float32, [0, 1]
    - {key}.prompt_metadata: UTF-8 text prompt
    - {key}.validation_metadata: JSON with disease labels & demographics
    
    """)
    
    # Uncomment and modify based on your data loading:
    PA_data = pd.read_csv("/home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/src/preprocessing/PA_data.csv")
    
    # Create validation dataset
    create_webdataset_with_validation_metadata(PA_data, split_to_create="val")
    
    # Optionally also create test dataset
    create_webdataset_with_validation_metadata(PA_data, split_to_create="test")
    
    # For training data, you may not need all validation metadata
    create_webdataset_with_validation_metadata(PA_data, split_to_create="train")
    
    print("\n✓ Script ready. Uncomment the lines above and provide your PA_data DataFrame.")



