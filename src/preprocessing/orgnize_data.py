""" Script to organise and downsampling the images from the dataset.

During resizing, the smaller edge of the image will be matched to this number.
Uses parallel processing for faster execution on large datasets.
"""
from pathlib import Path
from torchvision.transforms import Resize
from torchvision.datasets.folder import default_loader
import zipfile
import argparse
import sys
import multiprocessing as mp
from functools import partial
import time
from tqdm import tqdm
import os



def process_single_image(image_file, source_dir, raw_dir, resize_size):
    """Process a single image file."""
    
    try:
        # Load and resize image
        image = default_loader(image_file)
        resize_fn = Resize(resize_size)
        new_image = resize_fn(image)
        
        # Create output directory structure
        relative_path = image_file.relative_to(source_dir)
        new_dir = raw_dir / relative_path.parent
        new_dir.mkdir(parents=True, exist_ok=True)
        
        # Save resized image
        output_path = new_dir / image_file.name
        new_image.save(output_path)
        
        return True, str(image_file)
    except Exception as e:
        return False, f"Error processing {image_file}: {e}"

source_dir = Path("/home/vito/ibrahimm/projects/AI4Health/sourcedata/Chest Xray/physionet.org/files/mimic-cxr-jpg/2.0.0/")
raw_dir = Path("/home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/my_work/raw_data")
resize_size = 512
num_workers = None
batch_size = 10000  
# Check if source directory exists
if not source_dir.exists():
    print(f"Error: Source directory {source_dir} does not exist!")
    sys.exit(1)

# Set number of workers
if num_workers is None:
    num_workers = mp.cpu_count()

print(f"Processing images from: {source_dir}")
print(f"Output directory: {raw_dir}")
print(f"Resize size: {resize_size}")
print(f"Using {num_workers} parallel workers")
print(f"Batch size: {batch_size}")

# Create output directory if it doesn't exist
raw_dir.mkdir(parents=True, exist_ok=True)

# Find all JPG files
print("Scanning for JPG files...")
jpg_files = list(source_dir.glob('**/*.jpg'))
total_files = len(jpg_files)
print(f"Found {total_files} JPG files to process")

if total_files == 0:
    print("No JPG files found in source directory!")
    



# Process images in parallel
start_time = time.time()
successful_count = 0
error_count = 0

# Create partial function with fixed arguments
process_func = partial(process_single_image, 
                        source_dir=source_dir, 
                        raw_dir=raw_dir, 
                        resize_size=resize_size)

# Process in batches to avoid memory issues
for i in range(0, total_files, batch_size):
    batch_files = jpg_files[i:i + batch_size]
    batch_num = i // batch_size + 1
    total_batches = (total_files + batch_size - 1) // batch_size
    
    print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_files)} images)")
    
    # Use multiprocessing pool
    with mp.Pool(processes=num_workers) as pool:
        # Use tqdm for progress bar
        results = list(tqdm(
            pool.imap(process_func, batch_files),
            total=len(batch_files),
            desc=f"Batch {batch_num}/{total_batches}"
        ))
    
    # Count results
    for success, message in results:
        if success:
            successful_count += 1
        else:
            error_count += 1
            print(f"  {message}")

end_time = time.time()
processing_time = end_time - start_time

print(f"\n{'='*50}")
print(f"Processing complete!")
print(f"Total time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
print(f"Successfully processed: {successful_count}/{total_files} images")
print(f"Errors: {error_count}")
print(f"Average time per image: {processing_time/total_files:.3f} seconds")
print(f"Images per second: {total_files/processing_time:.1f}")
