import os
import sys
import shutil
import pandas as pd
import numpy as np
import tifffile
from PIL import Image

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATASET_DIR = os.path.join(WORKSPACE, "dataset")
BACKUP_DIR = os.path.join(WORKSPACE, "dataset_original")
KAGGLE_DATASET_ROOT = r"C:\Users\banga\.cache\kagglehub\datasets\narendraaironi\bigearthnet-14k\versions\1\BEN_14k"
CACHE_DIR = os.path.join(WORKSPACE, "backend", "cache")

def normalize_band(band, p_min=2, p_max=98):
    """Normalize a single band using percentiles to [0, 255] uint8."""
    b_min, b_max = np.percentile(band, [p_min, p_max])
    if b_max == b_min:
        return np.zeros_like(band, dtype=np.uint8)
    normalized = (band - b_min) / (b_max - b_min) * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)

def clear_cache():
    if os.path.exists(CACHE_DIR):
        print(f"Clearing backend cache at {CACHE_DIR}...")
        shutil.rmtree(CACHE_DIR)

def enable_bigearthnet(limit=100):
    print("Enabling BigEarthNet-14k dataset...")
    
    # Clear cache
    clear_cache()
    
    # 1. Back up current dataset if not already done
    if os.path.exists(DATASET_DIR):
        if not os.path.exists(BACKUP_DIR):
            print(f"Backing up current dataset to {BACKUP_DIR}...")
            os.rename(DATASET_DIR, BACKUP_DIR)
        else:
            print(f"Current dataset backup already exists at {BACKUP_DIR}. Removing active dataset folder...")
            shutil.rmtree(DATASET_DIR)
            
    # Create new clean dataset directory structure
    os.makedirs(os.path.join(DATASET_DIR, "optical"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "sar"), exist_ok=True)
    
    # 2. Load metadata
    metadata_path = os.path.join(KAGGLE_DATASET_ROOT, "metadata.parquet")
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.parquet not found at {metadata_path}")
        return
        
    print("Reading metadata.parquet...")
    df = pd.read_parquet(metadata_path)
    print(f"Metadata loaded. Total rows: {len(df)}")
    
    # 3. Iterate and copy/convert matching pairs
    count = 0
    # Group by split to look in correct folders
    for split in ["train", "validation", "test"]:
        if count >= limit:
            break
            
        split_df = df[df["split"] == split]
        print(f"Searching in split: {split} ({len(split_df)} candidate rows)...")
        
        s1_dir = os.path.join(KAGGLE_DATASET_ROOT, "BigEarthNet-S1", split)
        s2_dir = os.path.join(KAGGLE_DATASET_ROOT, "BigEarthNet-S2", split)
        
        if not os.path.exists(s1_dir) or not os.path.exists(s2_dir):
            print(f"Warning: split directory not found for {split}. Skipping.")
            continue
            
        for _, row in split_df.iterrows():
            if count >= limit:
                break
                
            patch_id = row["patch_id"]
            s1_name = row["s1_name"]
            
            s1_path = os.path.join(s1_dir, f"{s1_name}.tif")
            s2_path = os.path.join(s2_dir, f"{patch_id}.tif")
            
            if os.path.exists(s1_path) and os.path.exists(s2_path):
                try:
                    # Process Sentinel-2 (Optical) -> RGB PNG
                    s2_img = tifffile.imread(s2_path)
                    # S2 bands index: 2=Red, 1=Green, 0=Blue
                    r = normalize_band(s2_img[2])
                    g = normalize_band(s2_img[1])
                    b = normalize_band(s2_img[0])
                    s2_rgb = np.stack([r, g, b], axis=-1)
                    
                    # Process Sentinel-1 (SAR) -> False Color RGB PNG
                    s1_img = tifffile.imread(s1_path)
                    # S1 bands index: 0=VV, 1=VH
                    vv = normalize_band(s1_img[0])
                    vh = normalize_band(s1_img[1])
                    # use vv - vh for the third band
                    vv_vh_diff = normalize_band(s1_img[0] - s1_img[1])
                    s1_rgb = np.stack([vv, vh, vv_vh_diff], axis=-1)
                    
                    # Save both images as PNG with the same patch_id name
                    opt_out_path = os.path.join(DATASET_DIR, "optical", f"{patch_id}.png")
                    sar_out_path = os.path.join(DATASET_DIR, "sar", f"{patch_id}.png")
                    
                    Image.fromarray(s2_rgb).save(opt_out_path)
                    Image.fromarray(s1_rgb).save(sar_out_path)
                    
                    count += 1
                    if count % 10 == 0 or count == limit:
                        print(f"Processed {count}/{limit} pairs...")
                        
                except Exception as e:
                    print(f"Error processing patch {patch_id}: {e}")
                    
    print(f"\nSuccessfully enabled BigEarthNet-14k dataset with {count} pairs!")
    print(f"Images saved to: {DATASET_DIR}")

def disable_bigearthnet():
    print("Disabling BigEarthNet-14k dataset...")
    
    # Clear cache
    clear_cache()
    
    if os.path.exists(DATASET_DIR):
        print("Removing active dataset folder...")
        shutil.rmtree(DATASET_DIR)
        
    if os.path.exists(BACKUP_DIR):
        print(f"Restoring original dataset from {BACKUP_DIR}...")
        os.rename(BACKUP_DIR, DATASET_DIR)
        print("Successfully restored original dataset!")
    else:
        print("Original dataset backup not found. Clean slate created.")
        os.makedirs(os.path.join(DATASET_DIR, "optical"), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "sar"), exist_ok=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python manage_dataset.py [enable|disable] [limit]")
        sys.exit(1)
        
    cmd = sys.argv[1].lower()
    if cmd == "enable":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        enable_bigearthnet(limit)
    elif cmd == "disable":
        disable_bigearthnet()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
