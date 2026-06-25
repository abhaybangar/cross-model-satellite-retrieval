import os
import numpy as np
import tifffile
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OPT_DIR = os.path.join(WORKSPACE, "dataset", "test2", "optical")
SAR_DIR = os.path.join(WORKSPACE, "dataset", "test2", "sar")

def normalize_band(band, p_min=2, p_max=98):
    """Normalize a single band using percentiles to [0, 255] uint8."""
    b_min, b_max = np.percentile(band, [p_min, p_max])
    if b_max == b_min:
        return np.zeros_like(band, dtype=np.uint8)
    normalized = (band - b_min) / (b_max - b_min) * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)

def convert_optical_file(path):
    # S2 bands index: 2=Red, 1=Green, 0=Blue
    img = tifffile.imread(path)
    r = normalize_band(img[2])
    g = normalize_band(img[1])
    b = normalize_band(img[0])
    rgb = np.stack([r, g, b], axis=-1)
    Image.fromarray(rgb).save(path)

def convert_sar_file(path):
    # S1 bands index: 0=VV, 1=VH
    img = tifffile.imread(path)
    vv = normalize_band(img[0])
    vh = normalize_band(img[1])
    # VV - VH difference (ratio in dB scale)
    vv_vh_diff = normalize_band(img[0] - img[1])
    rgb = np.stack([vv, vh, vv_vh_diff], axis=-1)
    Image.fromarray(rgb).save(path)

def main():
    print("Converting test2 optical files to standard RGB TIFFs...")
    opt_files = [f for f in os.listdir(OPT_DIR) if f.endswith(".tif")]
    total = len(opt_files)
    for idx, filename in enumerate(opt_files):
        path = os.path.join(OPT_DIR, filename)
        try:
            convert_optical_file(path)
        except Exception as e:
            print(f"Error converting optical {filename}: {e}")
        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"Optical progress: {idx + 1}/{total}")

    print("\nConverting test2 SAR files to standard RGB TIFFs...")
    sar_files = [f for f in os.listdir(SAR_DIR) if f.endswith(".tif")]
    total = len(sar_files)
    for idx, filename in enumerate(sar_files):
        path = os.path.join(SAR_DIR, filename)
        try:
            convert_sar_file(path)
        except Exception as e:
            print(f"Error converting SAR {filename}: {e}")
        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"SAR progress: {idx + 1}/{total}")

    print("\nIn-place conversion of test2 dataset finished successfully!")

if __name__ == "__main__":
    main()
