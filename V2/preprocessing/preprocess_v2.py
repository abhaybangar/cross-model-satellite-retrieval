import os
import numpy as np
import tifffile
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# V2 Preprocessing uses 1st and 99th percentiles for wider dynamic range clipping
def normalize_band_v2(band, p_min=1, p_max=99):
    """Normalize a single band using V2 percentiles to [0, 255] uint8."""
    b_min, b_max = np.percentile(band, [p_min, p_max])
    if b_max == b_min:
        return np.zeros_like(band, dtype=np.uint8)
    normalized = (band - b_min) / (b_max - b_min) * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)

def convert_optical_file_v2(path):
    # S2 bands index: 2=Red, 1=Green, 0=Blue
    img = tifffile.imread(path)
    r = normalize_band_v2(img[2])
    g = normalize_band_v2(img[1])
    b = normalize_band_v2(img[0])
    rgb = np.stack([r, g, b], axis=-1)
    Image.fromarray(rgb).save(path)

def convert_sar_file_v2(path):
    # S1 bands index: 0=VV, 1=VH
    img = tifffile.imread(path)
    vv = normalize_band_v2(img[0])
    vh = normalize_band_v2(img[1])
    # VV - VH difference (ratio in dB scale)
    vv_vh_diff = normalize_band_v2(img[0] - img[1])
    rgb = np.stack([vv, vh, vv_vh_diff], axis=-1)
    Image.fromarray(rgb).save(path)

def main():
    print("V2 Preprocessing script initialized.")
    print("This script defines V2 percentile scaling (1% to 99% percentile clipping).")
    print("Since train2 and test2 datasets are already preprocessed and provided on disk, "
          "no manual preprocessing is strictly required. This file is placeholder documentation.")

if __name__ == "__main__":
    main()
