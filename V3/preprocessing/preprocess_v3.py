"""
preprocess_v3.py
================
CLIP-compatible preprocessing for RemoteCLIP V3 pipeline.
Handles both Sentinel-2 optical and Sentinel-1 SAR GeoTIFFs.

Unlike ben_preprocess.py (ImageNet norm), this uses CLIP normalization:
  mean=[0.48145466, 0.4578275, 0.40821073]
  std =[0.26862954, 0.26130258, 0.27577711]
"""

import numpy as np
import rasterio
from scipy.ndimage import zoom as _scipy_zoom

# CLIP normalization stats
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

TARGET_SIZE = 224

# S2 band order: 0:B02 1:B03 2:B04 3:B05 4:B06 5:B07 6:B08 7:B8A 8:B11 9:B12
S2_RGB_INDICES = [2, 1, 0]  # R=B04, G=B03, B=B02


def _resize_band(arr: np.ndarray, target: int) -> np.ndarray:
    """Resize a 2D array to (target, target) using scipy zoom."""
    if arr.shape == (target, target):
        return arr
    zy = target / arr.shape[0]
    zx = target / arr.shape[1]
    return _scipy_zoom(arr.astype(np.float32), (zy, zx), order=1)


def _clip_normalize(rgb_hwc: np.ndarray) -> np.ndarray:
    """Apply CLIP mean/std normalization and return [3, H, W] float32 tensor."""
    arr = rgb_hwc.astype(np.float32)
    arr = (arr - CLIP_MEAN) / CLIP_STD
    return arr.transpose(2, 0, 1)  # [3, 224, 224]


def preprocess_optical_clip(path: str) -> np.ndarray:
    """
    Sentinel-2 GeoTIFF → CLIP-normalized [3, 224, 224] float32.
    Reads R/G/B bands, resizes, scales to [0,1], applies CLIP norm.
    """
    with rasterio.open(path) as src:
        n_bands = src.count
        if n_bands >= 10:
            bands = [src.read(i + 1).astype(np.float32) for i in S2_RGB_INDICES]
        elif n_bands == 3:
            bands = [src.read(i + 1).astype(np.float32) for i in range(3)]
        else:
            bands = [src.read(1).astype(np.float32)] * 3

    resized = np.stack([_resize_band(b, TARGET_SIZE) for b in bands], axis=-1)  # [H,W,3]

    # Scale to [0, 1]: percentile clip then normalize
    p2  = np.percentile(resized, 2)
    p98 = np.percentile(resized, 98)
    if p98 > p2:
        resized = np.clip((resized - p2) / (p98 - p2), 0, 1)
    else:
        resized = np.clip(resized / 10000.0, 0, 1)

    return _clip_normalize(resized)


def preprocess_sar_clip(path: str) -> np.ndarray:
    """
    Sentinel-1 GeoTIFF → CLIP-normalized [3, 224, 224] float32.
    Reads VV/VH bands, creates [VV, VH, VV-VH], scales to [0,1], applies CLIP norm.
    """
    with rasterio.open(path) as src:
        n_bands = src.count
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32) if n_bands >= 2 else vv.copy()

    vv_r = _resize_band(vv, TARGET_SIZE)
    vh_r = _resize_band(vh, TARGET_SIZE)
    diff = vv_r - vh_r

    channels = np.stack([vv_r, vh_r, diff], axis=-1)  # [H,W,3]

    # Scale each channel to [0, 1] independently
    for c in range(3):
        ch = channels[:, :, c]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        if hi > lo:
            channels[:, :, c] = np.clip((ch - lo) / (hi - lo), 0, 1)
        else:
            channels[:, :, c] = 0.0

    return _clip_normalize(channels)
