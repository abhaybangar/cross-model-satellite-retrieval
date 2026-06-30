"""
ben_preprocess.py
=================
Reusable preprocessing module for Sentinel-2 (optical) and Sentinel-1 (SAR)
GeoTIFF imagery, targeting DINOv2 input requirements.

Pipeline:
  Optical: 10-band S2 → RGB (B04, B03, B02) → resize 224×224 → /10000 → [0,1] → ImageNet norm
  SAR:     2-band S1 → [VV, VH, VV−VH] → resize 224×224 → float32 (native dB, no extra scaling)

Dependencies: rasterio, numpy, scipy
"""

import numpy as np
import rasterio
from scipy.ndimage import zoom as _scipy_zoom

# ── Constants ────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# BigEarthNet-v2 Sentinel-2 band order:
#   0:B02(Blue) 1:B03(Green) 2:B04(Red) 3:B05 4:B06 5:B07 6:B08 7:B8A 8:B11 9:B12
S2_RGB_INDICES = [2, 1, 0]  # R=B04, G=B03, B=B02

S2_REFLECTANCE_SCALE = 10_000.0
TARGET_SIZE = 224


# ── Core functions ───────────────────────────────────────────────────

def inspect_tiff(path: str) -> dict:
    """Return band count, shape, dtype, CRS, and per-band statistics."""
    with rasterio.open(path) as src:
        data = src.read()
        info = {
            "path": path,
            "bands": src.count,
            "height": src.height,
            "width": src.width,
            "dtype": str(data.dtype),
            "crs": str(src.crs) if src.crs else None,
            "descriptions": src.descriptions,
            "band_stats": [],
        }
        for b in range(src.count):
            band = data[b]
            info["band_stats"].append({
                "band": b,
                "min": float(band.min()),
                "max": float(band.max()),
                "mean": float(band.mean()),
                "std": float(band.std()),
            })
    return info


def resize_to_224(arr: np.ndarray, target: int = TARGET_SIZE) -> np.ndarray:
    """Bilinear resize a (C, H, W) array to (C, target, target).

    Uses scipy.ndimage.zoom (order=1 = bilinear). Fast and accurate.
    """
    _, h, w = arr.shape
    if h == target and w == target:
        return arr
    return _scipy_zoom(arr, (1, target / h, target / w), order=1).astype(arr.dtype)


def preprocess_optical(
    source,
    *,
    rgb_indices: list[int] = S2_RGB_INDICES,
    scale: float = S2_REFLECTANCE_SCALE,
    imagenet_norm: bool = True,
    target_size: int = TARGET_SIZE,
) -> np.ndarray:
    """Preprocess a Sentinel-2 optical TIFF for DINOv2.
    Detects if the input is a standard uint8 image or raw uint16 reflectance,
    and applies correct scaling & ImageNet normalization.
    """
    from PIL import Image
    if isinstance(source, (str, bytes)):
        try:
            img = Image.open(source).convert("RGB")
            img = img.resize((target_size, target_size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)  # HWC to CHW
            if imagenet_norm:
                mean = IMAGENET_MEAN.reshape(3, 1, 1)
                std = IMAGENET_STD.reshape(3, 1, 1)
                arr = (arr - mean) / std
            return arr
        except Exception:
            with rasterio.open(source) as src:
                data = src.read()
    else:
        data = source

    if len(data.shape) == 3:
        if data.shape[0] >= 3:
            rgb = data[rgb_indices].astype(np.float32)
        else:
            rgb = data.astype(np.float32)
    else:
        rgb = np.stack([data, data, data], axis=0).astype(np.float32)

    rgb = resize_to_224(rgb, target_size)

    scale_to_use = 255.0 if data.dtype == np.uint8 else scale
    rgb /= scale_to_use
    np.clip(rgb, 0.0, 1.0, out=rgb)

    if imagenet_norm:
        for c in range(3):
            rgb[c] = (rgb[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

    return rgb


def preprocess_sar(
    source,
    *,
    target_size: int = TARGET_SIZE,
) -> np.ndarray:
    """Preprocess a Sentinel-1 SAR TIFF for DINOv2.
    Adapts to uint8 3-channel false-color images on disk, or raw 2-band files.
    """
    from PIL import Image
    if isinstance(source, (str, bytes)):
        try:
            img = Image.open(source)
            if img.mode == "RGB" or img.mode == "RGBA":
                img = img.convert("RGB").resize((target_size, target_size), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = arr.transpose(2, 0, 1)  # HWC to CHW
                mean = IMAGENET_MEAN.reshape(3, 1, 1)
                std = IMAGENET_STD.reshape(3, 1, 1)
                arr = (arr - mean) / std
                return arr
        except Exception:
            pass
            
        with rasterio.open(source) as src:
            data = src.read()
    else:
        data = source

    if data.shape[0] >= 2:
        vv = data[0].astype(np.float32)
        vh = data[1].astype(np.float32)
        sar_3ch = np.stack([vv, vh, vv - vh], axis=0)
    else:
        sar_3ch = np.stack([data[0], data[0], data[0]], axis=0).astype(np.float32)

    sar_3ch = resize_to_224(sar_3ch, target_size)

    if data.dtype == np.uint8:
        sar_3ch /= 255.0
        np.clip(sar_3ch, 0.0, 1.0, out=sar_3ch)
        for c in range(3):
            sar_3ch[c] = (sar_3ch[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

    return sar_3ch


def save_tiff(
    data: np.ndarray,
    path: str,
    *,
    crs=None,
    compress: str = "lzw",
) -> None:
    """Save a (C, H, W) float32 array as a GeoTIFF.

    Args:
        data: Array of shape (C, H, W).
        path: Output file path.
        crs: Optional CRS to embed.
        compress: Compression method. Default: lzw.
    """
    c, h, w = data.shape
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": w,
        "height": h,
        "count": c,
        "compress": compress,
    }
    if crs:
        profile["crs"] = crs

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))


def preprocess_pair(
    optical_path: str,
    sar_path: str,
    *,
    optical_out: str | None = None,
    sar_out: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess a matched optical/SAR pair. Optionally save to disk.

    Returns:
        (optical_array, sar_array) — both (3, 224, 224) float32.
    """
    opt = preprocess_optical(optical_path)
    sar = preprocess_sar(sar_path)

    if optical_out:
        save_tiff(opt, optical_out)
    if sar_out:
        save_tiff(sar, sar_out)

    return opt, sar


# ── CLI ──────────────────────────────────────────────────────────────

def _print_stats(label, arr):
    ch_names = {
        "optical": ["R (B04)", "G (B03)", "B (B02)"],
        "sar": ["VV", "VH", "VV-VH"],
    }
    names = ch_names.get(label, [f"Band {i}" for i in range(arr.shape[0])])
    print(f"  {label}: shape={arr.shape}  dtype={arr.dtype}")
    for i, name in enumerate(names):
        b = arr[i]
        print(f"    {name:8s}: [{b.min():.4f}, {b.max():.4f}]  mean={b.mean():.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess Sentinel-2/S1 TIFFs for DINOv2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect a TIFF
  python ben_preprocess.py inspect image.tif

  # Preprocess an optical TIFF and save
  python ben_preprocess.py optical input.tif output.tif

  # Preprocess a SAR TIFF (print stats only)
  python ben_preprocess.py sar input.tif

  # Preprocess a matched pair
  python ben_preprocess.py pair optical.tif sar.tif --out-optical opt.tif --out-sar sar.tif
""",
    )
    sub = parser.add_subparsers(dest="command")

    # inspect
    p_inspect = sub.add_parser("inspect", help="Inspect TIFF band structure")
    p_inspect.add_argument("input", help="Input TIFF path")

    # optical
    p_opt = sub.add_parser("optical", help="Preprocess optical TIFF")
    p_opt.add_argument("input", help="Input S2 TIFF (10-band)")
    p_opt.add_argument("output", nargs="?", help="Output TIFF path (optional)")
    p_opt.add_argument("--no-imagenet-norm", action="store_true")

    # sar
    p_sar = sub.add_parser("sar", help="Preprocess SAR TIFF")
    p_sar.add_argument("input", help="Input S1 TIFF (2-band)")
    p_sar.add_argument("output", nargs="?", help="Output TIFF path (optional)")

    # pair
    p_pair = sub.add_parser("pair", help="Preprocess matched optical/SAR pair")
    p_pair.add_argument("optical", help="Input S2 TIFF")
    p_pair.add_argument("sar", help="Input S1 TIFF")
    p_pair.add_argument("--out-optical", help="Output optical TIFF")
    p_pair.add_argument("--out-sar", help="Output SAR TIFF")

    args = parser.parse_args()

    if args.command == "inspect":
        info = inspect_tiff(args.input)
        print(f"File: {info['path']}")
        print(f"  Bands: {info['bands']}  Size: {info['height']}x{info['width']}  Dtype: {info['dtype']}")
        print(f"  CRS: {info['crs']}")
        print(f"  Descriptions: {info['descriptions']}")
        for s in info["band_stats"]:
            print(f"  Band {s['band']}: [{s['min']:.4f}, {s['max']:.4f}]  mean={s['mean']:.4f}  std={s['std']:.4f}")

    elif args.command == "optical":
        result = preprocess_optical(args.input, imagenet_norm=not args.no_imagenet_norm)
        _print_stats("optical", result)
        if args.output:
            save_tiff(result, args.output)
            print(f"  Saved: {args.output}")

    elif args.command == "sar":
        result = preprocess_sar(args.input)
        _print_stats("sar", result)
        if args.output:
            save_tiff(result, args.output)
            print(f"  Saved: {args.output}")

    elif args.command == "pair":
        opt, sar = preprocess_pair(
            args.optical, args.sar,
            optical_out=args.out_optical,
            sar_out=args.out_sar,
        )
        _print_stats("optical", opt)
        _print_stats("sar", sar)
        if args.out_optical:
            print(f"  Saved optical: {args.out_optical}")
        if args.out_sar:
            print(f"  Saved SAR: {args.out_sar}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
