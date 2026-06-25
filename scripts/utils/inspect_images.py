import os
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OPT_FOLDER = os.path.join(WORKSPACE, "dataset", "optical")
SAR_FOLDER = os.path.join(WORKSPACE, "dataset", "sar")

opt_files = os.listdir(OPT_FOLDER)
sar_files = os.listdir(SAR_FOLDER)

print(f"Optical files: {len(opt_files)}")
print(f"SAR files: {len(sar_files)}")

if opt_files:
    opt_img_path = os.path.join(OPT_FOLDER, opt_files[0])
    img = Image.open(opt_img_path)
    print(f"\nOptical Image ({opt_files[0]}):")
    print(f"  PIL Mode: {img.mode}")
    print(f"  Size: {img.size}")
    img_np = np.array(img)
    print(f"  Numpy Shape: {img_np.shape}")
    print(f"  Data Type: {img_np.dtype}")
    print(f"  Min Value: {img_np.min()}")
    print(f"  Max Value: {img_np.max()}")

if sar_files:
    sar_img_path = os.path.join(SAR_FOLDER, sar_files[0])
    img = Image.open(sar_img_path)
    print(f"\nSAR Image ({sar_files[0]}):")
    print(f"  PIL Mode: {img.mode}")
    print(f"  Size: {img.size}")
    img_np = np.array(img)
    print(f"  Numpy Shape: {img_np.shape}")
    print(f"  Data Type: {img_np.dtype}")
    print(f"  Min Value: {img_np.min()}")
    print(f"  Max Value: {img_np.max()}")
