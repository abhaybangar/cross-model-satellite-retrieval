# BigEarthNet-14K Preprocessed 1000-Pair Dataset

## Purpose

A **DINOv2-ready** preprocessed dataset of 1,000 paired Sentinel-2 (optical)
and Sentinel-1 (SAR) satellite images for cross-modal retrieval experiments.
All preprocessing is applied and saved -- load and feed directly to DINOv2.

| Property | Value |
|---|---|
| Total paired samples | 1000 |
| Image format | GeoTIFF (.tif), 3 channels, 224x224, float32 |
| Optical preprocessing | RGB bands, /10000, [0,1] clip, ImageNet norm |
| SAR preprocessing | VV+VH+VV-VH 3ch, float32 (dB, no extra scaling) |
| Country | Serbia |
| Season | Summer 2017 (Jul-Aug) |
| Prior patch_ids excluded | 4247 |

---

## Preprocessing Pipeline

### Optical (Sentinel-2) -- per the DINOv2 roadmap

1. **Band selection**: Extract RGB from 10-band S2 data
   - Band 2 (B04 Red), Band 1 (B03 Green), Band 0 (B02 Blue)
2. **Resize**: Bilinear interpolation from 120x120 to 224x224
3. **Float32 conversion**: Cast from uint16 to float32
4. **Scale to [0,1]**: Divide by 10,000 (Sentinel-2 L2A reflectance convention)
5. **Clip**: Ensure values are in [0, 1]
6. **ImageNet normalization**: Per-channel (x - mean) / std
   - Mean: [0.485, 0.456, 0.406]
   - Std:  [0.229, 0.224, 0.225]

### SAR (Sentinel-1) -- per the DINOv2 roadmap

1. **Band extraction**: VV (band 0) and VH (band 1), already float32 in dB
2. **3-channel construction**: Stack [VV, VH, VV-VH]
3. **Resize**: Bilinear interpolation from 120x120 to 224x224
4. **No extra scaling**: SAR data is already pre-scaled/normalized in dB;
   only slicing and tensor conversion applied per user instruction

> **Note**: The optical images have ImageNet normalization baked in. The SAR
> images are in raw dB scale. When feeding to DINOv2, the optical images
> are ready to go. For SAR, you may want to experiment with normalizing to
> match the optical distribution or using a separate projection head.

---

## Folder Structure

```
preprocessed_1000/
  optical/              1000 preprocessed S2 GeoTIFF files (3ch, 224x224, float32)
    img_0001.tif ... img_1000.tif
  sar/                  1000 preprocessed S1 GeoTIFF files (3ch, 224x224, float32)
    img_0001.tif ... img_1000.tif
  metadata.csv          1000 rows x 11 columns
  README.md             This file
```

---

## Metadata Columns

| Column | Description |
|---|---|
| `id` | img_0001 through img_1000 |
| `patch_id` | Original BigEarthNet-v2 identifier |
| `split` | Original BEN train/test/validation |
| `country` | Serbia |
| `labels` | Semicolon-separated land-cover labels |
| `optical_path` | Relative path to preprocessed S2 .tif |
| `sar_path` | Relative path to preprocessed S1 .tif |
| `acquisition_date` | YYYYMMDD |
| `acquisition_month` | 7 or 8 |
| `label_count` | Labels per sample |
| `label_group` | High-level groups |

---

## Label Group Coverage

| Group | Count | % |
|---|---|---|
| agriculture | 681 | 68.1% |
| forest | 620 | 62.0% |
| grassland_shrub | 431 | 43.1% |
| urban | 131 | 13.1% |
| water | 79 | 7.9% |

---

## Usage with DINOv2 + FAISS

```python
import torch, pandas as pd, numpy as np, rasterio
from torch.utils.data import Dataset, DataLoader

class PreprocessedDataset(Dataset):
    """Load already-preprocessed TIFFs. No further preprocessing needed."""
    def __init__(self, csv_path, root_dir):
        self.meta = pd.read_csv(csv_path)
        self.root = root_dir
    def __len__(self):
        return len(self.meta)
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        with rasterio.open(f"{self.root}/{row['optical_path']}") as src:
            optical = torch.tensor(src.read())  # (3, 224, 224) float32
        with rasterio.open(f"{self.root}/{row['sar_path']}") as src:
            sar = torch.tensor(src.read())      # (3, 224, 224) float32
        return optical, sar, row['id']

# Load DINOv2 -- images are ALREADY ImageNet-normalized
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
backbone.eval()

ds = PreprocessedDataset("preprocessed_1000/metadata.csv", "preprocessed_1000/")
loader = DataLoader(ds, batch_size=32)

# Extract embeddings directly -- no transform needed for optical
embeddings = []
for optical, sar, ids in loader:
    with torch.no_grad():
        emb = backbone(optical)  # Already preprocessed!
    embeddings.append(emb.numpy())

import faiss
emb = np.vstack(embeddings)
emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype(np.float32))
```

---

## Leakage Prevention

- 4247 patch_ids from prior builds excluded before sampling
- No overlap with: prototype_dataset, generalization_dataset, cross_modal_dataset

## Assumptions & Limitations

1. S2 band order assumed: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
2. S1 band order assumed: VV (band 0), VH (band 1)
3. Optical scaled by /10000 (standard Sentinel-2 L2A convention)
4. SAR left in dB -- no ImageNet normalization applied to SAR per user instruction
5. Resize from 120x120 to 224x224 via bilinear interpolation
6. Serbia-only, Summer 2017 only
7. Source files untouched

## Reproducibility
- Seed: 77
- Filter: Serbia + months 6,7,8
- Prior exclusions: 4247 patch_ids
