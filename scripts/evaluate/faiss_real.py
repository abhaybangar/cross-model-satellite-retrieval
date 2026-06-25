from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from transformers import AutoImageProcessor, AutoModel
import torch
import os
import numpy as np
import faiss

print("Loading DINOv2...")

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OPT_FOLDER = os.path.join(WORKSPACE, "dataset", "optical")
SAR_FOLDER = os.path.join(WORKSPACE, "dataset", "sar")

# ==========================
# FIND MATCHING FILES
# ==========================

opt_files = set(os.listdir(OPT_FOLDER))
sar_files = set(os.listdir(SAR_FOLDER))

matching_files = sorted(list(opt_files.intersection(sar_files)))

print(f"\nMatching Pairs Found: {len(matching_files)}")

# ==========================
# BUILD SAR EMBEDDINGS
# ==========================

sar_embeddings = []
sar_names = []

print("\nGenerating SAR Embeddings...")

for filename in matching_files:

    image_path = os.path.join(SAR_FOLDER, filename)

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)

    sar_embeddings.append(
        embedding.squeeze().numpy()
    )

    sar_names.append(filename)

sar_embeddings = np.array(sar_embeddings).astype("float32")

print("SAR Embeddings Shape:", sar_embeddings.shape)

# ==========================
# BUILD FAISS INDEX
# ==========================

index = faiss.IndexFlatL2(768)
index.add(sar_embeddings)

print("FAISS Index Ready")
print("Total SAR Images:", index.ntotal)

# ==========================
# EVALUATION
# ==========================

top1 = 0
top3 = 0
top5 = 0

total = len(matching_files)

print("\nStarting Evaluation...\n")

for filename in matching_files:

    image_path = os.path.join(OPT_FOLDER, filename)

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    query_embedding = (
        outputs
        .last_hidden_state
        .mean(dim=1)
    )

    query = query_embedding.numpy().astype("float32")

    distances, indices = index.search(query, 5)

    retrieved = [
        sar_names[idx]
        for idx in indices[0]
    ]

    # Top 1
    if filename == retrieved[0]:
        top1 += 1

    # Top 3
    if filename in retrieved[:3]:
        top3 += 1

    # Top 5
    if filename in retrieved[:5]:
        top5 += 1

print("\n" + "=" * 50)

print(f"Total Queries : {total}")

print(f"Top-1 Accuracy : {(top1/total)*100:.2f}%")
print(f"Top-3 Accuracy : {(top3/total)*100:.2f}%")
print(f"Top-5 Accuracy : {(top5/total)*100:.2f}%")

print("=" * 50)