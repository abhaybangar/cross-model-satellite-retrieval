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

OPT_FOLDER = "dataset/optical"
SAR_FOLDER = "dataset/sar"

opt_files = set(os.listdir(OPT_FOLDER))
sar_files = set(os.listdir(SAR_FOLDER))
matching_files = sorted(list(opt_files.intersection(sar_files)))

def extract_embedding(image_path, pooling="mean"):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    if pooling == "mean":
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    elif pooling == "cls":
        emb = outputs.last_hidden_state[:, 0].squeeze().cpu().numpy()
    elif pooling == "pooler":
        emb = outputs.pooler_output.squeeze().cpu().numpy()
    else:
        raise ValueError("Unknown pooling")
    return emb

for pooling in ["mean", "cls", "pooler"]:
    for normalize in [False, True]:
        print(f"\n--- Evaluating Pooling: {pooling}, Normalize: {normalize} ---")
        
        # Build SAR Embeddings
        sar_embeddings = []
        for filename in matching_files:
            emb = extract_embedding(os.path.join(SAR_FOLDER, filename), pooling=pooling)
            if normalize:
                emb = emb / np.linalg.norm(emb)
            sar_embeddings.append(emb)
        sar_embeddings = np.array(sar_embeddings).astype("float32")
        
        # Build FAISS index
        if normalize:
            index = faiss.IndexFlatIP(768)
        else:
            index = faiss.IndexFlatL2(768)
        index.add(sar_embeddings)
        
        # Evaluate
        top1 = 0
        top3 = 0
        top5 = 0
        total = len(matching_files)
        
        for filename in matching_files:
            query = extract_embedding(os.path.join(OPT_FOLDER, filename), pooling=pooling)
            if normalize:
                query = query / np.linalg.norm(query)
            query = query.reshape(1, 768).astype("float32")
            
            distances, indices = index.search(query, 5)
            retrieved = [matching_files[idx] for idx in indices[0]]
            
            if filename == retrieved[0]:
                top1 += 1
            if filename in retrieved[:3]:
                top3 += 1
            if filename in retrieved[:5]:
                top5 += 1
                
        print(f"Top-1: {(top1/total)*100:.2f}% | Top-3: {(top3/total)*100:.2f}% | Top-5: {(top5/total)*100:.2f}%")
