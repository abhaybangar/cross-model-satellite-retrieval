from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from transformers import AutoImageProcessor, AutoModel
import torch
import os
import numpy as np
import faiss

class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim=768, output_dim=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.LayerNorm(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, output_dim)
        )

    def forward(self, x):
        z = self.net(x)
        return torch.nn.functional.normalize(z, p=2, dim=-1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OPT_FOLDER = os.path.join(WORKSPACE, "dataset", "optical")
SAR_FOLDER = os.path.join(WORKSPACE, "dataset", "sar")
CACHE_PATH = os.path.join(WORKSPACE, "backend", "cache", "raw_dinov2_embeddings.npz")

opt_files = set(os.listdir(OPT_FOLDER))
sar_files = set(os.listdir(SAR_FOLDER))
matching_files = sorted(list(opt_files.intersection(sar_files)))

print("Loading dataset embeddings...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(CACHE_PATH):
    print("Found cached embeddings. Loading...")
    data = np.load(CACHE_PATH)
    opt_embeddings_dict = {f: emb for f, emb in zip(data["filenames"], data["opt"])}
    sar_embeddings_dict = {f: emb for f, emb in zip(data["filenames"], data["sar"])}
else:
    print("Cache not found. Extracting embeddings on the fly (this may take a few minutes)...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()
    
    opt_embeddings_dict = {}
    sar_embeddings_dict = {}
    
    for idx, filename in enumerate(matching_files):
        with torch.no_grad():
            # Optical
            img_opt = Image.open(os.path.join(OPT_FOLDER, filename)).convert("RGB").resize((224, 224))
            inputs_opt = processor(images=img_opt, return_tensors="pt").to(device)
            emb_opt = model(**inputs_opt).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            opt_embeddings_dict[filename] = emb_opt
            
            # SAR
            img_sar = Image.open(os.path.join(SAR_FOLDER, filename)).convert("RGB").resize((224, 224))
            inputs_sar = processor(images=img_sar, return_tensors="pt").to(device)
            emb_sar = model(**inputs_sar).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            sar_embeddings_dict[filename] = emb_sar
            
        if (idx + 1) % 100 == 0 or (idx + 1) == len(matching_files):
            print(f"Extraction progress: {idx + 1}/{len(matching_files)}")

# 1. EVALUATE MEAN POOLING (RAW DINOV2)
for normalize in [False, True]:
    print(f"\n--- Evaluating Pooling: mean, Normalize: {normalize} ---")
    
    # Build SAR matrix
    sar_embeddings = np.array([sar_embeddings_dict[f] for f in matching_files]).astype("float32")
    if normalize:
        sar_embeddings = sar_embeddings / np.linalg.norm(sar_embeddings, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(768)
    else:
        index = faiss.IndexFlatL2(768)
    index.add(sar_embeddings)
    
    # Evaluate
    top1, top3, top5 = 0, 0, 0
    total = len(matching_files)
    
    for filename in matching_files:
        query = opt_embeddings_dict[filename]
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

# 2. EVALUATE PROJECTION HEADS
opt_proj_path = os.path.join(WORKSPACE, "backend", "cache", "opt_proj.pt")
sar_proj_path = os.path.join(WORKSPACE, "backend", "cache", "sar_proj.pt")

if os.path.exists(opt_proj_path) and os.path.exists(sar_proj_path):
    print("\n--- Evaluating PyTorch Projection Heads (InfoNCE Aligned) ---")
    opt_proj = ProjectionHead()
    opt_proj.load_state_dict(torch.load(opt_proj_path, map_location="cpu"))
    opt_proj.eval()
    
    sar_proj = ProjectionHead()
    sar_proj.load_state_dict(torch.load(sar_proj_path, map_location="cpu"))
    sar_proj.eval()
    
    sar_embeddings = np.array([sar_embeddings_dict[f] for f in matching_files]).astype("float32")
    with torch.no_grad():
        sar_proj_embeddings = sar_proj(torch.tensor(sar_embeddings)).numpy()
        
    index = faiss.IndexFlatIP(256)
    index.add(sar_proj_embeddings)
    
    top1, top3, top5 = 0, 0, 0
    total = len(matching_files)
    
    for filename in matching_files:
        query = opt_embeddings_dict[filename].reshape(1, -1)
        with torch.no_grad():
            query_proj = opt_proj(torch.tensor(query)).numpy()
            
        distances, indices = index.search(query_proj, 5)
        retrieved = [matching_files[idx] for idx in indices[0]]
        
        if filename == retrieved[0]:
            top1 += 1
        if filename in retrieved[:3]:
            top3 += 1
        if filename in retrieved[:5]:
            top5 += 1
            
    print(f"Top-1: {(top1/total)*100:.2f}% | Top-3: {(top3/total)*100:.2f}% | Top-5: {(top5/total)*100:.2f}%")
