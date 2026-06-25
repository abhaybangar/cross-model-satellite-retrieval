import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# Paths
WORKSPACE = r"c:\Users\banga\Desktop\ps_11_proto"
DATASET_DIR = os.path.join(WORKSPACE, "dataset")
METADATA_CSV = os.path.join(DATASET_DIR, "test2_metadata.csv")
CACHE_DIR = os.path.join(WORKSPACE, "backend", "cache")

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)

def calculate_accuracy(opt_feat, sar_feat, filenames):
    scores = np.matmul(opt_feat, sar_feat.T) # [N, N]
    
    top1 = 0
    top3 = 0
    top4 = 0
    top10 = 0
    n = scores.shape[0]
    
    for i in range(n):
        pred_indices = np.argsort(-scores[i])[:10]
        if i == pred_indices[0]:
            top1 += 1
        if i in pred_indices[:3]:
            top3 += 1
        if i in pred_indices[:4]:
            top4 += 1
        if i in pred_indices[:10]:
            top10 += 1
            
    return top1/n * 100, top3/n * 100, top4/n * 100, top10/n * 100

def main():
    print("Reading test2_metadata.csv...")
    df = pd.read_csv(METADATA_CSV)
    
    print("Loading DINOv2 model...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    opt_embeddings = []
    sar_embeddings = []
    filenames = []
    
    total = len(df)
    print(f"Extracting DINOv2 embeddings for {total} test2 pairs on {device}...")
    
    for idx, row in df.iterrows():
        # Replace 'test/' with 'test2/' to point to the correct folder
        opt_rel = row["optical_path"].replace("test/", "test2/")
        sar_rel = row["sar_path"].replace("test/", "test2/")
        row_id = row["id"]
        
        opt_path = os.path.join(DATASET_DIR, opt_rel)
        sar_path = os.path.join(DATASET_DIR, sar_rel)
        
        if not os.path.exists(opt_path) or not os.path.exists(sar_path):
            continue
            
        try:
            with torch.no_grad():
                # Process Optical
                img_opt = Image.open(opt_path).convert("RGB").resize((224, 224))
                inputs_opt = processor(images=img_opt, return_tensors="pt").to(device)
                out_opt = model(**inputs_opt).last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
                
                # Process SAR
                img_sar = Image.open(sar_path).convert("RGB").resize((224, 224))
                inputs_sar = processor(images=img_sar, return_tensors="pt").to(device)
                out_sar = model(**inputs_sar).last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
                
                opt_embeddings.append(out_opt)
                sar_embeddings.append(out_sar)
                filenames.append(row_id)
        except Exception as e:
            print(f"Error extracting {row_id}: {e}")
            
    opt_embeddings = np.array(opt_embeddings).astype("float32")
    sar_embeddings = np.array(sar_embeddings).astype("float32")
    
    # 1. EVALUATE RAW ACCURACY
    print("\n--- Evaluating Raw DINOv2 Model (No Projection) ---")
    raw_opt_norm = opt_embeddings / np.linalg.norm(opt_embeddings, axis=1, keepdims=True)
    raw_sar_norm = sar_embeddings / np.linalg.norm(sar_embeddings, axis=1, keepdims=True)
    r1, r3, r4, r10 = calculate_accuracy(raw_opt_norm, raw_sar_norm, filenames)
    print(f"Top-1: {r1:.2f}% | Top-3: {r3:.2f}% | Top-4: {r4:.2f}% | Top-10: {r10:.2f}%")
    
    # 2. EVALUATE PROJECTED ACCURACY
    opt_proj_path = os.path.join(CACHE_DIR, "opt_proj.pt")
    sar_proj_path = os.path.join(CACHE_DIR, "sar_proj.pt")
    
    if os.path.exists(opt_proj_path) and os.path.exists(sar_proj_path):
        print("\n--- Evaluating PyTorch Projection Heads (InfoNCE Aligned) ---")
        opt_proj = ProjectionHead()
        opt_proj.load_state_dict(torch.load(opt_proj_path, map_location="cpu"))
        opt_proj.eval()
        
        sar_proj = ProjectionHead()
        sar_proj.load_state_dict(torch.load(sar_proj_path, map_location="cpu"))
        sar_proj.eval()
        
        with torch.no_grad():
            proj_opt = opt_proj(torch.tensor(opt_embeddings)).numpy()
            proj_sar = sar_proj(torch.tensor(sar_embeddings)).numpy()
            
        p1, p3, p4, p10 = calculate_accuracy(proj_opt, proj_sar, filenames)
        
        print("\n" + "=" * 50)
        print(f"ACCURACY ON THE NEW 100 TEST2 PAIRS:")
        print(f"Top-1  Accuracy : {p1:.2f}% (vs. Raw: {r1:.2f}%)")
        print(f"Top-3  Accuracy : {p3:.2f}% (vs. Raw: {r3:.2f}%)")
        print(f"Top-4  Accuracy : {p4:.2f}% (vs. Raw: {r4:.2f}%)")
        print(f"Top-10 Accuracy : {p10:.2f}% (vs. Raw: {r10:.2f}%)")
        print("=" * 50)
    else:
        print("\nError: Trained projection heads not found in cache.")

if __name__ == "__main__":
    main()
