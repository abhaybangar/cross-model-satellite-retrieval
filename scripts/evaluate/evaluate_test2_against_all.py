import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATASET_DIR = os.path.join(WORKSPACE, "dataset")
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

def calculate_accuracy(opt_feat, sar_feat, query_ids, gallery_ids):
    scores = np.matmul(opt_feat, sar_feat.T) # [N_queries, N_gallery]
    
    top1 = 0
    top3 = 0
    top4 = 0
    top10 = 0
    n = scores.shape[0]
    
    for i in range(n):
        q_id = query_ids[i]
        pred_indices = np.argsort(-scores[i])[:10]
        retrieved_ids = [gallery_ids[idx] for idx in pred_indices]
        
        if q_id == retrieved_ids[0]:
            top1 += 1
        if q_id in retrieved_ids[:3]:
            top3 += 1
        if q_id in retrieved_ids[:4]:
            top4 += 1
        if q_id in retrieved_ids[:10]:
            top10 += 1
            
    return top1/n * 100, top3/n * 100, top4/n * 100, top10/n * 100

def get_sar_files_from_folder(folder):
    sar_dir = os.path.join(DATASET_DIR, folder, "sar")
    if not os.path.exists(sar_dir):
        return []
    # Return list of tuples: (full_path, unique_id)
    # unique_id is the filename (e.g. img_0001.tif) prefixed by the folder name (e.g. train/img_0001.tif)
    # to avoid collisions since different datasets have duplicate filenames (e.g. img_0001.tif is in test, test2, train2)
    files = []
    for f in os.listdir(sar_dir):
        if f.endswith(".tif"):
            files.append((os.path.join(sar_dir, f), f"{folder}/{f}"))
    return files

def main():
    print("Gathering gallery image paths from all datasets...")
    # Combined gallery from train, train2, test, test2
    folders = ["train", "train2", "test", "test2"]
    all_sar_items = []
    for f in folders:
        items = get_sar_files_from_folder(f)
        all_sar_items.extend(items)
        print(f" - Found {len(items)} SAR images in folder: {f}")
        
    print(f"Total Combined Search Gallery size: {len(all_sar_items)} SAR images.")
    
    # test2 optical queries
    test2_opt_dir = os.path.join(DATASET_DIR, "test2", "optical")
    test2_queries = []
    for f in sorted(os.listdir(test2_opt_dir)):
        if f.endswith(".tif"):
            # matching query id is 'test2/img_XXXX.tif' to match the gallery unique_id
            test2_queries.append((os.path.join(test2_opt_dir, f), f"test2/{f}"))
            
    print(f"Total Queries (test2 optical): {len(test2_queries)} images.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if cached combined embeddings exist
    cache_path = os.path.join(CACHE_DIR, "combined_evaluation_embeddings.npz")
    if os.path.exists(cache_path):
        print("\nLoading cached combined embeddings...")
        data = np.load(cache_path)
        opt_embeddings = data["opt"]
        sar_embeddings = data["sar"]
        query_ids = list(data["query_ids"])
        gallery_ids = list(data["gallery_ids"])
    else:
        print("\nLoading DINOv2 model to extract embeddings...")
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        model.eval()
        
        # Extract Queries
        opt_embeddings = []
        query_ids = []
        print(f"\nExtracting optical query embeddings on {device}...")
        for idx, (path, q_id) in enumerate(test2_queries):
            with torch.no_grad():
                img = Image.open(path).convert("RGB").resize((224, 224))
                inputs = processor(images=img, return_tensors="pt").to(device)
                emb = model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                opt_embeddings.append(emb)
                query_ids.append(q_id)
            if (idx + 1) % 50 == 0 or (idx + 1) == len(test2_queries):
                print(f"Queries progress: {idx + 1}/{len(test2_queries)}")
                
        # Extract Gallery
        sar_embeddings = []
        gallery_ids = []
        print(f"\nExtracting combined SAR gallery embeddings (this may take a few minutes) on {device}...")
        for idx, (path, g_id) in enumerate(all_sar_items):
            with torch.no_grad():
                img = Image.open(path).convert("RGB").resize((224, 224))
                inputs = processor(images=img, return_tensors="pt").to(device)
                emb = model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                sar_embeddings.append(emb)
                gallery_ids.append(g_id)
            if (idx + 1) % 500 == 0 or (idx + 1) == len(all_sar_items):
                print(f"Gallery progress: {idx + 1}/{len(all_sar_items)}")
                
        opt_embeddings = np.array(opt_embeddings).astype("float32")
        sar_embeddings = np.array(sar_embeddings).astype("float32")
        
        # Save to cache
        print("\nSaving combined embeddings to cache...")
        np.savez_compressed(
            cache_path, 
            opt=opt_embeddings, 
            sar=sar_embeddings, 
            query_ids=np.array(query_ids), 
            gallery_ids=np.array(gallery_ids)
        )
        
    # 1. EVALUATE RAW ACCURACY
    print("\n--- Evaluating Raw DINOv2 Model (No Projection) ---")
    raw_opt_norm = opt_embeddings / np.linalg.norm(opt_embeddings, axis=1, keepdims=True)
    raw_sar_norm = sar_embeddings / np.linalg.norm(sar_embeddings, axis=1, keepdims=True)
    r1, r3, r4, r10 = calculate_accuracy(raw_opt_norm, raw_sar_norm, query_ids, gallery_ids)
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
            
        p1, p3, p4, p10 = calculate_accuracy(proj_opt, proj_sar, query_ids, gallery_ids)
        
        print("\n" + "=" * 50)
        print(f"ACCURACY OF TEST2 OPTICAL QUERIES AGAINST ALL SAR GALLERY ({len(gallery_ids)} images):")
        print(f"Top-1  Accuracy : {p1:.2f}% (vs. Raw: {r1:.2f}%)")
        print(f"Top-3  Accuracy : {p3:.2f}% (vs. Raw: {r3:.2f}%)")
        print(f"Top-4  Accuracy : {p4:.2f}% (vs. Raw: {r4:.2f}%)")
        print(f"Top-10 Accuracy : {p10:.2f}% (vs. Raw: {r10:.2f}%)")
        print("=" * 50)
    else:
        print("\nError: Trained projection heads not found in cache.")

if __name__ == "__main__":
    main()
