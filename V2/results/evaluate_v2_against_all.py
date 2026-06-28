import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import faiss

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)
from ben_preprocess import preprocess_optical, preprocess_sar

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATASET_DIR = os.path.join(WORKSPACE, "dataset")

# V2 Isolated directories under root V2 folder
V2_EMB_DIR = os.path.join(WORKSPACE, "V2", "embeddings")
V2_MODEL_DIR = os.path.join(WORKSPACE, "V2", "projection_head")
V2_RESULT_DIR = os.path.join(WORKSPACE, "V2", "results")

os.makedirs(V2_EMB_DIR, exist_ok=True)
os.makedirs(V2_MODEL_DIR, exist_ok=True)
os.makedirs(V2_RESULT_DIR, exist_ok=True)

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

def run_faiss_evaluation(query_feats, gallery_feats, query_ids, gallery_ids, index_path, dim):
    # Initialize FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(gallery_feats)
    
    # Save index
    faiss.write_index(index, index_path)
    print(f"Saved V2 Combined FAISS index ({index.ntotal} items, dim={dim}) to {index_path}")
    
    # Evaluate queries
    top1 = 0
    top3 = 0
    top4 = 0
    top10 = 0
    n = query_feats.shape[0]
    
    for i in range(n):
        q_feat = query_feats[i:i+1] # [1, dim]
        q_id = query_ids[i]
        
        distances, indices = index.search(q_feat, 10)
        retrieved_ids = [gallery_ids[idx] for idx in indices[0]]
        
        # Match using basename (e.g. img_2001.tif)
        q_base = os.path.basename(q_id)
        retrieved_bases = [os.path.basename(r_id) for r_id in retrieved_ids]
        
        if q_base == retrieved_bases[0]:
            top1 += 1
        if q_base in retrieved_bases[:3]:
            top3 += 1
        if q_base in retrieved_bases[:4]:
            top4 += 1
        if q_base in retrieved_bases[:10]:
            top10 += 1
            
    return {
        "top1": top1 / n * 100,
        "top3": top3 / n * 100,
        "top4": top4 / n * 100,
        "top10": top10 / n * 100,
        "total_queries": n
    }

def get_sar_files_from_folder(folder):
    sar_dir = os.path.join(DATASET_DIR, folder, "sar")
    if not os.path.exists(sar_dir):
        return []
    files = []
    for f in os.listdir(sar_dir):
        if f.endswith(".tif"):
            files.append((os.path.join(sar_dir, f), f"{folder}/{f}"))
    return files

def main():
    print("Gathering V2 gallery image paths from raw datasets (train, test, & sar)...")
    all_sar_items = []
    
    # train (img_0001 to img_1800)
    train_items = []
    train_sar_dir = os.path.join(DATASET_DIR, "train", "sar")
    for f in sorted(os.listdir(train_sar_dir)):
        if f.endswith(".tif"):
            train_items.append((os.path.join(train_sar_dir, f), f"train/sar/{f}"))
    all_sar_items.extend(train_items)
    print(f" - Found {len(train_items)} SAR images in train/sar")
    
    # test (img_1801 to img_2000)
    test_items = []
    test_sar_dir = os.path.join(DATASET_DIR, "test", "sar")
    for f in sorted(os.listdir(test_sar_dir)):
        if f.endswith(".tif"):
            test_items.append((os.path.join(test_sar_dir, f), f"test/sar/{f}"))
    all_sar_items.extend(test_items)
    print(f" - Found {len(test_items)} SAR images in test/sar")
    
    # sar (img_2001 to img_2100)
    sar_items = []
    sar_dir = os.path.join(DATASET_DIR, "sar")
    for f in sorted(os.listdir(sar_dir)):
        if f.endswith(".tif"):
            sar_items.append((os.path.join(sar_dir, f), f"sar/{f}"))
    all_sar_items.extend(sar_items)
    print(f" - Found {len(sar_items)} SAR images in sar")
    
    print(f"Total V2 Combined Search Gallery size: {len(all_sar_items)} SAR images.")
    
    # test2 optical queries from raw optical/ folder
    test2_opt_dir = os.path.join(DATASET_DIR, "optical")
    test2_queries = []
    for f in sorted(os.listdir(test2_opt_dir)):
        if f.endswith(".tif"):
            test2_queries.append((os.path.join(test2_opt_dir, f), f"optical/{f}"))
            
    print(f"Total V2 Queries (test2 optical): {len(test2_queries)} images.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if cached combined V2 embeddings exist
    cache_path = os.path.join(V2_EMB_DIR, "combined_evaluation_embeddings_v2.npz")
    if os.path.exists(cache_path):
        print(f"\nLoading cached combined V2 embeddings from {cache_path}...")
        data = np.load(cache_path)
        opt_embeddings = data["opt"]
        sar_embeddings = data["sar"]
        query_ids = list(data["query_ids"])
        gallery_ids = list(data["gallery_ids"])
    else:
        # Try to load from existing V2 raw training and test embeddings
        train2_emb_path = os.path.join(V2_EMB_DIR, "train2_raw_embeddings.npz")
        test2_emb_path = os.path.join(V2_EMB_DIR, "test2_raw_embeddings.npz")
        
        reused_cache = False
        if os.path.exists(train2_emb_path) and os.path.exists(test2_emb_path):
            try:
                print("Reusing raw V2 train and test embeddings to build combined gallery...")
                data_train2 = np.load(train2_emb_path)
                data_test2 = np.load(test2_emb_path)
                
                # Make lookups using basenames
                train2_sar_lookup = {os.path.basename(img_id): emb for img_id, emb in zip(data_train2["ids"], data_train2["sar"])}
                test2_opt_lookup = {os.path.basename(img_id): emb for img_id, emb in zip(data_test2["filenames"], data_test2["opt"])}
                test2_sar_lookup = {os.path.basename(img_id): emb for img_id, emb in zip(data_test2["filenames"], data_test2["sar"])}
                
                # Reconstruct
                opt_embeddings = []
                query_ids = []
                missing_opt = []
                
                for path, q_id in test2_queries:
                    q_base = os.path.basename(q_id)
                    if q_base in test2_opt_lookup:
                        opt_embeddings.append(test2_opt_lookup[q_base])
                        query_ids.append(q_id)
                    else:
                        missing_opt.append((path, q_id))
                        
                sar_embeddings = []
                gallery_ids = []
                missing_sar = []
                
                for path, g_id in all_sar_items:
                    g_base = os.path.basename(g_id)
                    if g_base in test2_sar_lookup:
                        sar_embeddings.append(test2_sar_lookup[g_base])
                        gallery_ids.append(g_id)
                    elif g_base in train2_sar_lookup:
                        sar_embeddings.append(train2_sar_lookup[g_base])
                        gallery_ids.append(g_id)
                    else:
                        missing_sar.append((path, g_id))
                
                # If there are any missing items, we will extract them on the fly
                if missing_opt or missing_sar:
                    print(f"Extraction required for {len(missing_opt)} missing queries and {len(missing_sar)} missing gallery items...")
                    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
                    model.eval()
                    
                    for path, q_id in missing_opt:
                        with torch.no_grad():
                            opt_array = preprocess_optical(path)
                            pixel_values_opt = torch.tensor(opt_array).unsqueeze(0).to(device)
                            emb = model(pixel_values=pixel_values_opt).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                            opt_embeddings.append(emb)
                            query_ids.append(q_id)
                            
                    for path, g_id in missing_sar:
                        with torch.no_grad():
                            sar_array = preprocess_sar(path)
                            pixel_values_sar = torch.tensor(sar_array).unsqueeze(0).to(device)
                            emb = model(pixel_values=pixel_values_sar).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                            sar_embeddings.append(emb)
                            gallery_ids.append(g_id)
                
                opt_embeddings = np.array(opt_embeddings).astype("float32")
                sar_embeddings = np.array(sar_embeddings).astype("float32")
                reused_cache = True
                print("Successfully constructed combined V2 embeddings from cache.")
                
            except Exception as e:
                print(f"Could not reconstruct from cache: {e}. Falling back to full extraction...")
                reused_cache = False
                
        if not reused_cache:
            print("\nLoading DINOv2 model to extract V2 combined embeddings from scratch...")
            model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
            model.eval()
            
            # Extract Queries
            opt_embeddings = []
            query_ids = []
            print(f"\nExtracting V2 optical query embeddings on {device}...")
            for idx, (path, q_id) in enumerate(test2_queries):
                with torch.no_grad():
                    opt_array = preprocess_optical(path)
                    pixel_values_opt = torch.tensor(opt_array).unsqueeze(0).to(device)
                    emb = model(pixel_values=pixel_values_opt).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                    opt_embeddings.append(emb)
                    query_ids.append(q_id)
                if (idx + 1) % 50 == 0 or (idx + 1) == len(test2_queries):
                    print(f"Queries progress: {idx + 1}/{len(test2_queries)}")
                    
            # Extract Gallery
            sar_embeddings = []
            gallery_ids = []
            print(f"\nExtracting combined V2 SAR gallery embeddings (this may take a few minutes) on {device}...")
            for idx, (path, g_id) in enumerate(all_sar_items):
                with torch.no_grad():
                    sar_array = preprocess_sar(path)
                    pixel_values_sar = torch.tensor(sar_array).unsqueeze(0).to(device)
                    emb = model(pixel_values=pixel_values_sar).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                    sar_embeddings.append(emb)
                    gallery_ids.append(g_id)
                if (idx + 1) % 500 == 0 or (idx + 1) == len(all_sar_items):
                    print(f"Gallery progress: {idx + 1}/{len(all_sar_items)}")
                    
            opt_embeddings = np.array(opt_embeddings).astype("float32")
            sar_embeddings = np.array(sar_embeddings).astype("float32")
            
        # Save to cache
        print(f"\nSaving V2 combined embeddings to cache at {cache_path}...")
        np.savez_compressed(
            cache_path, 
            opt=opt_embeddings, 
            sar=sar_embeddings, 
            query_ids=np.array(query_ids), 
            gallery_ids=np.array(gallery_ids)
        )
        
    # Normalize raw embeddings for cosine similarity FlatL2 FAISS
    raw_opt_norm = opt_embeddings / np.linalg.norm(opt_embeddings, axis=1, keepdims=True)
    raw_sar_norm = sar_embeddings / np.linalg.norm(sar_embeddings, axis=1, keepdims=True)
    
    # 1. Evaluate RAW accuracy
    print("\n--- Evaluating V2 Raw DINOv2 Model (No Projection) ---")
    raw_index_path = os.path.join(V2_EMB_DIR, "raw_combined_gallery_v2.index")
    raw_metrics = run_faiss_evaluation(
        raw_opt_norm, raw_sar_norm, query_ids, gallery_ids, raw_index_path, dim=768
    )
    print(f"Raw Combined Top-1: {raw_metrics['top1']:.2f}% | Top-3: {raw_metrics['top3']:.2f}% | Top-10: {raw_metrics['top10']:.2f}%")
    
    # 2. Evaluate PROJECTED accuracy
    opt_proj_path = os.path.join(V2_MODEL_DIR, "opt_proj.pt")
    sar_proj_path = os.path.join(V2_MODEL_DIR, "sar_proj.pt")
    
    projected_metrics = None
    if os.path.exists(opt_proj_path) and os.path.exists(sar_proj_path):
        print("\n--- Evaluating V2 Aligned Projection Heads (InfoNCE Aligned) ---")
        opt_proj = ProjectionHead()
        opt_proj.load_state_dict(torch.load(opt_proj_path, map_location="cpu"))
        opt_proj.eval()
        
        sar_proj = ProjectionHead()
        sar_proj.load_state_dict(torch.load(sar_proj_path, map_location="cpu"))
        sar_proj.eval()
        
        with torch.no_grad():
            proj_opt = opt_proj(torch.tensor(opt_embeddings)).numpy()
            proj_sar = sar_proj(torch.tensor(sar_embeddings)).numpy()
            
        proj_index_path = os.path.join(V2_EMB_DIR, "projected_combined_gallery_v2.index")
        projected_metrics = run_faiss_evaluation(
            proj_opt, proj_sar, query_ids, gallery_ids, proj_index_path, dim=256
        )
        
        print("\n" + "=" * 50)
        print(f"V2 ACCURACY OF TEST2 OPTICAL QUERIES AGAINST ALL V2 SAR GALLERY ({len(gallery_ids)} images):")
        print(f"Top-1  Accuracy : {projected_metrics['top1']:.2f}% (vs. Raw: {raw_metrics['top1']:.2f}%)")
        print(f"Top-3  Accuracy : {projected_metrics['top3']:.2f}% (vs. Raw: {raw_metrics['top3']:.2f}%)")
        print(f"Top-4  Accuracy : {projected_metrics['top4']:.2f}% (vs. Raw: {raw_metrics['top4']:.2f}%)")
        print(f"Top-10 Accuracy : {projected_metrics['top10']:.2f}% (vs. Raw: {raw_metrics['top10']:.2f}%)")
        print("=" * 50)
    else:
        print(f"\nError: Trained V2 projection heads not found in {V2_MODEL_DIR}.")

    # Save metrics JSON & Report txt
    metrics_json_path = os.path.join(V2_RESULT_DIR, "metrics_combined_gallery_v2.json")
    metrics_data = {
        "raw_accuracy": raw_metrics,
        "projected_accuracy": projected_metrics
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
        
    report_txt_path = os.path.join(V2_RESULT_DIR, "evaluation_v2_against_all.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("=== V2 PIPELINE EVALUATION ON COMBINED V2 GALLERY ===\n")
        f.write(f"Total Query Images: {raw_metrics['total_queries']}\n")
        f.write(f"Total Gallery Images: {len(gallery_ids)}\n\n")
        f.write("--- RAW DINOV2 EMBEDDINGS (No projection) ---\n")
        f.write(f"Top-1 Accuracy  : {raw_metrics['top1']:.2f}%\n")
        f.write(f"Top-3 Accuracy  : {raw_metrics['top3']:.2f}%\n")
        f.write(f"Top-4 Accuracy  : {raw_metrics['top4']:.2f}%\n")
        f.write(f"Top-10 Accuracy : {raw_metrics['top10']:.2f}%\n\n")
        if projected_metrics:
            f.write("--- PROJECTED DINOV2 EMBEDDINGS (V2 Projection Head) ---\n")
            f.write(f"Top-1 Accuracy  : {projected_metrics['top1']:.2f}%\n")
            f.write(f"Top-3 Accuracy  : {projected_metrics['top3']:.2f}%\n")
            f.write(f"Top-4 Accuracy  : {projected_metrics['top4']:.2f}%\n")
            f.write(f"Top-10 Accuracy : {projected_metrics['top10']:.2f}%\n")
            
    print(f"Combined metrics saved to {metrics_json_path}")
    print(f"Combined report saved to {report_txt_path}")

if __name__ == "__main__":
    main()
