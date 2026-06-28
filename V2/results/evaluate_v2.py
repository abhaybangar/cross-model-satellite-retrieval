import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
METADATA_CSV = os.path.join(DATASET_DIR, "test2_metadata.csv")

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

def run_faiss_evaluation(query_feats, gallery_feats, query_names, gallery_names, index_path, dim):
    # Initialize FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(gallery_feats)
    
    # Save index
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index ({index.ntotal} items, dim={dim}) to {index_path}")
    
    # Evaluate queries
    top1 = 0
    top3 = 0
    top5 = 0
    top10 = 0
    n = query_feats.shape[0]
    
    for i in range(n):
        q_feat = query_feats[i:i+1] # [1, dim]
        q_name = query_names[i]
        
        distances, indices = index.search(q_feat, 10)
        retrieved_names = [gallery_names[idx] for idx in indices[0]]
        
        if q_name == retrieved_names[0]:
            top1 += 1
        if q_name in retrieved_names[:3]:
            top3 += 1
        if q_name in retrieved_names[:5]:
            top5 += 1
        if q_name in retrieved_names[:10]:
            top10 += 1
            
    return {
        "top1": top1 / n * 100,
        "top3": top3 / n * 100,
        "top5": top5 / n * 100,
        "top10": top10 / n * 100,
        "total_queries": n
    }

def main():
    print("Reading test2_metadata.csv...")
    df = pd.read_csv(METADATA_CSV)
    
    # Extract or load embeddings
    emb_path = os.path.join(V2_EMB_DIR, "test2_raw_embeddings.npz")
    if os.path.exists(emb_path):
        print(f"Loading cached V2 raw test embeddings from {emb_path}...")
        data = np.load(emb_path)
        opt_embeddings = data["opt"]
        sar_embeddings = data["sar"]
        filenames = list(data["filenames"])
    else:
        print("Loading DINOv2 model to extract test2 embeddings...")
        from transformers import AutoModel
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
            opt_rel = row["optical_path"]
            sar_rel = row["sar_path"]
            row_id = row["id"]
            
            opt_path = os.path.join(DATASET_DIR, opt_rel)
            sar_path = os.path.join(DATASET_DIR, sar_rel)
            
            # Fallback to root optical/sar directories for raw files if not found
            if not os.path.exists(opt_path):
                opt_path = os.path.join(DATASET_DIR, "optical", os.path.basename(opt_rel))
            if not os.path.exists(sar_path):
                sar_path = os.path.join(DATASET_DIR, "sar", os.path.basename(sar_rel))
                
            if not os.path.exists(opt_path) or not os.path.exists(sar_path):
                continue
                
            try:
                with torch.no_grad():
                    # Process Optical
                    opt_array = preprocess_optical(opt_path)
                    pixel_values_opt = torch.tensor(opt_array).unsqueeze(0).to(device)
                    out_opt = model(pixel_values=pixel_values_opt).last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
                    
                    # Process SAR
                    sar_array = preprocess_sar(sar_path)
                    pixel_values_sar = torch.tensor(sar_array).unsqueeze(0).to(device)
                    out_sar = model(pixel_values=pixel_values_sar).last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
                    
                    opt_embeddings.append(out_opt)
                    sar_embeddings.append(out_sar)
                    filenames.append(row_id)
            except Exception as e:
                print(f"Error extracting row {row_id}: {e}")
                
        opt_embeddings = np.array(opt_embeddings).astype("float32")
        sar_embeddings = np.array(sar_embeddings).astype("float32")
        
        print(f"Saving extracted test2 embeddings to {emb_path}...")
        np.savez_compressed(emb_path, opt=opt_embeddings, sar=sar_embeddings, filenames=np.array(filenames))

    # Normalize raw embeddings for cosine similarity FlatL2 FAISS
    raw_opt_norm = opt_embeddings / np.linalg.norm(opt_embeddings, axis=1, keepdims=True)
    raw_sar_norm = sar_embeddings / np.linalg.norm(sar_embeddings, axis=1, keepdims=True)
    
    # 1. Evaluate RAW accuracy
    print("\n--- Evaluating V2 Raw DINOv2 Model (No Projection) using FAISS ---")
    raw_index_path = os.path.join(V2_EMB_DIR, "raw_test2_gallery.index")
    raw_metrics = run_faiss_evaluation(
        raw_opt_norm, raw_sar_norm, filenames, filenames, raw_index_path, dim=768
    )
    print(f"Raw Test2 Top-1: {raw_metrics['top1']:.2f}% | Top-3: {raw_metrics['top3']:.2f}% | Top-10: {raw_metrics['top10']:.2f}%")

    # 2. Evaluate PROJECTED accuracy
    opt_proj_path = os.path.join(V2_MODEL_DIR, "opt_proj.pt")
    sar_proj_path = os.path.join(V2_MODEL_DIR, "sar_proj.pt")
    
    projected_metrics = None
    if os.path.exists(opt_proj_path) and os.path.exists(sar_proj_path):
        print("\n--- Evaluating V2 Aligned Projection Heads (InfoNCE Aligned) using FAISS ---")
        opt_proj = ProjectionHead()
        opt_proj.load_state_dict(torch.load(opt_proj_path, map_location="cpu"))
        opt_proj.eval()
        
        sar_proj = ProjectionHead()
        sar_proj.load_state_dict(torch.load(sar_proj_path, map_location="cpu"))
        sar_proj.eval()
        
        with torch.no_grad():
            proj_opt = opt_proj(torch.tensor(opt_embeddings)).numpy()
            proj_sar = sar_proj(torch.tensor(sar_embeddings)).numpy()
            
        proj_index_path = os.path.join(V2_EMB_DIR, "projected_test2_gallery.index")
        projected_metrics = run_faiss_evaluation(
            proj_opt, proj_sar, filenames, filenames, proj_index_path, dim=256
        )
        
        print("\n" + "=" * 50)
        print(f"V2 ACCURACY ON THE 100 TEST2 PAIRS:")
        print(f"Top-1 Accuracy  : {projected_metrics['top1']:.2f}% (vs. Raw: {raw_metrics['top1']:.2f}%)")
        print(f"Top-3 Accuracy  : {projected_metrics['top3']:.2f}% (vs. Raw: {raw_metrics['top3']:.2f}%)")
        print(f"Top-5 Accuracy  : {projected_metrics['top5']:.2f}% (vs. Raw: {raw_metrics['top5']:.2f}%)")
        print(f"Top-10 Accuracy : {projected_metrics['top10']:.2f}% (vs. Raw: {raw_metrics['top10']:.2f}%)")
        print("=" * 50)
    else:
        print(f"\nError: Trained V2 projection heads not found in {V2_MODEL_DIR}.")

    # Save metrics JSON & Report txt
    metrics_json_path = os.path.join(V2_RESULT_DIR, "metrics_test2.json")
    metrics_data = {
        "raw_accuracy": raw_metrics,
        "projected_accuracy": projected_metrics
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
        
    report_txt_path = os.path.join(V2_RESULT_DIR, "evaluation_v2.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("=== V2 PIPELINE EVALUATION ON TEST2 PAIRS ===\n")
        f.write(f"Total Queries evaluated: {raw_metrics['total_queries']}\n\n")
        f.write("--- RAW DINOV2 EMBEDDINGS (No projection) ---\n")
        f.write(f"Top-1 Accuracy  : {raw_metrics['top1']:.2f}%\n")
        f.write(f"Top-3 Accuracy  : {raw_metrics['top3']:.2f}%\n")
        f.write(f"Top-5 Accuracy  : {raw_metrics['top5']:.2f}%\n")
        f.write(f"Top-10 Accuracy : {raw_metrics['top10']:.2f}%\n\n")
        if projected_metrics:
            f.write("--- PROJECTED DINOV2 EMBEDDINGS (V2 Projection Head) ---\n")
            f.write(f"Top-1 Accuracy  : {projected_metrics['top1']:.2f}%\n")
            f.write(f"Top-3 Accuracy  : {projected_metrics['top3']:.2f}%\n")
            f.write(f"Top-5 Accuracy  : {projected_metrics['top5']:.2f}%\n")
            f.write(f"Top-10 Accuracy : {projected_metrics['top10']:.2f}%\n")
            
    print(f"Metrics saved to {metrics_json_path}")
    print(f"Text report saved to {report_txt_path}")

if __name__ == "__main__":
    main()
