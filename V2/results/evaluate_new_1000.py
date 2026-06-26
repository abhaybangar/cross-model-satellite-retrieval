import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import faiss

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
NEW_DATASET_DIR = os.path.join(WORKSPACE, "preprocessed_1000 (1)", "preprocessed_1000")
METADATA_CSV = os.path.join(NEW_DATASET_DIR, "metadata.csv")

# V1 and V2 Models/Cache
V1_MODEL_DIR = os.path.join(WORKSPACE, "backend", "cache")
V2_EMB_DIR = os.path.join(WORKSPACE, "V2", "embeddings")
V2_MODEL_DIR = os.path.join(WORKSPACE, "V2", "projection_head")
V2_RESULT_DIR = os.path.join(WORKSPACE, "V2", "results")

os.makedirs(V2_EMB_DIR, exist_ok=True)
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

def run_faiss_evaluation(query_feats, gallery_feats, query_names, gallery_names, dim):
    index = faiss.IndexFlatL2(dim)
    index.add(gallery_feats)
    
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
    if not os.path.exists(METADATA_CSV):
        print(f"Error: metadata.csv not found at {METADATA_CSV}")
        return
        
    print(f"Reading metadata.csv from {METADATA_CSV}...")
    df = pd.read_csv(METADATA_CSV)
    
    # Extract or load embeddings
    emb_path = os.path.join(V2_EMB_DIR, "new_1000_raw_embeddings.npz")
    if os.path.exists(emb_path):
        print(f"Loading cached raw embeddings for preprocessed_1000 from {emb_path}...")
        data = np.load(emb_path)
        opt_embeddings = data["opt"]
        sar_embeddings_asis = data["sar_asis"]
        sar_embeddings_norm = data["sar_norm"]
        filenames = list(data["filenames"])
    else:
        print("Loading DINOv2 model to extract embeddings...")
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        opt_embeddings = []
        sar_embeddings_asis = []
        sar_embeddings_norm = []
        filenames = []
        
        total = len(df)
        print(f"Extracting DINOv2 embeddings for {total} pairs on {device}...")
        
        for idx, row in df.iterrows():
            opt_rel = row["optical_path"]
            sar_rel = row["sar_path"]
            row_id = row["id"]
            
            opt_path = os.path.join(NEW_DATASET_DIR, opt_rel)
            sar_path = os.path.join(NEW_DATASET_DIR, sar_rel)
            
            if not os.path.exists(opt_path) or not os.path.exists(sar_path):
                continue
                
            try:
                with torch.no_grad():
                    # 1. Process Optical
                    # Optical is already normalized and float32 format, shape (224, 224, 3)
                    img_opt = tifffile.imread(opt_path)
                    tensor_opt = torch.tensor(img_opt).permute(2, 0, 1).unsqueeze(0).to(device)
                    out_opt = model(tensor_opt).last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
                    
                    # 2. Process SAR as-is (dB scale)
                    img_sar = tifffile.imread(sar_path)
                    tensor_sar_asis = torch.tensor(img_sar).permute(2, 0, 1).unsqueeze(0).to(device)
                    out_sar_asis = model(tensor_sar_asis).last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
                    
                    # 3. Process SAR with V2 Percentile Normalization
                    sar_norm_channels = []
                    for c in range(3):
                        channel = img_sar[:, :, c]
                        b_min, b_max = np.percentile(channel, [1, 99])
                        if b_max == b_min:
                            norm_channel = np.zeros_like(channel, dtype=np.uint8)
                        else:
                            normalized = (channel - b_min) / (b_max - b_min) * 255.0
                            norm_channel = np.clip(normalized, 0, 255).astype(np.uint8)
                        sar_norm_channels.append(norm_channel)
                    img_sar_norm = np.stack(sar_norm_channels, axis=-1)
                    
                    img_sar_pil = Image.fromarray(img_sar_norm)
                    inputs_sar_norm = processor(images=img_sar_pil, return_tensors="pt").to(device)
                    out_sar_norm = model(**inputs_sar_norm).last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
                    
                    opt_embeddings.append(out_opt)
                    sar_embeddings_asis.append(out_sar_asis)
                    sar_embeddings_norm.append(out_sar_norm)
                    filenames.append(row_id)
            except Exception as e:
                print(f"Error extracting row {row_id}: {e}")
                
            if (idx + 1) % 100 == 0 or (idx + 1) == total:
                print(f"Extracted progress: {idx + 1}/{total}")
                
        opt_embeddings = np.array(opt_embeddings).astype("float32")
        sar_embeddings_asis = np.array(sar_embeddings_asis).astype("float32")
        sar_embeddings_norm = np.array(sar_embeddings_norm).astype("float32")
        
        print(f"Saving extracted embeddings to {emb_path}...")
        np.savez_compressed(
            emb_path, 
            opt=opt_embeddings, 
            sar_asis=sar_embeddings_asis, 
            sar_norm=sar_embeddings_norm, 
            filenames=np.array(filenames)
        )

    # Normalize raw embeddings
    raw_opt_norm = opt_embeddings / np.linalg.norm(opt_embeddings, axis=1, keepdims=True)
    raw_sar_asis_norm = sar_embeddings_asis / np.linalg.norm(sar_embeddings_asis, axis=1, keepdims=True)
    raw_sar_norm_norm = sar_embeddings_norm / np.linalg.norm(sar_embeddings_norm, axis=1, keepdims=True)
    
    # ------------------ MODE A: EVALUATION WITH RAW dB SAR IMAGES ------------------
    print("\n" + "="*80)
    print("MODE A: EVALUATION WITH RAW dB SAR IMAGES (NO SCALING)")
    print("="*80)
    
    raw_metrics_asis = run_faiss_evaluation(
        raw_opt_norm, raw_sar_asis_norm, filenames, filenames, dim=768
    )
    print(f"Raw DINOv2 (asis) Top-1: {raw_metrics_asis['top1']:.2f}% | Top-3: {raw_metrics_asis['top3']:.2f}% | Top-10: {raw_metrics_asis['top10']:.2f}%")

    v1_opt_proj_path = os.path.join(V1_MODEL_DIR, "opt_proj.pt")
    v1_sar_proj_path = os.path.join(V1_MODEL_DIR, "sar_proj.pt")
    
    v1_metrics_asis = None
    if os.path.exists(v1_opt_proj_path) and os.path.exists(v1_sar_proj_path):
        opt_proj = ProjectionHead()
        opt_proj.load_state_dict(torch.load(v1_opt_proj_path, map_location="cpu"))
        opt_proj.eval()
        
        sar_proj = ProjectionHead()
        sar_proj.load_state_dict(torch.load(v1_sar_proj_path, map_location="cpu"))
        sar_proj.eval()
        
        with torch.no_grad():
            proj_opt = opt_proj(torch.tensor(opt_embeddings)).numpy()
            proj_sar = sar_proj(torch.tensor(sar_embeddings_asis)).numpy()
            
        v1_metrics_asis = run_faiss_evaluation(
            proj_opt, proj_sar, filenames, filenames, dim=256
        )
        print(f"V1 Projected (asis) Top-1: {v1_metrics_asis['top1']:.2f}% | Top-3: {v1_metrics_asis['top3']:.2f}% | Top-10: {v1_metrics_asis['top10']:.2f}%")

    v2_opt_proj_path = os.path.join(V2_MODEL_DIR, "opt_proj.pt")
    v2_sar_proj_path = os.path.join(V2_MODEL_DIR, "sar_proj.pt")
    
    v2_metrics_asis = None
    if os.path.exists(v2_opt_proj_path) and os.path.exists(v2_sar_proj_path):
        opt_proj = ProjectionHead()
        opt_proj.load_state_dict(torch.load(v2_opt_proj_path, map_location="cpu"))
        opt_proj.eval()
        
        sar_proj = ProjectionHead()
        sar_proj.load_state_dict(torch.load(v2_sar_proj_path, map_location="cpu"))
        sar_proj.eval()
        
        with torch.no_grad():
            proj_opt = opt_proj(torch.tensor(opt_embeddings)).numpy()
            proj_sar = sar_proj(torch.tensor(sar_embeddings_asis)).numpy()
            
        v2_metrics_asis = run_faiss_evaluation(
            proj_opt, proj_sar, filenames, filenames, dim=256
        )
        print(f"V2 Projected (asis) Top-1: {v2_metrics_asis['top1']:.2f}% | Top-3: {v2_metrics_asis['top3']:.2f}% | Top-10: {v2_metrics_asis['top10']:.2f}%")

    # ------------------ MODE B: EVALUATION WITH PERCENTILE-NORMALIZED SAR IMAGES ------------------
    print("\n" + "="*80)
    print("MODE B: EVALUATION WITH PERCENTILE-NORMALIZED SAR IMAGES (RECOMMENDED)")
    print("="*80)
    
    raw_metrics_norm = run_faiss_evaluation(
        raw_opt_norm, raw_sar_norm_norm, filenames, filenames, dim=768
    )
    print(f"Raw DINOv2 (norm) Top-1: {raw_metrics_norm['top1']:.2f}% | Top-3: {raw_metrics_norm['top3']:.2f}% | Top-10: {raw_metrics_norm['top10']:.2f}%")

    v1_metrics_norm = None
    if os.path.exists(v1_opt_proj_path) and os.path.exists(v1_sar_proj_path):
        opt_proj = ProjectionHead()
        opt_proj.load_state_dict(torch.load(v1_opt_proj_path, map_location="cpu"))
        opt_proj.eval()
        
        sar_proj = ProjectionHead()
        sar_proj.load_state_dict(torch.load(v1_sar_proj_path, map_location="cpu"))
        sar_proj.eval()
        
        with torch.no_grad():
            proj_opt = opt_proj(torch.tensor(opt_embeddings)).numpy()
            proj_sar = sar_proj(torch.tensor(sar_embeddings_norm)).numpy()
            
        v1_metrics_norm = run_faiss_evaluation(
            proj_opt, proj_sar, filenames, filenames, dim=256
        )
        print(f"V1 Projected (norm) Top-1: {v1_metrics_norm['top1']:.2f}% | Top-3: {v1_metrics_norm['top3']:.2f}% | Top-10: {v1_metrics_norm['top10']:.2f}%")

    v2_metrics_norm = None
    if os.path.exists(v2_opt_proj_path) and os.path.exists(v2_sar_proj_path):
        opt_proj = ProjectionHead()
        opt_proj.load_state_dict(torch.load(v2_opt_proj_path, map_location="cpu"))
        opt_proj.eval()
        
        sar_proj = ProjectionHead()
        sar_proj.load_state_dict(torch.load(v2_sar_proj_path, map_location="cpu"))
        sar_proj.eval()
        
        with torch.no_grad():
            proj_opt = opt_proj(torch.tensor(opt_embeddings)).numpy()
            proj_sar = sar_proj(torch.tensor(sar_embeddings_norm)).numpy()
            
        v2_metrics_norm = run_faiss_evaluation(
            proj_opt, proj_sar, filenames, filenames, dim=256
        )
        print(f"V2 Projected (norm) Top-1: {v2_metrics_norm['top1']:.2f}% | Top-3: {v2_metrics_norm['top3']:.2f}% | Top-10: {v2_metrics_norm['top10']:.2f}%")

    # Comparative Prints
    print("\n" + "=" * 80)
    print("COMPARISON ON NEW PREPROCESSED 1000 DATASET:")
    print("=" * 80)
    print("SAR Handling | Metric | Raw DINOv2 | V1 Projected | V2 Projected")
    print("-" * 80)
    if v1_metrics_asis and v2_metrics_asis:
        print(f"Raw dB (asis) | Top-1  | {raw_metrics_asis['top1']:.2f}%\t   | {v1_metrics_asis['top1']:.2f}%\t   | {v2_metrics_asis['top1']:.2f}%")
        print(f"Raw dB (asis) | Top-3  | {raw_metrics_asis['top3']:.2f}%\t   | {v1_metrics_asis['top3']:.2f}%\t   | {v2_metrics_asis['top3']:.2f}%")
        print(f"Raw dB (asis) | Top-10 | {raw_metrics_asis['top10']:.2f}%\t   | {v1_metrics_asis['top10']:.2f}%\t   | {v2_metrics_asis['top10']:.2f}%")
    print("-" * 80)
    if v1_metrics_norm and v2_metrics_norm:
        print(f"V2 Norm (dB->[0,255]) | Top-1  | {raw_metrics_norm['top1']:.2f}%\t   | {v1_metrics_norm['top1']:.2f}%\t   | {v2_metrics_norm['top1']:.2f}%")
        print(f"V2 Norm (dB->[0,255]) | Top-3  | {raw_metrics_norm['top3']:.2f}%\t   | {v1_metrics_norm['top3']:.2f}%\t   | {v2_metrics_norm['top3']:.2f}%")
        print(f"V2 Norm (dB->[0,255]) | Top-10 | {raw_metrics_norm['top10']:.2f}%\t   | {v1_metrics_norm['top10']:.2f}%\t   | {v2_metrics_norm['top10']:.2f}%")
    print("=" * 80)

    # Save metrics JSON & Report txt
    metrics_json_path = os.path.join(V2_RESULT_DIR, "metrics_new_1000.json")
    metrics_data = {
        "mode_asis": {
            "raw_accuracy": raw_metrics_asis,
            "v1_projected_accuracy": v1_metrics_asis,
            "v2_projected_accuracy": v2_metrics_asis
        },
        "mode_normalized": {
            "raw_accuracy": raw_metrics_norm,
            "v1_projected_accuracy": v1_metrics_norm,
            "v2_projected_accuracy": v2_metrics_norm
        }
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
        
    report_txt_path = os.path.join(V2_RESULT_DIR, "evaluation_new_1000.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("=== PIPELINE EVALUATION ON PREPROCESSED_1000 DATASET ===\n")
        f.write(f"Total Images Evaluated: {raw_metrics_asis['total_queries']}\n\n")
        
        f.write("========================================================\n")
        f.write("MODE A: EVALUATION WITH RAW dB SAR IMAGES (NO SCALING)\n")
        f.write("========================================================\n")
        f.write("--- RAW DINOV2 EMBEDDINGS ---\n")
        f.write(f"Top-1  Accuracy: {raw_metrics_asis['top1']:.2f}%\n")
        f.write(f"Top-3  Accuracy: {raw_metrics_asis['top3']:.2f}%\n")
        f.write(f"Top-5  Accuracy: {raw_metrics_asis['top5']:.2f}%\n")
        f.write(f"Top-10 Accuracy: {raw_metrics_asis['top10']:.2f}%\n\n")
        if v1_metrics_asis:
            f.write("--- V1 PROJECTED EMBEDDINGS ---\n")
            f.write(f"Top-1  Accuracy: {v1_metrics_asis['top1']:.2f}%\n")
            f.write(f"Top-3  Accuracy: {v1_metrics_asis['top3']:.2f}%\n")
            f.write(f"Top-5  Accuracy: {v1_metrics_asis['top5']:.2f}%\n")
            f.write(f"Top-10 Accuracy: {v1_metrics_asis['top10']:.2f}%\n\n")
        if v2_metrics_asis:
            f.write("--- V2 PROJECTED EMBEDDINGS ---\n")
            f.write(f"Top-1  Accuracy: {v2_metrics_asis['top1']:.2f}%\n")
            f.write(f"Top-3  Accuracy: {v2_metrics_asis['top3']:.2f}%\n")
            f.write(f"Top-5  Accuracy: {v2_metrics_asis['top5']:.2f}%\n")
            f.write(f"Top-10 Accuracy: {v2_metrics_asis['top10']:.2f}%\n\n")
            
        f.write("========================================================\n")
        f.write("MODE B: EVALUATION WITH PERCENTILE-NORMALIZED SAR IMAGES\n")
        f.write("========================================================\n")
        f.write("--- RAW DINOV2 EMBEDDINGS ---\n")
        f.write(f"Top-1  Accuracy: {raw_metrics_norm['top1']:.2f}%\n")
        f.write(f"Top-3  Accuracy: {raw_metrics_norm['top3']:.2f}%\n")
        f.write(f"Top-5  Accuracy: {raw_metrics_norm['top5']:.2f}%\n")
        f.write(f"Top-10 Accuracy: {raw_metrics_norm['top10']:.2f}%\n\n")
        if v1_metrics_norm:
            f.write("--- V1 PROJECTED EMBEDDINGS ---\n")
            f.write(f"Top-1  Accuracy: {v1_metrics_norm['top1']:.2f}%\n")
            f.write(f"Top-3  Accuracy: {v1_metrics_norm['top3']:.2f}%\n")
            f.write(f"Top-5  Accuracy: {v1_metrics_norm['top5']:.2f}%\n")
            f.write(f"Top-10 Accuracy: {v1_metrics_norm['top10']:.2f}%\n\n")
        if v2_metrics_norm:
            f.write("--- V2 PROJECTED EMBEDDINGS ---\n")
            f.write(f"Top-1  Accuracy: {v2_metrics_norm['top1']:.2f}%\n")
            f.write(f"Top-3  Accuracy: {v2_metrics_norm['top3']:.2f}%\n")
            f.write(f"Top-5  Accuracy: {v2_metrics_norm['top5']:.2f}%\n")
            f.write(f"Top-10 Accuracy: {v2_metrics_norm['top10']:.2f}%\n")

    print(f"Metrics saved to {metrics_json_path}")
    print(f"Report saved to {report_txt_path}")

if __name__ == "__main__":
    main()
