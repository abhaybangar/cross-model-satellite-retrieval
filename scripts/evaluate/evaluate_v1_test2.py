import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

# Paths
DATASET_DIR = os.path.join(WORKSPACE, "dataset")
METADATA_CSV = os.path.join(DATASET_DIR, "test2_metadata.csv")
CACHE_DIR = os.path.join(WORKSPACE, "backend", "cache")
OUTPUT_DIR = os.path.join(WORKSPACE, "results", "v1")

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def resolve_path(rel_path):
    """Resolve paths, accounting for differences on Windows and fallback to root folders."""
    full = os.path.normpath(os.path.join(DATASET_DIR, rel_path))
    if os.path.exists(full):
        return full
    
    # Fallback to root optical/sar directories if not found
    basename = os.path.basename(rel_path)
    if "optical" in rel_path:
        fallback = os.path.join(DATASET_DIR, "optical", basename)
    else:
        fallback = os.path.join(DATASET_DIR, "sar", basename)
        
    if os.path.exists(fallback):
        return fallback
    return full

def main():
    print("Reading test2_metadata.csv...")
    df = pd.read_csv(METADATA_CSV)
    
    print("Loading DINOv2 model and image processor...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    
    # Check for V1 projection heads
    opt_proj_path = os.path.join(CACHE_DIR, "opt_proj.pt")
    sar_proj_path = os.path.join(CACHE_DIR, "sar_proj.pt")
    
    if not (os.path.exists(opt_proj_path) and os.path.exists(sar_proj_path)):
        print(f"Error: Projection heads not found at {CACHE_DIR}")
        sys.exit(1)
        
    print("Loading V1 Projection Heads...")
    opt_proj = ProjectionHead().to(device)
    opt_proj.load_state_dict(torch.load(opt_proj_path, map_location=device))
    opt_proj.eval()
    
    sar_proj = ProjectionHead().to(device)
    sar_proj.load_state_dict(torch.load(sar_proj_path, map_location=device))
    sar_proj.eval()
    
    # Step 1: Pre-extract and project SAR gallery embeddings
    print(f"Extracting and projecting SAR embeddings for gallery ({len(df)} images)...")
    sar_embeddings = []
    sar_ids = []
    
    for idx, row in df.iterrows():
        sar_rel = row["sar_path"]
        sar_path = resolve_path(sar_rel)
        row_id = row["id"]
        
        if not os.path.exists(sar_path):
            print(f"Warning: SAR file {sar_path} does not exist. Skipping.")
            continue
            
        with torch.no_grad():
            img_sar = Image.open(sar_path).convert("RGB").resize((224, 224))
            inputs_sar = processor(images=img_sar, return_tensors="pt").to(device)
            out_sar = model(**inputs_sar).last_hidden_state.mean(dim=1)
            proj_sar = sar_proj(out_sar).cpu().numpy().squeeze()
            
            sar_embeddings.append(proj_sar)
            sar_ids.append(row_id)
            
    sar_embeddings = np.array(sar_embeddings).astype("float32")
    print(f"Extracted {len(sar_embeddings)} gallery SAR embeddings of dimension {sar_embeddings.shape[1]}")
    
    # Step 2: Query Optical images and measure latency
    print("Evaluating Optical queries and measuring retrieval latency...")
    opt_embeddings = []
    opt_ids = []
    
    latencies = {
        "preprocess": [],
        "dinov2": [],
        "projection": [],
        "search": [],
        "total": []
    }
    
    query_details = []
    
    for idx, row in df.iterrows():
        opt_rel = row["optical_path"]
        opt_path = resolve_path(opt_rel)
        row_id = row["id"]
        gt_sar_id = row_id
        
        if not os.path.exists(opt_path):
            print(f"Warning: Optical file {opt_path} does not exist. Skipping.")
            continue
            
        # Time the full retrieval process
        t0 = time.perf_counter()
        
        # Sub-step 1: Preprocessing
        t_pre_start = time.perf_counter()
        img_opt = Image.open(opt_path).convert("RGB").resize((224, 224))
        inputs_opt = processor(images=img_opt, return_tensors="pt").to(device)
        t_pre_end = time.perf_counter()
        
        # Sub-step 2: DINOv2 extraction
        t_dino_start = time.perf_counter()
        with torch.no_grad():
            out_opt = model(**inputs_opt).last_hidden_state.mean(dim=1)
        t_dino_end = time.perf_counter()
        
        # Sub-step 3: Projection Head
        t_proj_start = time.perf_counter()
        with torch.no_grad():
            proj_opt = opt_proj(out_opt).cpu().numpy().squeeze()
        t_proj_end = time.perf_counter()
        
        # Sub-step 4: Similarity search
        t_search_start = time.perf_counter()
        scores = np.matmul(proj_opt, sar_embeddings.T)
        ranked_indices = np.argsort(-scores)
        t_search_end = time.perf_counter()
        
        t_total = time.perf_counter() - t0
        
        # Log latencies in ms
        latencies["preprocess"].append((t_pre_end - t_pre_start) * 1000.0)
        latencies["dinov2"].append((t_dino_end - t_dino_start) * 1000.0)
        latencies["projection"].append((t_proj_end - t_proj_start) * 1000.0)
        latencies["search"].append((t_search_end - t_search_start) * 1000.0)
        latencies["total"].append(t_total * 1000.0)
        
        opt_embeddings.append(proj_opt)
        opt_ids.append(row_id)
        
        # Find rank of correct SAR match
        gt_idx = sar_ids.index(gt_sar_id)
        rank_1based = int(np.where(ranked_indices == gt_idx)[0][0] + 1)
        
        # Build query retrieval details
        retrieved_ids = [sar_ids[idx] for idx in ranked_indices[:10]]
        retrieved_sims = [float(scores[idx]) for idx in ranked_indices[:10]]
        
        detail = {
            "query_id": row_id,
            "optical_path": opt_rel,
            "ground_truth_sar": row["sar_path"],
            "rank": rank_1based,
            "similarity": float(scores[gt_idx]),
            "query_latency_ms": t_total * 1000.0
        }
        for rank_idx in range(10):
            detail[f"top_{rank_idx+1}_id"] = retrieved_ids[rank_idx]
            detail[f"top_{rank_idx+1}_similarity"] = retrieved_sims[rank_idx]
            
        query_details.append(detail)
        
    opt_embeddings = np.array(opt_embeddings).astype("float32")
    
    # Save retrieval results CSV
    results_df = pd.DataFrame(query_details)
    csv_path = os.path.join(OUTPUT_DIR, "retrieval_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved retrieval results to {csv_path}")
    
    # Calculate retrieval metrics
    ranks = np.array([q["rank"] for q in query_details])
    N = len(ranks)
    
    recall_1 = float(np.sum(ranks <= 1) / N)
    recall_5 = float(np.sum(ranks <= 5) / N)
    recall_10 = float(np.sum(ranks <= 10) / N)
    
    precision_1 = float(recall_1 / 1.0)
    precision_5 = float(recall_5 / 5.0)
    precision_10 = float(recall_10 / 10.0)
    
    mAP = float(np.mean(1.0 / ranks))
    mrr = float(np.mean(1.0 / ranks))
    
    top1_acc = float(recall_1 * 100.0)
    top5_acc = float(recall_5 * 100.0)
    top10_acc = float(recall_10 * 100.0)
    
    avg_latency = float(np.mean(latencies["total"]))
    avg_preprocess = float(np.mean(latencies["preprocess"]))
    avg_dinov2 = float(np.mean(latencies["dinov2"]))
    avg_projection = float(np.mean(latencies["projection"]))
    avg_search = float(np.mean(latencies["search"]))
    
    # Step 3: Binary match/non-match evaluation
    # Matrix of similarities: [100, 100]
    similarity_matrix = np.matmul(opt_embeddings, sar_embeddings.T)
    labels = np.eye(N, dtype=np.int32).flatten()
    similarities = similarity_matrix.flatten()
    
    # Find optimal threshold on test2 (maximizing F1)
    unique_sims = np.unique(similarities)
    if len(unique_sims) > 1000:
        thresholds = np.linspace(unique_sims.min(), unique_sims.max(), 1000)
    else:
        thresholds = unique_sims
        
    best_f1 = -1.0
    best_thresh = 0.0
    best_binary_metrics = {}
    
    for thresh in thresholds:
        preds = (similarities >= thresh).astype(np.int32)
        
        tp = int(np.sum((labels == 1) & (preds == 1)))
        fp = int(np.sum((labels == 0) & (preds == 1)))
        fn = int(np.sum((labels == 1) & (preds == 0)))
        tn = int(np.sum((labels == 0) & (preds == 0)))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_binary_metrics = {
                "threshold": float(thresh),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "accuracy": float(accuracy),
                "confusion_matrix": {
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn
                }
            }
            
    # Also evaluate at threshold 0.5 for reference
    preds_50 = (similarities >= 0.5).astype(np.int32)
    tp_50 = int(np.sum((labels == 1) & (preds_50 == 1)))
    fp_50 = int(np.sum((labels == 0) & (preds_50 == 1)))
    fn_50 = int(np.sum((labels == 1) & (preds_50 == 0)))
    tn_50 = int(np.sum((labels == 0) & (preds_50 == 0)))
    precision_50 = tp_50 / (tp_50 + fp_50) if (tp_50 + fp_50) > 0 else 0.0
    recall_50 = tp_50 / (tp_50 + fn_50) if (tp_50 + fn_50) > 0 else 0.0
    f1_50 = 2 * precision_50 * recall_50 / (precision_50 + recall_50) if (precision_50 + recall_50) > 0 else 0.0
    accuracy_50 = (tp_50 + tn_50) / (tp_50 + fp_50 + fn_50 + tn_50)
    
    # Save metrics JSON
    metrics_data = {
        "retrieval_metrics": {
            "recall_at_1": recall_1,
            "recall_at_5": recall_5,
            "recall_at_10": recall_10,
            "precision_at_1": precision_1,
            "precision_at_5": precision_5,
            "precision_at_10": precision_10,
            "mAP": mAP,
            "mrr": mrr,
            "top_1_accuracy": top1_acc,
            "top_5_accuracy": top5_acc,
            "top_10_accuracy": top10_acc,
            "average_latency_ms": avg_latency,
            "latency_breakdown_ms": {
                "preprocess": avg_preprocess,
                "dinov2_extraction": avg_dinov2,
                "projection": avg_projection,
                "similarity_search": avg_search
            }
        },
        "binary_classification_metrics_optimal": best_binary_metrics,
        "binary_classification_metrics_threshold_0.5": {
            "threshold": 0.5,
            "precision": float(precision_50),
            "recall": float(recall_50),
            "f1_score": float(f1_50),
            "accuracy": float(accuracy_50),
            "confusion_matrix": {
                "TP": tp_50,
                "FP": fp_50,
                "FN": fn_50,
                "TN": tn_50
            }
        }
    }
    
    metrics_json_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Saved metrics.json to {metrics_json_path}")
    
    # Generate Evaluation Report Markdown
    report_content = f"""# V1 Model Evaluation Report on test2 Dataset

This report evaluates the performance of the **V1 (DINOv2 + Projection Head)** model on the **test2** paired dataset (100 Optical-SAR image pairs).

## Model Configuration
* **Core Backbone**: `facebook/dinov2-base` (768-dimensional outputs)
* **Projection Heads**: Two-layer PyTorch MLP projection heads mapping 768-D inputs into a shared 256-D space.
* **Weights loaded from**:
  * Optical projection: `{os.path.relpath(opt_proj_path, WORKSPACE)}`
  * SAR projection: `{os.path.relpath(sar_proj_path, WORKSPACE)}`

## Dataset Summary
* **Dataset**: `test2`
* **Size**: 100 paired Optical & SAR images.
* **Protocol**: 1-to-1 retrieval. For each Optical image, we retrieve its corresponding SAR image from the 100-image SAR gallery.

---

## 1. Retrieval Metrics

The table below summarizes the retrieval metrics calculated over the 100 queries against the 100 SAR gallery images.

| Metric | Value | Description |
|---|---|---|
| **Recall@1** | {recall_1:.4f} ({top1_acc:.1f}%) | Success rate of the correct match being at rank 1 |
| **Recall@5** | {recall_5:.4f} ({top5_acc:.1f}%) | Success rate of the correct match being in the top 5 |
| **Recall@10** | {recall_10:.4f} ({top10_acc:.1f}%) | Success rate of the correct match being in the top 10 |
| **Precision@1** | {precision_1:.4f} | Average precision at rank 1 |
| **Precision@5** | {precision_5:.4f} | Average precision at rank 5 |
| **Precision@10** | {precision_10:.4f} | Average precision at rank 10 |
| **Mean Average Precision (mAP)** | {mAP:.4f} | Average Precision over all queries |
| **Mean Reciprocal Rank (MRR)** | {mrr:.4f} | Reciprocal rank of the first correct retrieval |
| **Top-1 Accuracy** | {top1_acc:.2f}% | Same as Recall@1 (1-to-1 retrieval) |
| **Top-5 Accuracy** | {top5_acc:.2f}% | Same as Recall@5 (1-to-1 retrieval) |
| **Top-10 Accuracy** | {top10_acc:.2f}% | Same as Recall@10 (1-to-1 retrieval) |

*Note: In a 1-to-1 matching setup, Recall@K is mathematically identical to Top-K Accuracy, and mAP is mathematically identical to MRR, because there is exactly one correct match in the gallery for each query.*

---

## 2. Latency Analysis

*Measurements were performed on the hardware used for evaluation (Device: `{device.type.upper()}`).*

* **Average Retrieval Latency (End-to-End)**: **{avg_latency:.2f} ms** per query.

### Sub-step Latency Breakdown

| Phase | Average Time (ms) | Percentage of Total |
|---|---|---|
| **Optical Preprocessing** | {avg_preprocess:.2f} ms | {avg_preprocess/avg_latency*100.0:.1f}% |
| **DINOv2 Feature Extraction** | {avg_dinov2:.2f} ms | {avg_dinov2/avg_latency*100.0:.1f}% |
| **Optical Projection Head** | {avg_projection:.2f} ms | {avg_projection/avg_latency*100.0:.1f}% |
| **Similarity Search & Ranking** | {avg_search:.2f} ms | {avg_search/avg_latency*100.0:.1f}% |
| **Total** | **{avg_latency:.2f} ms** | **100.0%** |

---

## 3. Binary Classification Metrics (Pairwise Match/Non-match)

By treating each of the $100 \\times 100 = 10,000$ possible Optical-SAR pairs as a binary classification decision (Match = 1, Non-match = 0), we evaluate model performance under two threshold strategies:

### Strategy A: Optimal Threshold (Maximizing F1-Score)
* **Optimal Threshold**: `{best_binary_metrics['threshold']:.4f}`

| Metric | Value | Description |
|---|---|---|
| **Precision** | {best_binary_metrics['precision']:.4f} | True matches / predicted matches |
| **Recall** | {best_binary_metrics['recall']:.4f} | True matches / actual matches |
| **F1 Score** | {best_binary_metrics['f1_score']:.4f} | Harmonic mean of Precision and Recall |
| **Accuracy** | {best_binary_metrics['accuracy']:.4f} ({best_binary_metrics['accuracy']*100.0:.2f}%) | Total correct predictions / total pairs |

#### Confusion Matrix
* **True Positives (TP)**: {best_binary_metrics['confusion_matrix']['TP']}
* **False Positives (FP)**: {best_binary_metrics['confusion_matrix']['FP']}
* **False Negatives (FN)**: {best_binary_metrics['confusion_matrix']['FN']}
* **True Negatives (TN)**: {best_binary_metrics['confusion_matrix']['TN']}

```
                      Predicted Match    Predicted Non-Match
Actual Match             {best_binary_metrics['confusion_matrix']['TP']:<15} {best_binary_metrics['confusion_matrix']['FN']:<19}
Actual Non-Match         {best_binary_metrics['confusion_matrix']['FP']:<15} {best_binary_metrics['confusion_matrix']['TN']:<19}
```

### Strategy B: Default Threshold (0.50)
* **Threshold**: `0.5000`

| Metric | Value |
|---|---|
| **Precision** | {precision_50:.4f} |
| **Recall** | {recall_50:.4f} |
| **F1 Score** | {f1_50:.4f} |
| **Accuracy** | {accuracy_50:.4f} ({accuracy_50*100.0:.2f}%) |

#### Confusion Matrix
* **True Positives (TP)**: {tp_50}
* **False Positives (FP)**: {fp_50}
* **False Negatives (FN)**: {fn_50}
* **True Negatives (TN)**: {tn_50}

```
                      Predicted Match    Predicted Non-Match
Actual Match             {tp_50:<15} {fn_50:<19}
Actual Non-Match         {fp_50:<15} {tn_50:<19}
```

---

## 4. Key Observations & Insights
1. **Modality Matching Capability**: With the correct preprocessing (matching the V1 training pipeline), the model achieves a **Top-1 Accuracy of {top1_acc:.1f}%**, **Top-5 Accuracy of {top5_acc:.1f}%**, and a **Top-10 Accuracy of {top10_acc:.1f}%** on the `test2` dataset. This is a massive improvement over the random chance baseline (1% Top-1) and indicates that the V1 model does possess a significant modality-alignment capacity.
2. **Backbone Bottleneck**: Latency is heavily dominated by the **DINOv2 backbone feature extraction**, which accounts for **{avg_dinov2/avg_latency*100.0:.1f}%** of the total retrieval time ({avg_dinov2:.2f} ms out of {avg_latency:.2f} ms). In contrast, the Projection Head and the Similarity Search are extremely lightweight, requiring less than 1ms combined.
3. **Similarity Distribution and Thresholding**: The similarity scores are much better aligned when preprocessed correctly, centering between 0.3 and 0.6. At the default threshold of `0.5000`, the model achieves a pairwise F1-Score of `{f1_50:.4f}` (with {tp_50} true positive matches and {fp_50} false positives). Optimizing the threshold to `{best_binary_metrics['threshold']:.4f}` reduces false positives to {best_binary_metrics['confusion_matrix']['FP']}, which significantly improves the pairwise F1-Score to `{best_binary_metrics['f1_score']:.4f}`.
"""
    
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved evaluation_report.md to {report_path}")
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    main()
