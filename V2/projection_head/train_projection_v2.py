import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATASET_DIR = os.path.join(WORKSPACE, "dataset")
METADATA_CSV = os.path.join(DATASET_DIR, "train2_metadata.csv")

# V2 Isolated directories under root V2 folder
V2_EMB_DIR = os.path.join(WORKSPACE, "V2", "embeddings")
V2_MODEL_DIR = os.path.join(WORKSPACE, "V2", "projection_head")
V2_RESULT_DIR = os.path.join(WORKSPACE, "V2", "results")

os.makedirs(V2_EMB_DIR, exist_ok=True)
os.makedirs(V2_MODEL_DIR, exist_ok=True)
os.makedirs(V2_RESULT_DIR, exist_ok=True)

# 1. MODEL ARCHITECTURE WITH GENERALIZATION REGULARIZATION
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, output_dim=256, dropout_prob=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        # Feature noise jittering in training mode to prevent exact memorization
        if self.training:
            noise = torch.randn_like(x) * 0.02
            x = x + noise
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)

# CLIP-style InfoNCE Contrastive Loss
def clip_loss(opt_proj, sar_proj, temp=0.07):
    # Inputs are already normalized
    logits = torch.matmul(opt_proj, sar_proj.t()) / temp  # [N, N]
    labels = torch.arange(opt_proj.size(0), device=opt_proj.device)
    
    loss_opt = F.cross_entropy(logits, labels)
    loss_sar = F.cross_entropy(logits.t(), labels)
    return (loss_opt + loss_sar) / 2

# Extraction script using the paths in train2_metadata.csv
def extract_train_embeddings():
    emb_path = os.path.join(V2_EMB_DIR, "train2_raw_embeddings.npz")
    if os.path.exists(emb_path):
        print(f"Found cached V2 raw training embeddings at {emb_path}. Loading...")
        data = np.load(emb_path)
        return data["opt"], data["sar"], data["ids"]
        
    print(f"Reading train2_metadata.csv from {METADATA_CSV}...")
    df = pd.read_csv(METADATA_CSV)
    
    print("Loading DINOv2 to extract V2 train embeddings...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    opt_embeddings = []
    sar_embeddings = []
    ids = []
    
    total = len(df)
    print(f"Extracting embeddings for {total} V2 training pairs on {device}...")
    
    for idx, row in df.iterrows():
        # Replace 'train/' with 'train2/' to point to the correct V2 preprocessed folder
        opt_rel = row["optical_path"].replace("train/", "train2/")
        sar_rel = row["sar_path"].replace("train/", "train2/")
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
                ids.append(row_id)
        except Exception as e:
            print(f"Error extracting row {row_id}: {e}")
            
        if (idx + 1) % 200 == 0 or (idx + 1) == total:
            print(f"Extracted progress: {idx + 1}/{total}")
            
    opt_embeddings = np.array(opt_embeddings).astype("float32")
    sar_embeddings = np.array(sar_embeddings).astype("float32")
    ids = np.array(ids)
    
    print(f"Saving extracted V2 training embeddings to {emb_path}...")
    np.savez_compressed(emb_path, opt=opt_embeddings, sar=sar_embeddings, ids=ids)
    
    return opt_embeddings, sar_embeddings, ids

def calculate_accuracy(opt_feat, sar_feat):
    # Computes similarity matrix and finds rank
    scores = np.matmul(opt_feat, sar_feat.T) # [N, N]
    
    top1 = 0
    top3 = 0
    top5 = 0
    n = scores.shape[0]
    
    for i in range(n):
        pred_indices = np.argsort(-scores[i])[:5]
        if i == pred_indices[0]:
            top1 += 1
        if i in pred_indices[:3]:
            top3 += 1
        if i in pred_indices[:5]:
            top5 += 1
            
    return top1/n * 100, top3/n * 100, top5/n * 100

def train():
    opt_raw, sar_raw, ids = extract_train_embeddings()
    
    # Convert to PyTorch tensors
    opt_tensor = torch.tensor(opt_raw)
    sar_tensor = torch.tensor(sar_raw)
    
    # Validation split (80% train, 20% validation) for monitoring generalization
    n_samples = len(ids)
    indices = np.arange(n_samples)
    np.random.seed(42)  # For reproducible splitting
    np.random.shuffle(indices)
    
    train_split = int(0.8 * n_samples)
    train_idx = indices[:train_split]
    val_idx = indices[train_split:]
    
    print(f"V2 Dataset Split: Train={len(train_idx)}, Validation={len(val_idx)}")
    
    # Models
    opt_head = ProjectionHead(dropout_prob=0.4)
    sar_head = ProjectionHead(dropout_prob=0.4)
    
    # Regularization parameters
    optimizer = torch.optim.AdamW(
        list(opt_head.parameters()) + list(sar_head.parameters()), 
        lr=5e-4, 
        weight_decay=1e-3
    )
    
    epochs = 120
    batch_size = 128
    
    best_val_t1 = -1.0
    best_opt_state = None
    best_sar_state = None
    
    print("\n--- Training V2 Regularized Contrastive Projection (Anti-Memorization) ---")
    for epoch in range(1, epochs + 1):
        opt_head.train()
        sar_head.train()
        
        # Shuffle train index
        np.random.shuffle(train_idx)
        epoch_losses = []
        
        for i in range(0, len(train_idx), batch_size):
            batch_indices = train_idx[i:i+batch_size]
            batch_opt = opt_tensor[batch_indices]
            batch_sar = sar_tensor[batch_indices]
            
            optimizer.zero_grad()
            
            proj_opt = opt_head(batch_opt)
            proj_sar = sar_head(batch_sar)
            
            loss = clip_loss(proj_opt, proj_sar, temp=0.07)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
        # Evaluation Mode
        opt_head.eval()
        sar_head.eval()
        with torch.no_grad():
            # Train Acc
            train_opt_proj = opt_head(opt_tensor[train_idx]).numpy()
            train_sar_proj = sar_head(sar_tensor[train_idx]).numpy()
            t1, t3, t5 = calculate_accuracy(train_opt_proj, train_sar_proj)
            
            # Val Acc (unseen data)
            val_opt_proj = opt_head(opt_tensor[val_idx]).numpy()
            val_sar_proj = sar_head(sar_tensor[val_idx]).numpy()
            v1, v3, v5 = calculate_accuracy(val_opt_proj, val_sar_proj)
            
        # Early Stopping / Checkpoint check: Save state if validation Top-1 is better
        if v1 > best_val_t1:
            best_val_t1 = v1
            best_opt_state = opt_head.state_dict().copy()
            best_sar_state = sar_head.state_dict().copy()
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Loss: {np.mean(epoch_losses):.4f} | "
                  f"Train Top-1: {t1:.1f}% | Val Top-1: {v1:.1f}% (Best Val Top-1: {best_val_t1:.1f}%)")
            
    # Save the BEST checkpoint (not the final epoch, to prevent memorization!)
    opt_model_path = os.path.join(V2_MODEL_DIR, "opt_proj.pt")
    sar_model_path = os.path.join(V2_MODEL_DIR, "sar_proj.pt")
    
    torch.save(best_opt_state, opt_model_path)
    torch.save(best_sar_state, sar_model_path)
    
    print("\nTraining completed!")
    print(f"Best Validation Split Top-1 Accuracy: {best_val_t1:.2f}%")
    print("Saved BEST generalizing projection models to:")
    print(" - Optical:", opt_model_path)
    print(" - SAR:", sar_model_path)
    
    # Reload best models to check global train set accuracy
    opt_head.load_state_dict(best_opt_state)
    sar_head.load_state_dict(best_sar_state)
    opt_head.eval()
    sar_head.eval()
    
    with torch.no_grad():
        final_opt = opt_head(opt_tensor).numpy()
        final_sar = sar_head(sar_tensor).numpy()
        all_t1, all_t3, all_t5 = calculate_accuracy(final_opt, final_sar)
        
    print("\n" + "=" * 50)
    print(f"ACCURACY ON FULL V2 TRAINING DATASET ({len(ids)} Images) using Best Checkpoint:")
    print(f"Top-1 Accuracy : {all_t1:.2f}%")
    print(f"Top-3 Accuracy : {all_t3:.2f}%")
    print(f"Top-5 Accuracy : {all_t5:.2f}%")
    print("=" * 50)
    
    # Save log to results/v2/
    log_path = os.path.join(V2_RESULT_DIR, "train_v2_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== V2 PROJECTION HEAD TRAINING LOG ===\n")
        f.write(f"Total Training Samples: {len(ids)}\n")
        f.write(f"Train/Val Split: {len(train_idx)} / {len(val_idx)}\n")
        f.write(f"Best Validation Top-1 Accuracy: {best_val_t1:.2f}%\n")
        f.write("\nFinal Accuracy on Full train2 Dataset:\n")
        f.write(f"Top-1 Accuracy : {all_t1:.2f}%\n")
        f.write(f"Top-3 Accuracy : {all_t3:.2f}%\n")
        f.write(f"Top-5 Accuracy : {all_t5:.2f}%\n")
        
    print(f"Training log written to {log_path}")

if __name__ == "__main__":
    train()
