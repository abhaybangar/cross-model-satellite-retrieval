import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)
from ben_preprocess import preprocess_optical, preprocess_sar

# Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OPT_DIR = os.path.join(WORKSPACE, "dataset", "optical")
SAR_DIR = os.path.join(WORKSPACE, "dataset", "sar")
CACHE_DIR = os.path.join(WORKSPACE, "backend", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 1. ARCHITECTURE DEFINITION
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
        # L2 normalize projection outputs
        return F.normalize(z, p=2, dim=-1)

# Loss function: Symmetric CLIP-style InfoNCE
def clip_loss(opt_proj, sar_proj, temp=0.05):
    # opt_proj, sar_proj are shape [N, D] and already L2-normalized
    logits = torch.matmul(opt_proj, sar_proj.t()) / temp  # [N, N]
    labels = torch.arange(opt_proj.size(0), device=opt_proj.device)
    
    loss_opt = F.cross_entropy(logits, labels)
    loss_sar = F.cross_entropy(logits.t(), labels)
    return (loss_opt + loss_sar) / 2

def extract_all_embeddings():
    emb_path = os.path.join(CACHE_DIR, "raw_dinov2_embeddings.npz")
    if os.path.exists(emb_path):
        print("Found cached raw DINOv2 embeddings. Loading...")
        data = np.load(emb_path)
        return data["opt"], data["sar"], data["filenames"]
        
    print("Loading DINOv2 model to extract embeddings...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    opt_files = set(os.listdir(OPT_DIR))
    sar_files = set(os.listdir(SAR_DIR))
    matching_files = sorted(list(opt_files.intersection(sar_files)))
    
    print(f"Extracting embeddings for {len(matching_files)} pairs on {device}...")
    opt_embeddings = []
    sar_embeddings = []
    
    for idx, filename in enumerate(matching_files):
        opt_path = os.path.join(OPT_DIR, filename)
        sar_path = os.path.join(SAR_DIR, filename)
        
        with torch.no_grad():
            # Optical
            opt_array = preprocess_optical(opt_path)
            pixel_values_opt = torch.tensor(opt_array).unsqueeze(0).to(device)
            out_opt = model(pixel_values=pixel_values_opt).last_hidden_state.mean(dim=1).cpu().numpy()
            opt_embeddings.append(out_opt.squeeze())
            
            # SAR
            sar_array = preprocess_sar(sar_path)
            pixel_values_sar = torch.tensor(sar_array).unsqueeze(0).to(device)
            out_sar = model(pixel_values=pixel_values_sar).last_hidden_state.mean(dim=1).cpu().numpy()
            sar_embeddings.append(out_sar.squeeze())
            
        if (idx + 1) % 100 == 0 or (idx + 1) == len(matching_files):
            print(f"Extracted progress: {idx + 1}/{len(matching_files)}")
            
    opt_embeddings = np.array(opt_embeddings).astype("float32")
    sar_embeddings = np.array(sar_embeddings).astype("float32")
    
    print("Saving raw DINOv2 embeddings to cache...")
    np.savez_compressed(emb_path, opt=opt_embeddings, sar=sar_embeddings, filenames=matching_files)
    
    return opt_embeddings, sar_embeddings, matching_files

def calculate_accuracy(opt_feat, sar_feat):
    # opt_feat: [N, D] normalized
    # sar_feat: [N, D] normalized
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
    opt_raw, sar_raw, filenames = extract_all_embeddings()
    
    # Convert to PyTorch tensors
    opt_tensor = torch.tensor(opt_raw)
    sar_tensor = torch.tensor(sar_raw)
    
    # Train-test split (80% train, 20% validation)
    n_samples = len(filenames)
    indices = np.arange(n_samples)
    # Using fixed seed for reproducibility
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_split = int(0.8 * n_samples)
    train_idx = indices[:train_split]
    val_idx = indices[train_split:]
    
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
    
    # Define models
    opt_head = ProjectionHead()
    sar_head = ProjectionHead()
    
    optimizer = torch.optim.AdamW(
        list(opt_head.parameters()) + list(sar_head.parameters()), 
        lr=3e-4, 
        weight_decay=1e-4
    )
    
    epochs = 60
    batch_size = 128
    
    print("\n--- Training InfoNCE Projection Head ---")
    for epoch in range(1, epochs + 1):
        opt_head.train()
        sar_head.train()
        
        # Shuffle training set
        np.random.shuffle(train_idx)
        epoch_losses = []
        
        for i in range(0, len(train_idx), batch_size):
            batch_indices = train_idx[i:i+batch_size]
            batch_opt = opt_tensor[batch_indices]
            batch_sar = sar_tensor[batch_indices]
            
            optimizer.zero_grad()
            
            proj_opt = opt_head(batch_opt)
            proj_sar = sar_head(batch_sar)
            
            loss = clip_loss(proj_opt, proj_sar)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
        # Validate accuracy
        opt_head.eval()
        sar_head.eval()
        with torch.no_grad():
            # Train Acc
            train_opt_proj = opt_head(opt_tensor[train_idx]).numpy()
            train_sar_proj = sar_head(sar_tensor[train_idx]).numpy()
            t1, t3, t5 = calculate_accuracy(train_opt_proj, train_sar_proj)
            
            # Val Acc
            val_opt_proj = opt_head(opt_tensor[val_idx]).numpy()
            val_sar_proj = sar_head(sar_tensor[val_idx]).numpy()
            v1, v3, v5 = calculate_accuracy(val_opt_proj, val_sar_proj)
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | Loss: {np.mean(epoch_losses):.4f} | "
                  f"Train Top-1: {t1:.1f}% | Val Top-1: {v1:.1f}% (Top-5: {v5:.1f}%)")
            
    # Save the models
    opt_model_path = os.path.join(CACHE_DIR, "opt_proj.pt")
    sar_model_path = os.path.join(CACHE_DIR, "sar_proj.pt")
    
    torch.save(opt_head.state_dict(), opt_model_path)
    torch.save(sar_head.state_dict(), sar_model_path)
    
    print("\nSaved projection heads to:")
    print(" - Optical:", opt_model_path)
    print(" - SAR:", sar_model_path)
    
    # Calculate Overall Accuracy
    with torch.no_grad():
        final_opt = opt_head(opt_tensor).numpy()
        final_sar = sar_head(sar_tensor).numpy()
        all_t1, all_t3, all_t5 = calculate_accuracy(final_opt, final_sar)
        
    print("\n" + "=" * 50)
    print(f"FINAL PROJECTED ACCURACY (1000 Images):")
    print(f"Top-1 Accuracy : {all_t1:.2f}%")
    print(f"Top-3 Accuracy : {all_t3:.2f}%")
    print(f"Top-5 Accuracy : {all_t5:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    train()
