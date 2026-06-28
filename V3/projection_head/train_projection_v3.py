"""
train_projection_v3.py
======================
V3 pipeline: RemoteCLIP (ViT-B-32) encoder + MLP projection head.
Trained on the same 2,000 pairs as V2 (train2_metadata.csv).
Keeps test2 (100 pairs) completely unknown / held-out.

RemoteCLIP checkpoint is downloaded automatically from HuggingFace:
  chendelong/RemoteCLIP  ->  RemoteCLIP-ViT-B-32.pt  (feature dim: 512)

Saves:
  V3/projection_head/opt_proj.pt
  V3/projection_head/sar_proj.pt
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Project root -------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
V3_DIR     = os.path.join(WORKSPACE, "V3")
sys.path.insert(0, os.path.join(V3_DIR, "preprocessing"))
from preprocess_v3 import preprocess_optical_clip, preprocess_sar_clip

# ── Paths --------------------------------------------------------
DATASET_DIR   = os.path.join(WORKSPACE, "dataset")
METADATA_CSV  = os.path.join(DATASET_DIR, "train2_metadata.csv")

V3_EMB_DIR    = os.path.join(V3_DIR, "embeddings")
V3_MODEL_DIR  = os.path.join(V3_DIR, "projection_head")
V3_RESULT_DIR = os.path.join(V3_DIR, "results")
CKPT_DIR      = os.path.join(V3_DIR, "checkpoints")

for d in [V3_EMB_DIR, V3_MODEL_DIR, V3_RESULT_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── RemoteCLIP feature dimension --------------------------------
CLIP_DIM    = 512   # ViT-B-32
PROJ_DIM    = 256


# ── Download RemoteCLIP checkpoint ------------------------------
def get_remote_clip_model():
    ckpt_path = os.path.join(CKPT_DIR, "RemoteCLIP-ViT-B-32.pt")
    if not os.path.exists(ckpt_path):
        print("Downloading RemoteCLIP-ViT-B-32 checkpoint from HuggingFace...")
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(
            repo_id="chendelong/RemoteCLIP",
            filename="RemoteCLIP-ViT-B-32.pt",
            local_dir=CKPT_DIR
        )
        print(f"  Downloaded to {ckpt_path}")
    else:
        print(f"  Found cached RemoteCLIP checkpoint at {ckpt_path}")

    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# ── Architecture ------------------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=CLIP_DIM, output_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * 0.02
        return F.normalize(self.net(x), p=2, dim=-1)


# ── Loss --------------------------------------------------------
def clip_loss(opt_proj, sar_proj, temp=0.07):
    logits = torch.matmul(opt_proj, sar_proj.t()) / temp
    labels = torch.arange(opt_proj.size(0), device=opt_proj.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


# ── Embedding extraction ----------------------------------------
def extract_train_embeddings(model, device):
    emb_path = os.path.join(V3_EMB_DIR, "train2_v3_embeddings.npz")
    if os.path.exists(emb_path):
        print(f"Found cached V3 train embeddings at {emb_path}. Loading...")
        d = np.load(emb_path)
        return d["opt"], d["sar"], list(d["ids"])

    print(f"Reading {METADATA_CSV}...")
    df = pd.read_csv(METADATA_CSV)
    total = len(df)
    print(f"Extracting RemoteCLIP embeddings for {total} V3 training pairs on {device}...")

    model = model.to(device)

    opt_embs, sar_embs, ids = [], [], []
    for idx, row in df.iterrows():
        opt_rel = row["optical_path"]
        sar_rel = row["sar_path"]
        row_id  = row["id"]

        opt_path = os.path.join(DATASET_DIR, opt_rel)
        sar_path = os.path.join(DATASET_DIR, sar_rel)

        # Fallback: img_1801-2000 are in test/ not train/
        if not os.path.exists(opt_path):
            opt_path = opt_path.replace(os.sep + "train" + os.sep, os.sep + "test" + os.sep)
        if not os.path.exists(sar_path):
            sar_path = sar_path.replace(os.sep + "train" + os.sep, os.sep + "test" + os.sep)
        if not os.path.exists(opt_path) or not os.path.exists(sar_path):
            continue

        try:
            with torch.no_grad():
                # Optical
                opt_arr = preprocess_optical_clip(opt_path)
                pv_opt  = torch.tensor(opt_arr).unsqueeze(0).to(device)
                e_opt   = model.encode_image(pv_opt).squeeze().cpu().float().numpy()

                # SAR
                sar_arr = preprocess_sar_clip(sar_path)
                pv_sar  = torch.tensor(sar_arr).unsqueeze(0).to(device)
                e_sar   = model.encode_image(pv_sar).squeeze().cpu().float().numpy()

            opt_embs.append(e_opt)
            sar_embs.append(e_sar)
            ids.append(row_id)
        except Exception as e:
            print(f"  Error at {row_id}: {e}")

        if (idx + 1) % 200 == 0 or (idx + 1) == total:
            print(f"  Progress: {idx+1}/{total}")

    opt_embs = np.array(opt_embs, dtype="float32")
    sar_embs = np.array(sar_embs, dtype="float32")
    np.savez_compressed(emb_path, opt=opt_embs, sar=sar_embs, ids=np.array(ids))
    print(f"Saved V3 train embeddings -> {emb_path}")
    return opt_embs, sar_embs, list(ids)


# ── Accuracy ----------------------------------------------------
def calculate_accuracy(opt_feat, sar_feat):
    scores = np.matmul(opt_feat, sar_feat.T)
    n = scores.shape[0]
    t1 = t3 = t5 = 0
    for i in range(n):
        ranked = np.argsort(-scores[i])[:5]
        if i == ranked[0]:   t1 += 1
        if i in ranked[:3]:  t3 += 1
        if i in ranked[:5]:  t5 += 1
    return t1/n*100, t3/n*100, t5/n*100


# ── Training ----------------------------------------------------
def train():
    print("\n=== V3 RemoteCLIP Pipeline Training ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = get_remote_clip_model()
    opt_raw, sar_raw, ids = extract_train_embeddings(model, device)
    print(f"\nTotal training pairs: {len(ids)}")

    opt_tensor = torch.tensor(opt_raw)
    sar_tensor = torch.tensor(sar_raw)

    n = len(ids)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    split     = int(0.8 * n)
    train_idx = indices[:split]
    val_idx   = indices[split:]
    print(f"Split -> Train: {len(train_idx)} | Val: {len(val_idx)}")

    opt_head  = ProjectionHead()
    sar_head  = ProjectionHead()
    optimizer = torch.optim.AdamW(
        list(opt_head.parameters()) + list(sar_head.parameters()),
        lr=5e-4, weight_decay=1e-3
    )

    epochs     = 120
    batch_size = 128
    best_val   = -1.0
    best_opt_state = best_sar_state = None

    print("\n--- Training V3 RemoteCLIP Projection Head ---")
    for epoch in range(1, epochs + 1):
        opt_head.train(); sar_head.train()
        np.random.shuffle(train_idx)
        losses = []

        for i in range(0, len(train_idx), batch_size):
            idx_b  = train_idx[i:i+batch_size]
            b_opt  = opt_tensor[idx_b]
            b_sar  = sar_tensor[idx_b]
            optimizer.zero_grad()
            loss = clip_loss(opt_head(b_opt), sar_head(b_sar))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        opt_head.eval(); sar_head.eval()
        with torch.no_grad():
            v_opt = opt_head(opt_tensor[val_idx]).numpy()
            v_sar = sar_head(sar_tensor[val_idx]).numpy()
            v1, v3, v5 = calculate_accuracy(v_opt, v_sar)

            t_opt = opt_head(opt_tensor[train_idx]).numpy()
            t_sar = sar_head(sar_tensor[train_idx]).numpy()
            t1, _, _ = calculate_accuracy(t_opt, t_sar)

        if v1 > best_val:
            best_val       = v1
            best_opt_state = {k: v.clone() for k, v in opt_head.state_dict().items()}
            best_sar_state = {k: v.clone() for k, v in sar_head.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Loss: {np.mean(losses):.4f} | "
                  f"Train Top-1: {t1:.1f}% | Val Top-1: {v1:.1f}% (Best: {best_val:.1f}%)")

    # Save best checkpoint
    torch.save(best_opt_state, os.path.join(V3_MODEL_DIR, "opt_proj.pt"))
    torch.save(best_sar_state, os.path.join(V3_MODEL_DIR, "sar_proj.pt"))

    # Final metrics
    opt_head.load_state_dict(best_opt_state); opt_head.eval()
    sar_head.load_state_dict(best_sar_state); sar_head.eval()
    with torch.no_grad():
        f_opt = opt_head(opt_tensor).numpy()
        f_sar = sar_head(sar_tensor).numpy()
        a1, a3, a5 = calculate_accuracy(f_opt, f_sar)

    print("\n" + "=" * 55)
    print(f"V3 RemoteCLIP FINAL ACCURACY ({len(ids)} Training Pairs):")
    print(f"  Top-1 : {a1:.2f}%")
    print(f"  Top-3 : {a3:.2f}%")
    print(f"  Top-5 : {a5:.2f}%")
    print(f"  Best Val Top-1 : {best_val:.2f}%")
    print("=" * 55)
    print(f"Saved V3 projection heads to {V3_MODEL_DIR}")


if __name__ == "__main__":
    train()
