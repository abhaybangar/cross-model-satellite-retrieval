"""
train_projection_combined.py
============================
Retrains the V1 projection head on ALL available data:
  - 1,800 pairs from dataset/train  (cached in backend/cache/train_raw_embeddings.npz)
  - 200  pairs from dataset/test    (extracted on the fly if not yet cached)
  -----------------------------------------------
  Total: 2,000 pairs
  Held-out (NEVER touched): dataset/optical + dataset/sar (test2, 100 pairs)

Saves to: backend/cache/opt_proj.pt and backend/cache/sar_proj.pt
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)
from ben_preprocess import preprocess_optical, preprocess_sar

# ── Paths ────────────────────────────────────────────────────────────
DATASET_DIR      = os.path.join(WORKSPACE, "dataset")
CACHE_DIR        = os.path.join(WORKSPACE, "backend", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TRAIN_CACHE      = os.path.join(CACHE_DIR, "train_raw_embeddings.npz")        # 1,800 items
TEST_EMB_CACHE   = os.path.join(CACHE_DIR, "test_raw_embeddings.npz")         # 200 items  (created here)
COMBINED_CACHE   = os.path.join(CACHE_DIR, "combined_2000_embeddings.npz")    # 2,000 items (created here)

TEST_OPT_DIR     = os.path.join(DATASET_DIR, "test", "optical")
TEST_SAR_DIR     = os.path.join(DATASET_DIR, "test", "sar")

# ── Architecture ─────────────────────────────────────────────────────
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
        if self.training:
            x = x + torch.randn_like(x) * 0.02   # feature jitter → anti-memorization
        return F.normalize(self.net(x), p=2, dim=-1)

# ── Loss ─────────────────────────────────────────────────────────────
def clip_loss(opt_proj, sar_proj, temp=0.05):
    logits = torch.matmul(opt_proj, sar_proj.t()) / temp
    labels = torch.arange(opt_proj.size(0), device=opt_proj.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

# ── Embedding extraction: test set (200 pairs) ───────────────────────
def extract_test_embeddings():
    if os.path.exists(TEST_EMB_CACHE):
        print(f"[test] Loaded cached test embeddings from {TEST_EMB_CACHE}")
        d = np.load(TEST_EMB_CACHE)
        return d["opt"], d["sar"], list(d["ids"])

    print("[test] Extracting DINOv2 embeddings for test set (200 pairs)…")
    from transformers import AutoModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()

    opt_files = sorted([f for f in os.listdir(TEST_OPT_DIR) if f.endswith(".tif")])
    sar_files = sorted([f for f in os.listdir(TEST_SAR_DIR) if f.endswith(".tif")])
    # Keep only matched filenames
    matched   = sorted(set(opt_files) & set(sar_files))
    print(f"  Found {len(matched)} matched pairs in dataset/test/")

    opt_embs, sar_embs, ids = [], [], []
    for idx, fname in enumerate(matched):
        opt_path = os.path.join(TEST_OPT_DIR, fname)
        sar_path = os.path.join(TEST_SAR_DIR, fname)
        with torch.no_grad():
            opt_arr  = preprocess_optical(opt_path)
            pv_opt   = torch.tensor(opt_arr).unsqueeze(0).to(device)
            e_opt    = model(pixel_values=pv_opt).last_hidden_state.mean(1).squeeze().cpu().numpy()

            sar_arr  = preprocess_sar(sar_path)
            pv_sar   = torch.tensor(sar_arr).unsqueeze(0).to(device)
            e_sar    = model(pixel_values=pv_sar).last_hidden_state.mean(1).squeeze().cpu().numpy()

        opt_embs.append(e_opt.astype("float32"))
        sar_embs.append(e_sar.astype("float32"))
        ids.append(fname)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(matched):
            print(f"  Progress: {idx+1}/{len(matched)}")

    opt_embs = np.array(opt_embs, dtype="float32")
    sar_embs = np.array(sar_embs, dtype="float32")
    np.savez_compressed(TEST_EMB_CACHE, opt=opt_embs, sar=sar_embs, ids=np.array(ids))
    print(f"  Saved test embeddings -> {TEST_EMB_CACHE}")
    return opt_embs, sar_embs, ids

# ── Build 2,000-pair combined embedding set ───────────────────────────
def build_combined_embeddings():
    if os.path.exists(COMBINED_CACHE):
        print(f"[combined] Loaded cached 2,000-pair embeddings from {COMBINED_CACHE}")
        d = np.load(COMBINED_CACHE)
        return d["opt"], d["sar"], list(d["ids"])

    # 1,800 train embeddings (already cached)
    print(f"[train]  Loading 1,800 train embeddings from {TRAIN_CACHE} …")
    d_train    = np.load(TRAIN_CACHE)
    train_opt  = d_train["opt"].astype("float32")
    train_sar  = d_train["sar"].astype("float32")
    train_ids  = list(d_train["ids"])
    print(f"  Loaded {len(train_ids)} train pairs")

    # 200 test embeddings
    test_opt, test_sar, test_ids = extract_test_embeddings()

    # Concatenate
    combined_opt = np.concatenate([train_opt, test_opt], axis=0)
    combined_sar = np.concatenate([train_sar, test_sar], axis=0)
    combined_ids = train_ids + list(test_ids)

    np.savez_compressed(COMBINED_CACHE,
                        opt=combined_opt, sar=combined_sar,
                        ids=np.array(combined_ids))
    print(f"[combined] Saved {len(combined_ids)} combined embeddings -> {COMBINED_CACHE}")
    return combined_opt, combined_sar, combined_ids

# ── Accuracy helper ───────────────────────────────────────────────────
def calculate_accuracy(opt_feat, sar_feat):
    scores = np.matmul(opt_feat, sar_feat.T)
    n = scores.shape[0]
    t1 = t3 = t5 = 0
    for i in range(n):
        ranked = np.argsort(-scores[i])[:5]
        if i == ranked[0]:        t1 += 1
        if i in ranked[:3]:       t3 += 1
        if i in ranked[:5]:       t5 += 1
    return t1/n*100, t3/n*100, t5/n*100

# ── Training ──────────────────────────────────────────────────────────
def train():
    opt_raw, sar_raw, ids = build_combined_embeddings()
    print(f"\n[train] Total training pairs: {len(ids)}")
    assert len(ids) == 2000, f"Expected 2000 pairs, got {len(ids)}"

    opt_tensor = torch.tensor(opt_raw)
    sar_tensor = torch.tensor(sar_raw)

    # 80/20 split
    n = len(ids)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    split      = int(0.8 * n)
    train_idx  = indices[:split]
    val_idx    = indices[split:]
    print(f"[train] Split -> Train: {len(train_idx)} | Val: {len(val_idx)}")

    opt_head   = ProjectionHead()
    sar_head   = ProjectionHead()
    optimizer  = torch.optim.AdamW(
        list(opt_head.parameters()) + list(sar_head.parameters()),
        lr=3e-4, weight_decay=1e-4
    )

    epochs     = 80
    batch_size = 128
    best_val   = -1.0
    best_opt_state = best_sar_state = None

    print("\n--- Training V1 Combined Projection Head (2,000 pairs) ---")
    for epoch in range(1, epochs + 1):
        opt_head.train(); sar_head.train()
        np.random.shuffle(train_idx)
        losses = []

        for i in range(0, len(train_idx), batch_size):
            idx_b   = train_idx[i:i+batch_size]
            b_opt   = opt_tensor[idx_b]
            b_sar   = sar_tensor[idx_b]
            optimizer.zero_grad()
            loss    = clip_loss(opt_head(b_opt), sar_head(b_sar))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Validation
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
    opt_path = os.path.join(CACHE_DIR, "opt_proj.pt")
    sar_path = os.path.join(CACHE_DIR, "sar_proj.pt")
    torch.save(best_opt_state, opt_path)
    torch.save(best_sar_state, sar_path)

    # Final metrics on all 2,000 pairs
    opt_head.load_state_dict(best_opt_state); opt_head.eval()
    sar_head.load_state_dict(best_sar_state); sar_head.eval()
    with torch.no_grad():
        f_opt = opt_head(opt_tensor).numpy()
        f_sar = sar_head(sar_tensor).numpy()
        a1, a3, a5 = calculate_accuracy(f_opt, f_sar)

    print("\n" + "=" * 55)
    print(f"V1 FINAL ACCURACY (2,000 Training Pairs):")
    print(f"  Top-1 : {a1:.2f}%")
    print(f"  Top-3 : {a3:.2f}%")
    print(f"  Top-5 : {a5:.2f}%")
    print(f"  Best Val Top-1 : {best_val:.2f}%")
    print("=" * 55)
    print(f"Saved best V1 projection heads to {CACHE_DIR}")

if __name__ == "__main__":
    train()
