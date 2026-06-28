"""
evaluate_v3.py
==============
V3 Evaluation: Test2 (100 unknown pairs) vs. Restricted Gallery (100 matching SAR).
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
V3_DIR     = os.path.join(WORKSPACE, "V3")
sys.path.insert(0, os.path.join(V3_DIR, "preprocessing"))
from preprocess_v3 import preprocess_optical_clip, preprocess_sar_clip

DATASET_DIR  = os.path.join(WORKSPACE, "dataset")
V3_EMB_DIR   = os.path.join(V3_DIR, "embeddings")
V3_MODEL_DIR = os.path.join(V3_DIR, "projection_head")
CKPT_DIR     = os.path.join(V3_DIR, "checkpoints")
METADATA_CSV = os.path.join(DATASET_DIR, "test2_metadata.csv")
os.makedirs(V3_EMB_DIR, exist_ok=True)

CLIP_DIM = 512
PROJ_DIM = 256


def resolve_path(rel_path):
    """test2_metadata has 'test/optical/...' but files live at 'optical/...'
    Must normpath first to unify mixed slashes on Windows."""
    full = os.path.normpath(os.path.join(DATASET_DIR, rel_path))
    if os.path.exists(full):
        return full
    remapped = full.replace(
        os.sep + "test" + os.sep + "optical" + os.sep, os.sep + "optical" + os.sep
    ).replace(
        os.sep + "test" + os.sep + "sar" + os.sep, os.sep + "sar" + os.sep
    )
    return remapped



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
        return F.normalize(self.net(x), p=2, dim=-1)


def get_remote_clip_model():
    ckpt_path = os.path.join(CKPT_DIR, "RemoteCLIP-ViT-B-32.pt")
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate():
    print("Reading test2_metadata.csv...")
    df = pd.read_csv(METADATA_CSV)

    emb_cache = os.path.join(V3_EMB_DIR, "test2_v3_embeddings.npz")

    if os.path.exists(emb_cache):
        print("Loading cached V3 test2 embeddings...")
        d = np.load(emb_cache)
        opt_embs = d["opt"]
        sar_embs = d["sar"]
    else:
        print("Loading RemoteCLIP model...")
        model = get_remote_clip_model()
        print(f"Extracting RemoteCLIP embeddings for {len(df)} test2 pairs...")

        opt_embs, sar_embs = [], []
        for idx, row in df.iterrows():
            opt_path = resolve_path(row["optical_path"])
            sar_path = resolve_path(row["sar_path"])

            with torch.no_grad():
                opt_arr = preprocess_optical_clip(opt_path)
                pv_opt  = torch.tensor(opt_arr).unsqueeze(0)
                e_opt   = model.encode_image(pv_opt).squeeze().cpu().float().numpy()

                sar_arr = preprocess_sar_clip(sar_path)
                pv_sar  = torch.tensor(sar_arr).unsqueeze(0)
                e_sar   = model.encode_image(pv_sar).squeeze().cpu().float().numpy()

            opt_embs.append(e_opt)
            sar_embs.append(e_sar)

            if (idx + 1) % 25 == 0:
                print(f"  Progress: {idx+1}/{len(df)}")

        opt_embs = np.array(opt_embs, dtype="float32")
        sar_embs = np.array(sar_embs, dtype="float32")
        np.savez_compressed(emb_cache, opt=opt_embs, sar=sar_embs)
        print(f"Cached test2 V3 embeddings -> {emb_cache}")

    n = len(opt_embs)

    # Raw RemoteCLIP
    scores_raw = np.matmul(opt_embs, sar_embs.T)
    t1_r = t3_r = t10_r = 0
    for i in range(n):
        r = np.argsort(-scores_raw[i])
        if i == r[0]:       t1_r  += 1
        if i in r[:3]:      t3_r  += 1
        if i in r[:10]:     t10_r += 1

    print("\n--- Raw RemoteCLIP (No Projection) ---")
    print(f"Top-1: {t1_r/n*100:.2f}% | Top-3: {t3_r/n*100:.2f}% | Top-10: {t10_r/n*100:.2f}%")

    # Projected
    opt_pt = os.path.join(V3_MODEL_DIR, "opt_proj.pt")
    sar_pt = os.path.join(V3_MODEL_DIR, "sar_proj.pt")
    if not os.path.exists(opt_pt):
        print("\nNo V3 projection heads found. Run train_projection_v3.py first.")
        return

    opt_head = ProjectionHead()
    sar_head = ProjectionHead()
    opt_head.load_state_dict(torch.load(opt_pt, map_location="cpu"))
    sar_head.load_state_dict(torch.load(sar_pt, map_location="cpu"))
    opt_head.eval(); sar_head.eval()

    with torch.no_grad():
        p_opt = opt_head(torch.tensor(opt_embs)).numpy()
        p_sar = sar_head(torch.tensor(sar_embs)).numpy()

    scores = np.matmul(p_opt, p_sar.T)
    t1 = t3 = t5 = t10 = 0
    for i in range(n):
        r = np.argsort(-scores[i])
        if i == r[0]:       t1  += 1
        if i in r[:3]:      t3  += 1
        if i in r[:5]:      t5  += 1
        if i in r[:10]:     t10 += 1

    print("\n" + "=" * 55)
    print("V3 RemoteCLIP ACCURACY ON 100 TEST2 PAIRS (Restricted):")
    print(f"  Top-1  : {t1/n*100:.2f}%  (Raw: {t1_r/n*100:.2f}%)")
    print(f"  Top-3  : {t3/n*100:.2f}%  (Raw: {t3_r/n*100:.2f}%)")
    print(f"  Top-5  : {t5/n*100:.2f}%")
    print(f"  Top-10 : {t10/n*100:.2f}%  (Raw: {t10_r/n*100:.2f}%)")
    print("=" * 55)


if __name__ == "__main__":
    evaluate()
