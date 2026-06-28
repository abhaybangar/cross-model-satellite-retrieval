"""
evaluate_v3_against_all.py
==========================
V3 Evaluation: Test2 (100 unknown queries) vs. Full 2,100-image SAR gallery.
Gallery = train/sar (1800) + test/sar (200) + sar (100 = test2 SAR).
Correct match IS present in the gallery for each query.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
V3_DIR     = os.path.join(WORKSPACE, "V3")
sys.path.insert(0, os.path.join(V3_DIR, "preprocessing"))
from preprocess_v3 import preprocess_optical_clip, preprocess_sar_clip

DATASET_DIR   = os.path.join(WORKSPACE, "dataset")
V3_EMB_DIR    = os.path.join(V3_DIR, "embeddings")
V3_MODEL_DIR  = os.path.join(V3_DIR, "projection_head")
CKPT_DIR      = os.path.join(V3_DIR, "checkpoints")
METADATA_CSV  = os.path.join(DATASET_DIR, "test2_metadata.csv")

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


def extract_gallery_embeddings(model, device):
    """Extract 2,100 SAR embeddings: train(1800) + test(200) + test2(100)."""
    cache_path = os.path.join(V3_EMB_DIR, "combined_gallery_v3.npz")
    if os.path.exists(cache_path):
        print(f"Loading cached V3 gallery embeddings from {cache_path}...")
        d = np.load(cache_path, allow_pickle=True)
        return d["embs"], list(d["names"])

    sar_sources = [
        os.path.join(DATASET_DIR, "train", "sar"),
        os.path.join(DATASET_DIR, "test",  "sar"),
        os.path.join(DATASET_DIR, "sar"),
    ]

    all_paths, all_names = [], []
    for folder in sar_sources:
        files = sorted([f for f in os.listdir(folder) if f.endswith(".tif")])
        for f in files:
            all_paths.append(os.path.join(folder, f))
            all_names.append(f)
        print(f" Found {len(files)} SAR images in {os.path.relpath(folder, WORKSPACE)}")

    print(f"Extracting RemoteCLIP embeddings for {len(all_paths)} gallery images...")
    embs = []
    for i, path in enumerate(all_paths):
        with torch.no_grad():
            arr = preprocess_sar_clip(path)
            pv  = torch.tensor(arr).unsqueeze(0).to(device)
            e   = model.encode_image(pv).squeeze().cpu().float().numpy()
        embs.append(e)
        if (i + 1) % 200 == 0 or (i + 1) == len(all_paths):
            print(f"  Gallery: {i+1}/{len(all_paths)}")

    embs = np.array(embs, dtype="float32")
    np.savez_compressed(cache_path, embs=embs, names=np.array(all_names))
    print(f"Saved V3 gallery embeddings -> {cache_path}")
    return embs, all_names


def evaluate():
    import pandas as pd
    device = torch.device("cpu")

    # Load query embeddings (test2 optical)
    test2_cache = os.path.join(V3_EMB_DIR, "test2_v3_embeddings.npz")
    if os.path.exists(test2_cache):
        print("Loading cached V3 test2 optical embeddings...")
        d = np.load(test2_cache)
        query_opt = d["opt"]
        query_ids = list(pd.read_csv(METADATA_CSV)["id"].astype(str))
    else:
        print("Loading RemoteCLIP model to extract test2 queries...")
        model = get_remote_clip_model().to(device)
        df = pd.read_csv(METADATA_CSV)
        query_opt, query_ids = [], []
        for idx, row in df.iterrows():
            opt_path = resolve_path(row["optical_path"])
            with torch.no_grad():
                arr = preprocess_optical_clip(opt_path)
                pv  = torch.tensor(arr).unsqueeze(0)
                e   = model.encode_image(pv).squeeze().cpu().float().numpy()
            query_opt.append(e)
            query_ids.append(str(row["id"]))
        query_opt = np.array(query_opt, dtype="float32")

    # Load or build gallery
    gallery_cache = os.path.join(V3_EMB_DIR, "combined_gallery_v3.npz")
    if not os.path.exists(gallery_cache):
        model = get_remote_clip_model().to(device)
        gallery_embs, gallery_names = extract_gallery_embeddings(model, device)
    else:
        d2 = np.load(gallery_cache, allow_pickle=True)
        gallery_embs  = d2["embs"]
        gallery_names = list(d2["names"])

    print(f"\nTotal V3 Queries: {len(query_opt)} | Gallery: {len(gallery_embs)} SAR images")

    # Map query filename -> gallery index (correct SAR match)
    df = pd.read_csv(METADATA_CSV)
    query_to_sar = {}
    for _, row in df.iterrows():
        fname = os.path.basename(row["sar_path"])
        query_to_sar[str(row["id"])] = fname

    # Ground truth gallery indices
    gallery_name_to_idx = {n: i for i, n in enumerate(gallery_names)}

    n = len(query_opt)

    # --- Raw RemoteCLIP ---
    scores_raw = np.matmul(query_opt, gallery_embs.T)
    t1_r = t3_r = t10_r = 0
    for i, qid in enumerate(query_ids):
        sar_fname = query_to_sar.get(qid, "")
        gt_idx    = gallery_name_to_idx.get(sar_fname, -1)
        if gt_idx == -1:
            continue
        ranked = np.argsort(-scores_raw[i])
        if gt_idx == ranked[0]:       t1_r  += 1
        if gt_idx in ranked[:3]:      t3_r  += 1
        if gt_idx in ranked[:10]:     t10_r += 1

    print("\n--- Raw RemoteCLIP (No Projection) ---")
    print(f"Top-1: {t1_r/n*100:.2f}% | Top-3: {t3_r/n*100:.2f}% | Top-10: {t10_r/n*100:.2f}%")

    # --- Projected ---
    opt_pt = os.path.join(V3_MODEL_DIR, "opt_proj.pt")
    sar_pt = os.path.join(V3_MODEL_DIR, "sar_proj.pt")
    if not os.path.exists(opt_pt):
        print("\nNo V3 projection heads found. Run train_projection_v3.py first.")
        return

    opt_head = ProjectionHead(); opt_head.load_state_dict(torch.load(opt_pt, map_location="cpu"))
    sar_head = ProjectionHead(); sar_head.load_state_dict(torch.load(sar_pt, map_location="cpu"))
    opt_head.eval(); sar_head.eval()

    with torch.no_grad():
        p_opt  = opt_head(torch.tensor(query_opt)).numpy()
        p_gall = sar_head(torch.tensor(gallery_embs)).numpy()

    scores = np.matmul(p_opt, p_gall.T)
    t1 = t3 = t5 = t10 = 0
    for i, qid in enumerate(query_ids):
        sar_fname = query_to_sar.get(qid, "")
        gt_idx    = gallery_name_to_idx.get(sar_fname, -1)
        if gt_idx == -1:
            continue
        ranked = np.argsort(-scores[i])
        if gt_idx == ranked[0]:       t1  += 1
        if gt_idx in ranked[:3]:      t3  += 1
        if gt_idx in ranked[:5]:      t5  += 1
        if gt_idx in ranked[:10]:     t10 += 1

    print("\n" + "=" * 60)
    print(f"V3 RemoteCLIP: TEST2 vs. ALL ({len(gallery_embs)} SAR images):")
    print(f"  Top-1  : {t1/n*100:.2f}%  (Raw: {t1_r/n*100:.2f}%)")
    print(f"  Top-3  : {t3/n*100:.2f}%  (Raw: {t3_r/n*100:.2f}%)")
    print(f"  Top-5  : {t5/n*100:.2f}%")
    print(f"  Top-10 : {t10/n*100:.2f}%  (Raw: {t10_r/n*100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
