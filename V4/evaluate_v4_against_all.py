import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from v4_common import (
    WORKSPACE,
    build_arg_parser,
    encode_image,
    ensure_dirs,
    load_config,
    load_projection_heads,
    load_remoteclip_v4,
    preprocess_sar,
    retrieval_metrics,
    resolve_dataset_path,
    PairDataset,
)


class SarGalleryDataset(Dataset):
    def __init__(self):
        roots = [WORKSPACE / "dataset" / "train" / "sar", WORKSPACE / "dataset" / "test" / "sar", WORKSPACE / "dataset" / "sar"]
        self.paths = []
        self.names = []
        for root in roots:
            for path in sorted(root.glob("*.tif")):
                self.paths.append(path)
                self.names.append(path.name)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        arr = preprocess_sar(str(self.paths[idx]))
        return {"sar": torch.tensor(arr, dtype=torch.float32), "name": self.names[idx]}


def encode_gallery(model, sar_head, cfg, device):
    cache_path = Path(cfg["embedding_dir"]) / "combined_gallery_v4.npz"
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        return d["raw"], d["proj"], list(d["names"])

    ds = SarGalleryDataset()
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)
    raw, proj, names = [], [], []
    with torch.no_grad():
        for batch in loader:
            sar = batch["sar"].to(device)
            sar_e = encode_image(model, sar)
            raw.append(sar_e.cpu().numpy())
            proj.append(sar_head(sar_e).cpu().numpy())
            names.extend(list(batch["name"]))
    raw = np.vstack(raw).astype("float32")
    proj = np.vstack(proj).astype("float32")
    np.savez_compressed(cache_path, raw=raw, proj=proj, names=np.array(names))
    return raw, proj, names


def evaluate():
    parser = build_arg_parser("Evaluate V4 test2 queries against all 2100 SAR images")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.no_lora:
        cfg["use_lora"] = False
    ensure_dirs(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_remoteclip_v4(cfg, device, load_adapters=cfg.get("use_lora", True))
    opt_head, sar_head = load_projection_heads(cfg, device)
    model.eval()

    test_ds = PairDataset(cfg["test_metadata"])
    test_loader = DataLoader(test_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)
    query_raw, query_proj, query_ids = [], [], []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            opt = batch["opt"].to(device)
            opt_e = encode_image(model, opt)
            query_raw.append(opt_e.cpu().numpy())
            query_proj.append(opt_head(opt_e).cpu().numpy())
            query_ids.extend(list(batch["id"]))
    query_latency_ms = (time.perf_counter() - start) * 1000.0 / max(len(test_ds), 1)

    query_raw = np.vstack(query_raw).astype("float32")
    query_proj = np.vstack(query_proj).astype("float32")
    gallery_raw, gallery_proj, gallery_names = encode_gallery(model, sar_head, cfg, device)

    df = pd.read_csv(cfg["test_metadata"])
    query_to_sar = {str(row["id"]): os.path.basename(row["sar_path"]) for _, row in df.iterrows()}
    gallery_name_to_idx = {name: idx for idx, name in enumerate(gallery_names)}
    gt = [gallery_name_to_idx.get(query_to_sar.get(qid, ""), -1) for qid in query_ids]

    raw = retrieval_metrics(query_raw, gallery_raw, gt)
    projected = retrieval_metrics(query_proj, gallery_proj, gt)
    result = {
        "queries": len(test_ds),
        "gallery_size": len(gallery_names),
        "raw": raw,
        "projected": projected,
        "inference_latency_ms_per_query": query_latency_ms,
    }
    result["raw"].pop("ranks", None)
    result["projected"].pop("ranks", None)

    out_path = Path(cfg["results_dir"]) / "evaluation_v4_against_all.txt"
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("=== V4 REMOTECLIP LORA EVALUATION ON COMBINED GALLERY ===\n")
        handle.write(f"Total Query Images: {len(test_ds)}\n")
        handle.write(f"Total Gallery Images: {len(gallery_names)}\n")
        handle.write(f"Inference latency: {query_latency_ms:.2f} ms/query\n\n")
        handle.write("--- RAW REMOTECLIP+LORA EMBEDDINGS (No projection) ---\n")
        handle.write(f"Recall@1 / Top-1  : {raw['top1']:.2f}%\n")
        handle.write(f"Recall@5 / Top-5  : {raw['top5']:.2f}%\n")
        handle.write(f"Recall@10 / Top-10: {raw['top10']:.2f}%\n\n")
        handle.write("--- PROJECTED REMOTECLIP+LORA EMBEDDINGS (V4 Projection Head) ---\n")
        handle.write(f"Recall@1 / Top-1  : {projected['top1']:.2f}%\n")
        handle.write(f"Recall@5 / Top-5  : {projected['top5']:.2f}%\n")
        handle.write(f"Recall@10 / Top-10: {projected['top10']:.2f}%\n")

    with open(Path(cfg["results_dir"]) / "metrics_combined_gallery_v4.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(out_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    evaluate()
