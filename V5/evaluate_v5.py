import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from v5_common import (
    PairDataset,
    build_arg_parser,
    encode_image,
    ensure_dirs,
    load_config,
    load_projection_heads,
    load_dinov2_v5,
    retrieval_metrics,
)


def evaluate():
    parser = build_arg_parser("Evaluate V5 on test2 restricted 100-image gallery")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.no_lora:
        cfg["use_lora"] = False
    ensure_dirs(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2_v5(cfg, device, load_adapters=cfg.get("use_lora", True))
    opt_head, sar_head = load_projection_heads(cfg, device)
    model.eval()

    ds = PairDataset(cfg["test_metadata"])
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)

    opt_raw, sar_raw, opt_proj, sar_proj = [], [], [], []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            opt = batch["opt"].to(device)
            sar = batch["sar"].to(device)
            opt_e = encode_image(model, opt)
            sar_e = encode_image(model, sar)
            opt_raw.append(opt_e.cpu().numpy())
            sar_raw.append(sar_e.cpu().numpy())
            opt_proj.append(opt_head(opt_e).cpu().numpy())
            sar_proj.append(sar_head(sar_e).cpu().numpy())
    latency_ms = (time.perf_counter() - start) * 1000.0 / max(len(ds), 1)

    opt_raw = np.vstack(opt_raw).astype("float32")
    sar_raw = np.vstack(sar_raw).astype("float32")
    opt_proj = np.vstack(opt_proj).astype("float32")
    sar_proj = np.vstack(sar_proj).astype("float32")

    emb_path = Path(cfg["embedding_dir"]) / "test2_v5_embeddings.npz"
    np.savez_compressed(emb_path, opt=opt_raw, sar=sar_raw, opt_proj=opt_proj, sar_proj=sar_proj)

    raw = retrieval_metrics(opt_raw, sar_raw)
    projected = retrieval_metrics(opt_proj, sar_proj)
    result = {
        "gallery_size": len(ds),
        "raw": raw,
        "projected": projected,
        "inference_latency_ms_per_pair": latency_ms,
    }
    result["raw"].pop("ranks", None)
    result["projected"].pop("ranks", None)

    out_path = Path(cfg["results_dir"]) / "evaluation_v5.txt"
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("=== V5 DINOV2 LORA EVALUATION ON TEST2 PAIRS ===\n")
        handle.write(f"Total Queries evaluated: {len(ds)}\n")
        handle.write(f"Inference latency: {latency_ms:.2f} ms/pair\n\n")
        handle.write("--- RAW DINOV2+LORA EMBEDDINGS (No projection) ---\n")
        handle.write(f"Recall@1 / Top-1  : {raw['top1']:.2f}%\n")
        handle.write(f"Recall@5 / Top-5  : {raw['top5']:.2f}%\n")
        handle.write(f"Recall@10 / Top-10: {raw['top10']:.2f}%\n\n")
        handle.write("--- PROJECTED DINOV2+LORA EMBEDDINGS (V5 Projection Head) ---\n")
        handle.write(f"Recall@1 / Top-1  : {projected['top1']:.2f}%\n")
        handle.write(f"Recall@5 / Top-5  : {projected['top5']:.2f}%\n")
        handle.write(f"Recall@10 / Top-10: {projected['top10']:.2f}%\n")

    with open(Path(cfg["results_dir"]) / "metrics_test2_v5.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(out_path.read_text(encoding="utf-8"))
    print(f"Saved embeddings to: {emb_path}")


if __name__ == "__main__":
    evaluate()
