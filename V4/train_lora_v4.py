import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from v4_common import (
    PairDataset,
    ProjectionHead,
    build_arg_parser,
    clip_loss,
    encode_image,
    ensure_dirs,
    load_config,
    load_remoteclip_v4,
    print_trainable_parameters,
    retrieval_metrics,
    save_lora_adapter,
    save_projection_heads,
)


def evaluate_split(model, opt_head, sar_head, loader, device):
    model.eval()
    opt_head.eval()
    sar_head.eval()
    opt_feats, sar_feats = [], []
    with torch.no_grad():
        for batch in loader:
            opt = batch["opt"].to(device)
            sar = batch["sar"].to(device)
            opt_feats.append(opt_head(encode_image(model, opt)).cpu().numpy())
            sar_feats.append(sar_head(encode_image(model, sar)).cpu().numpy())
    return retrieval_metrics(np.vstack(opt_feats), np.vstack(sar_feats))


def train():
    parser = build_arg_parser("Train V4 RemoteCLIP LoRA + Projection Head")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.no_lora:
        cfg["use_lora"] = False
    ensure_dirs(cfg)

    seed = int(cfg["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== V4 RemoteCLIP LoRA Training ===")
    print(f"Device: {device}")
    print(f"LoRA enabled: {cfg.get('use_lora', True)}")
    if cfg.get("use_lora", True):
        print(f"LoRA config: {json.dumps(cfg['lora'])}")

    df_len = len(PairDataset(cfg["train_metadata"]))
    indices = np.arange(df_len)
    np.random.shuffle(indices)
    split = int(float(cfg["train_split"]) * df_len)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_ds = PairDataset(cfg["train_metadata"], train_idx)
    val_ds = PairDataset(cfg["train_metadata"], val_idx)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)

    model = load_remoteclip_v4(cfg, device, load_adapters=False)
    opt_head = ProjectionHead(output_dim=int(cfg["projection_dim"])).to(device)
    sar_head = ProjectionHead(output_dim=int(cfg["projection_dim"])).to(device)

    print_trainable_parameters(model, opt_head, sar_head)

    optimizer = torch.optim.AdamW(
        [p for p in list(model.parameters()) + list(opt_head.parameters()) + list(sar_head.parameters()) if p.requires_grad],
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = None

    best_val = -1.0
    best_opt_state = None
    best_sar_state = None
    log_rows = []

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        opt_head.train()
        sar_head.train()
        losses = []

        for batch in train_loader:
            opt = batch["opt"].to(device)
            sar = batch["sar"].to(device)

            optimizer.zero_grad()
            opt_proj = opt_head(encode_image(model, opt))
            sar_proj = sar_head(encode_image(model, sar))
            loss = clip_loss(opt_proj, sar_proj, temp=float(cfg["temperature"]))
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(loss.item())

        metrics = evaluate_split(model, opt_head, sar_head, val_loader, device)
        avg_loss = float(np.mean(losses))
        if metrics["recall@1"] > best_val:
            best_val = metrics["recall@1"]
            best_opt_state = {k: v.detach().cpu().clone() for k, v in opt_head.state_dict().items()}
            best_sar_state = {k: v.detach().cpu().clone() for k, v in sar_head.state_dict().items()}
            save_lora_adapter(cfg, model)
            save_projection_heads(cfg, opt_head, sar_head)

        row = {
            "epoch": epoch,
            "loss": avg_loss,
            "val_recall@1": metrics["recall@1"],
            "val_recall@5": metrics["recall@5"],
            "val_recall@10": metrics["recall@10"],
            "val_top1": metrics["top1"],
            "val_top5": metrics["top5"],
            "val_top10": metrics["top10"],
            "best_val_recall@1": best_val,
        }
        log_rows.append(row)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{cfg['epochs']} | Loss: {avg_loss:.4f} | "
                f"Val R@1: {metrics['recall@1']:.2f}% | R@5: {metrics['recall@5']:.2f}% | "
                f"R@10: {metrics['recall@10']:.2f}% | Best R@1: {best_val:.2f}%"
            )

    if best_opt_state is not None:
        opt_head.load_state_dict(best_opt_state)
        sar_head.load_state_dict(best_sar_state)
        save_projection_heads(cfg, opt_head, sar_head)

    log_path = Path(cfg["results_dir"]) / "train_v4_log.json"
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(log_rows, handle, indent=2)

    print("\nTraining completed.")
    print(f"Best validation Recall@1: {best_val:.2f}%")
    print(f"LoRA adapters saved to: {Path(cfg['model_dir']) / 'lora_adapter'}")
    print(f"Projection heads saved to: {cfg['model_dir']}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    train()
