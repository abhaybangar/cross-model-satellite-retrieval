import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from ben_preprocess import preprocess_optical, preprocess_sar

CLIP_DIM = 512


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=CLIP_DIM, output_dim=256, dropout_prob=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * 0.02
        return F.normalize(self.net(x), p=2, dim=-1)


def clip_loss(opt_proj, sar_proj, temp=0.07):
    logits = torch.matmul(opt_proj, sar_proj.t()) / temp
    labels = torch.arange(opt_proj.size(0), device=opt_proj.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def load_config(path):
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = WORKSPACE / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    for key in ["model_dir", "embedding_dir", "results_dir"]:
        cfg[key] = str((WORKSPACE / cfg[key]).resolve())
    cfg["remoteclip_checkpoint"] = str((WORKSPACE / cfg["remoteclip_checkpoint"]).resolve())
    cfg["train_metadata"] = str((WORKSPACE / cfg["train_metadata"]).resolve())
    cfg["test_metadata"] = str((WORKSPACE / cfg["test_metadata"]).resolve())
    return cfg


def ensure_dirs(cfg):
    for key in ["model_dir", "embedding_dir", "results_dir"]:
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)


def resolve_dataset_path(rel_path):
    full = WORKSPACE / "dataset" / str(rel_path)
    if full.exists():
        return full

    text = str(full)
    train_to_test = Path(text.replace(os.sep + "train" + os.sep, os.sep + "test" + os.sep))
    if train_to_test.exists():
        return train_to_test

    test2_optical = Path(text.replace(os.sep + "test" + os.sep + "optical" + os.sep, os.sep + "optical" + os.sep))
    if test2_optical.exists():
        return test2_optical

    test2_sar = Path(text.replace(os.sep + "test" + os.sep + "sar" + os.sep, os.sep + "sar" + os.sep))
    if test2_sar.exists():
        return test2_sar

    return full


class PairDataset(Dataset):
    def __init__(self, metadata_csv, indices=None):
        self.df = pd.read_csv(metadata_csv)
        if indices is not None:
            self.df = self.df.iloc[list(indices)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        opt_path = resolve_dataset_path(row["optical_path"])
        sar_path = resolve_dataset_path(row["sar_path"])
        opt = preprocess_optical(str(opt_path))
        sar = preprocess_sar(str(sar_path))
        return {
            "id": str(row["id"]),
            "opt": torch.tensor(opt, dtype=torch.float32),
            "sar": torch.tensor(sar, dtype=torch.float32),
            "sar_name": os.path.basename(str(row["sar_path"])),
        }


def load_remoteclip_base(cfg, device):
    import open_clip

    ckpt_path = Path(cfg["remoteclip_checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"RemoteCLIP checkpoint not found: {ckpt_path}. "
            "Place RemoteCLIP-ViT-B-32.pt there or update V4/config_v4.json."
        )
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    return model


def freeze_all(module):
    for param in module.parameters():
        param.requires_grad = False


def apply_lora_to_visual(model, cfg):
    if not cfg.get("use_lora", True):
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PEFT is required for V4 LoRA. Install it with: "
            "venv\\Scripts\\python.exe -m pip install peft"
        ) from exc

    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        bias="none",
    )
    model.visual = get_peft_model(model.visual, peft_config)
    return model


def load_remoteclip_v4(cfg, device, load_adapters=False):
    model = load_remoteclip_base(cfg, device)
    freeze_all(model)
    if load_adapters and cfg.get("use_lora", True):
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PEFT is required to load V4 LoRA adapters. Install peft first."
            ) from exc
        adapter_dir = Path(cfg["model_dir"]) / "lora_adapter"
        model.visual = PeftModel.from_pretrained(model.visual, adapter_dir)
    else:
        model = apply_lora_to_visual(model, cfg)
    return model


def encode_image(model, images):
    return F.normalize(model.encode_image(images).float(), p=2, dim=-1)


def retrieval_metrics(query_feat, gallery_feat, gt_indices=None):
    scores = np.matmul(query_feat, gallery_feat.T)
    n = scores.shape[0]
    top1 = top5 = top10 = 0
    ranks = []
    for i in range(n):
        gt = i if gt_indices is None else gt_indices[i]
        ranked = np.argsort(-scores[i])
        if gt == ranked[0]:
            top1 += 1
        if gt in ranked[:5]:
            top5 += 1
        if gt in ranked[:10]:
            top10 += 1
        rank_pos = int(np.where(ranked == gt)[0][0]) + 1 if gt >= 0 else -1
        ranks.append(rank_pos)
    return {
        "recall@1": top1 / n * 100,
        "recall@5": top5 / n * 100,
        "recall@10": top10 / n * 100,
        "top1": top1 / n * 100,
        "top5": top5 / n * 100,
        "top10": top10 / n * 100,
        "ranks": ranks,
    }


def print_trainable_parameters(model, opt_head=None, sar_head=None):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if opt_head is not None:
        trainable += sum(p.numel() for p in opt_head.parameters() if p.requires_grad)
        total += sum(p.numel() for p in opt_head.parameters())
    if sar_head is not None:
        trainable += sum(p.numel() for p in sar_head.parameters() if p.requires_grad)
        total += sum(p.numel() for p in sar_head.parameters())
    pct = trainable / max(total, 1) * 100
    print(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.4f}%)")
    return trainable, total


def save_projection_heads(cfg, opt_head, sar_head):
    model_dir = Path(cfg["model_dir"])
    torch.save(opt_head.state_dict(), model_dir / "opt_proj.pt")
    torch.save(sar_head.state_dict(), model_dir / "sar_proj.pt")


def load_projection_heads(cfg, device):
    opt_head = ProjectionHead(output_dim=int(cfg["projection_dim"])).to(device)
    sar_head = ProjectionHead(output_dim=int(cfg["projection_dim"])).to(device)
    opt_head.load_state_dict(torch.load(Path(cfg["model_dir"]) / "opt_proj.pt", map_location=device))
    sar_head.load_state_dict(torch.load(Path(cfg["model_dir"]) / "sar_proj.pt", map_location=device))
    opt_head.eval()
    sar_head.eval()
    return opt_head, sar_head


def save_lora_adapter(cfg, model):
    if cfg.get("use_lora", True):
        adapter_dir = Path(cfg["model_dir"]) / "lora_adapter"
        model.visual.save_pretrained(adapter_dir)


def build_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="V4/config_v4.json")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA and train/evaluate projection heads only.")
    return parser


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0
