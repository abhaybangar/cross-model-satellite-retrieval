import time
import sys
t_start = time.time()

import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel, logging as transformers_logging

t_import_end = time.time()

transformers_logging.set_verbosity_error()

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
CACHE_DIR = BACKEND_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GALLERY_DIR = DATASET_ROOT / "sar" if (DATASET_ROOT / "sar").exists() else DATASET_ROOT

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

t_model_start = time.time()
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")

class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim=768, output_dim=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.LayerNorm(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, output_dim)
        )

    def forward(self, x):
        z = self.net(x)
        return torch.nn.functional.normalize(z, p=2, dim=-1)

# Load projection heads if they exist
opt_proj = None
sar_proj = None

opt_proj_path = CACHE_DIR / "opt_proj.pt"
sar_proj_path = CACHE_DIR / "sar_proj.pt"

if opt_proj_path.exists() and sar_proj_path.exists():
    opt_proj = ProjectionHead()
    opt_proj.load_state_dict(torch.load(opt_proj_path, map_location="cpu"))
    opt_proj.eval()

    sar_proj = ProjectionHead()
    sar_proj.load_state_dict(torch.load(sar_proj_path, map_location="cpu"))
    sar_proj.eval()
t_model_end = time.time()

def build_gallery():
    image_paths = []
    image_names = []

    for image_path in sorted(GALLERY_DIR.rglob("*")):
        if image_path.suffix.lower() in IMAGE_EXTENSIONS and image_path.is_file():
            rel_path = image_path.relative_to(DATASET_ROOT).as_posix()
            image_paths.append(image_path)
            image_names.append(rel_path)

    if not image_paths:
        raise RuntimeError(f"No gallery images found in {GALLERY_DIR}")

    embeddings = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(embedding.astype("float32"))

    embeddings = np.stack(embeddings, axis=0)
    np.save(CACHE_DIR / "gallery_embeddings.npy", embeddings)
    with open(CACHE_DIR / "gallery_names.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(image_names))

    return image_names, embeddings


def load_gallery():
    names_path = CACHE_DIR / "gallery_names.txt"
    embeddings_path = CACHE_DIR / "gallery_embeddings.npy"

    if names_path.exists() and embeddings_path.exists():
        with open(names_path, "r", encoding="utf-8") as handle:
            image_names = [line.strip() for line in handle if line.strip()]
        embeddings = np.load(embeddings_path)
        if embeddings.shape[0] != len(image_names):
            return build_gallery()
        return image_names, embeddings

    return build_gallery()


def compute_embedding(image_path: Path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype("float32")
    return embedding


def search_gallery(query_embedding, image_names, gallery_embeddings, top_k=5):
    if opt_proj is not None and sar_proj is not None:
        with torch.no_grad():
            q_t = torch.tensor(query_embedding).reshape(1, -1)
            g_t = torch.tensor(gallery_embeddings)
            query_embedding = opt_proj(q_t).squeeze().numpy()
            gallery_embeddings = sar_proj(g_t).numpy()

    query_norm = query_embedding / np.linalg.norm(query_embedding)
    gallery_norm = gallery_embeddings / np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
    scores = gallery_norm.dot(query_norm)
    order = np.argsort(-scores)[:top_k]

    return [
        {
            "filename": image_names[idx],
            "score": float(scores[idx]),
        }
        for idx in order
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search a gallery using an optical query image.")
    parser.add_argument("--query", required=True, help="Path to the optical query image.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return.")
    args = parser.parse_args()

    query_path = Path(args.query)
    if not query_path.exists():
        raise FileNotFoundError(f"Query image not found: {query_path}")

    t_load_start = time.time()
    image_names, gallery_embeddings = load_gallery()
    t_load_end = time.time()

    t_embed_start = time.time()
    query_embedding = compute_embedding(query_path)
    t_embed_end = time.time()

    t_search_start = time.time()
    results = search_gallery(query_embedding, image_names, gallery_embeddings, top_k=args.top_k)
    t_search_end = time.time()

    t_end = time.time()

    timings = {
        "import_libraries": round(t_import_end - t_start, 4),
        "load_models": round(t_model_end - t_model_start, 4),
        "load_gallery": round(t_load_end - t_load_start, 4),
        "compute_query_embedding": round(t_embed_end - t_embed_start, 4),
        "search_gallery": round(t_search_end - t_search_start, 4),
        "total_python": round(t_end - t_start, 4)
    }

    # Print to stderr for terminal visibility during standalone runs
    sys.stderr.write("\n⏱️  [Python CLI] Query Timing Breakdown:\n")
    sys.stderr.write(f"  - Import Libraries:          {timings['import_libraries']:.4f}s\n")
    sys.stderr.write(f"  - Load Models:              {timings['load_models']:.4f}s\n")
    sys.stderr.write(f"  - Load Gallery Embeddings:   {timings['load_gallery']:.4f}s\n")
    sys.stderr.write(f"  - Query Feature Extraction:  {timings['compute_query_embedding']:.4f}s\n")
    sys.stderr.write(f"  - Similarity Match & Rank:   {timings['search_gallery']:.4f}s\n")
    sys.stderr.write(f"  -----------------------------------\n")
    sys.stderr.write(f"  - Total Python Execution:    {timings['total_python']:.4f}s\n\n")

    payload = {
        "query": query_path.name,
        "gallery": str(GALLERY_DIR.relative_to(DATASET_ROOT)),
        "results": results,
        "timings": timings,
    }
    print(json.dumps(payload))
