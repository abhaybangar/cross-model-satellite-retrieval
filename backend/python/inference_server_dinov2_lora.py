import sys
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "V5") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "V5"))

from ben_preprocess import preprocess_optical
from V5.v5_common import encode_image, load_config, load_projection_heads, load_dinov2_v5

app = FastAPI(title="V5 DINOv2 LoRA Inference Server")

cfg = None
model = None
opt_proj = None
index = None
image_names = []


@app.on_event("startup")
def startup_event():
    global cfg, model, opt_proj, index, image_names
    cfg = load_config("V5/config_v5.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.perf_counter()
    model = load_dinov2_v5(cfg, device, load_adapters=cfg.get("use_lora", True))
    opt_proj, _ = load_projection_heads(cfg, device)
    model.eval()
    t1 = time.perf_counter()

    index_path = Path(cfg["embedding_dir"]) / "dinov2_lora_projected_gallery.index"
    names_path = Path(cfg["embedding_dir"]) / "dinov2_lora_gallery_names.npy"
    if not index_path.exists() or not names_path.exists():
        raise RuntimeError("V5 FAISS index is missing. Make sure to download or train the model, then build index.")
    index = faiss.read_index(str(index_path))
    image_names = list(np.load(names_path, allow_pickle=True))
    t2 = time.perf_counter()

    print(f"[V5 Startup] Base DINOv2 + LoRA + projection loaded in {(t1 - t0):.3f}s")
    print(f"[V5 Startup] FAISS index loaded in {(t2 - t1):.3f}s ({len(image_names)} images)")


@app.get("/health")
def health():
    return {"status": "ok", "system": "V5 DINOv2 LoRA", "gallery_size": len(image_names)}


@app.post("/search")
async def search(request: Request):
    content_type = request.headers.get("content-type", "")
    top_k = 5
    image_path = None
    image_data = None

    if "application/json" in content_type:
        body = await request.json()
        image_path = body.get("image_path")
        top_k = int(body.get("top_k", 5))
    elif "multipart/form-data" in content_type:
        form = await request.form()
        image_path = form.get("image_path")
        top_k = int(form.get("top_k", 5))
        upload = form.get("file")
        if upload and hasattr(upload, "file"):
            image_data = await upload.read()
    else:
        raise HTTPException(status_code=400, detail="Unsupported content type")

    t0 = time.perf_counter()
    tmp_path = None
    try:
        if image_data:
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(image_data)
                tmp_path = tmp.name
            opt_arr = preprocess_optical(tmp_path)
            query_name = getattr(upload, "filename", "uploaded_file")
        elif image_path:
            path = Path(image_path)
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
            opt_arr = preprocess_optical(str(path))
            query_name = path.name
        else:
            raise HTTPException(status_code=400, detail="Provide image_path or uploaded file.")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
    t1 = time.perf_counter()

    device = next(model.parameters()).device
    with torch.no_grad():
        image = torch.tensor(opt_arr, dtype=torch.float32).unsqueeze(0).to(device)
        embedding = encode_image(model, image)
        query = opt_proj(embedding).squeeze().cpu().numpy().astype("float32")
    t2 = time.perf_counter()

    query = query / np.linalg.norm(query)
    scores, inds = index.search(query.reshape(1, -1), top_k)
    t3 = time.perf_counter()

    return {
        "query": query_name,
        "system": "V5 DINOv2 LoRA",
        "results": [
            {"filename": str(image_names[idx]), "score": float(score)}
            for score, idx in zip(scores[0], inds[0])
        ],
        "timings": {
            "image_preprocessing": round(t1 - t0, 4),
            "compute_query_embedding": round(t2 - t1, 4),
            "search_gallery": round(t3 - t2, 4),
            "total_python": round(t3 - t0, 4),
        },
    }
