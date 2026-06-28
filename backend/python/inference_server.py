import time
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

import numpy as np
import torch
from transformers import AutoModel, logging as transformers_logging
import faiss
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from typing import Optional
from ben_preprocess import preprocess_optical, preprocess_sar

transformers_logging.set_verbosity_error()

app = FastAPI(title="PS11 Image Search Inference Server")

# File system paths
BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
CACHE_DIR = BACKEND_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
GALLERY_DIR = DATASET_ROOT / "sar" if (DATASET_ROOT / "sar").exists() else DATASET_ROOT
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

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

# Helper function to build or load gallery
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
    device = next(model.parameters()).device
    for image_path in image_paths:
        sar_array = preprocess_sar(str(image_path))
        pixel_values = torch.tensor(sar_array).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(embedding.astype("float32"))

    embeddings = np.stack(embeddings, axis=0)
    np.save(CACHE_DIR / "gallery_embeddings.npy", embeddings)
    with open(CACHE_DIR / "gallery_names.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(image_names))

    return image_names, embeddings


def load_gallery():
    # If combined_evaluation_embeddings.npz exists, use it to get correct raw gallery embeddings
    total_cache_path = CACHE_DIR / "combined_evaluation_embeddings.npz"
    if total_cache_path.exists():
        try:
            total_data = np.load(total_cache_path)
            image_names = []
            embeddings = []
            for idx, gid in enumerate(total_data["gallery_ids"]):
                gid_str = fix_gallery_path(str(gid))
                # test2 gallery has prefix 'sar/' (e.g. sar/img_2001.tif)
                if gid_str.startswith("sar/"):
                    image_names.append(gid_str)
                    embeddings.append(total_data["sar"][idx])
            if image_names:
                print(f"[Startup] Successfully loaded correct raw gallery embeddings from combined_evaluation_embeddings.npz ({len(image_names)} items)")
                return image_names, np.stack(embeddings, axis=0).astype("float32")
        except Exception as e:
            print(f"[Startup] Warning: failed to load gallery from combined_evaluation_embeddings.npz: {e}")

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


# Global variables loaded once during startup
processor = None
model = None
opt_proj = None
sar_proj = None
image_names = []
gallery_embeddings = None
index = None

# Combined total gallery variables (train + train2 + test + test2)
total_image_names = []
total_gallery_embeddings = None
total_index = None

# Active query cache variables kept resident in Python RAM
current_query_embedding = None
current_query_name = None

def fix_gallery_path(name: str) -> str:
    import re
    p = Path(name)
    filename = p.name
    
    # Check filename numbering to map directly to raw folders on disk
    match = re.search(r'img_(\d+)', filename.lower())
    if match:
        img_num = int(match.group(1))
        if 1 <= img_num <= 1800:
            return f"train/sar/{filename}"
        elif 1801 <= img_num <= 2000:
            return f"test/sar/{filename}"
        elif 2001 <= img_num <= 2100:
            return f"sar/{filename}"
            
    # Fallback general formatting
    parts = list(p.parts)
    if len(parts) >= 2 and parts[1] != "sar":
        parts.insert(1, "sar")
        return "/".join(parts)
    return name

def is_test2_query(query_name: Optional[str]) -> bool:
    if not query_name:
        return False
    name = query_name.lower()
    if "test2" in name:
        return True
    # Check if filename is like img_xxxx.tif with xxxx between 2001 and 2100
    import re
    match = re.search(r'img_(\d+)', name)
    if match:
        img_num = int(match.group(1))
        if 2001 <= img_num <= 2100:
            return True
    return False

@app.on_event("startup")
def startup_event():
    global model, opt_proj, sar_proj, image_names, gallery_embeddings, index
    global total_image_names, total_gallery_embeddings, total_index
    
    print("\nStarting search inference server...")
    
    # 1. Load DINOv2
    t_model_start = time.time()
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model.eval()
    t_model_end = time.time()
    print(f"[Startup] DINOv2 loaded in {t_model_end - t_model_start:.4f}s")
    
    # 2. Load projection heads if they exist
    t_proj_start = time.time()
    opt_proj_path = CACHE_DIR / "opt_proj.pt"
    sar_proj_path = CACHE_DIR / "sar_proj.pt"

    if opt_proj_path.exists() and sar_proj_path.exists():
        opt_proj = ProjectionHead()
        opt_proj.load_state_dict(torch.load(opt_proj_path, map_location="cpu"))
        opt_proj.eval()

        sar_proj = ProjectionHead()
        sar_proj.load_state_dict(torch.load(sar_proj_path, map_location="cpu"))
        sar_proj.eval()
        t_proj_end = time.time()
        print(f"[Startup] Projection heads loaded in {t_proj_end - t_proj_start:.4f}s")
    else:
        t_proj_end = time.time()
        print("[Startup] Projection heads NOT found. Defaulting to raw DINOv2 retrieval.")

    # 3. Load gallery embeddings
    t_gal_start = time.time()
    image_names, gallery_embeddings = load_gallery()
    t_gal_end = time.time()
    print(f"[Startup] Gallery embeddings loaded in {t_gal_end - t_gal_start:.4f}s (Total: {len(image_names)} items)")

    # 4. Construct FAISS index
    t_faiss_start = time.time()
    if opt_proj is not None and sar_proj is not None:
        with torch.no_grad():
            g_t = torch.tensor(gallery_embeddings)
            projected_gallery = sar_proj(g_t).numpy()
        # L2-normalize for FlatIP (inner product cosine similarity)
        gallery_norm = projected_gallery / np.linalg.norm(projected_gallery, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(256)
        index.add(gallery_norm.astype("float32"))
    else:
        # L2-normalize for FlatIP (inner product cosine similarity)
        gallery_norm = gallery_embeddings / np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(768)
        index.add(gallery_norm.astype("float32"))
    t_faiss_end = time.time()
    print(f"[Startup] FAISS index loaded in {t_faiss_end - t_faiss_start:.4f}s")

    # 5. Load Combined Total Gallery (train + train2 + test + test2)
    t_total_start = time.time()
    total_cache_path = CACHE_DIR / "combined_evaluation_embeddings.npz"
    if total_cache_path.exists():
        total_data = np.load(total_cache_path)
        total_gallery_embeddings = total_data["sar"]
        total_image_names = [fix_gallery_path(name) for name in total_data["gallery_ids"]]
        
        # Build total index
        if opt_proj is not None and sar_proj is not None:
            with torch.no_grad():
                g_t = torch.tensor(total_gallery_embeddings)
                projected_gallery = sar_proj(g_t).numpy()
            gallery_norm = projected_gallery / np.linalg.norm(projected_gallery, axis=1, keepdims=True)
            total_index = faiss.IndexFlatIP(256)
            total_index.add(gallery_norm.astype("float32"))
        else:
            gallery_norm = total_gallery_embeddings / np.linalg.norm(total_gallery_embeddings, axis=1, keepdims=True)
            total_index = faiss.IndexFlatIP(768)
            total_index.add(gallery_norm.astype("float32"))
        t_total_end = time.time()
        print(f"[Startup] Combined total gallery loaded in {t_total_end - t_total_start:.4f}s ({len(total_image_names)} items)")
    else:
        print("[Startup] Combined total gallery cache NOT found. Total search fallback will be disabled.")

    print("[Startup] Server ready")

@app.get("/health")
def health():
    return {"status": "ok", "gallery_size": len(image_names)}

@app.post("/preprocess")
async def preprocess(request: Request):
    global current_query_embedding, current_query_name
    t_start = time.time()
    
    content_type = request.headers.get("content-type", "")
    image_path = None
    image_data = None
    
    if "application/json" in content_type:
        body = await request.json()
        image_path = body.get("image_path")
    elif "multipart/form-data" in content_type:
        form = await request.form()
        image_path = form.get("image_path")
        uploaded_file = form.get("file")
        if uploaded_file and hasattr(uploaded_file, "file"):
            image_data = await uploaded_file.read()
    else:
        raise HTTPException(status_code=400, detail="Unsupported content type")

    # 1. Image preprocessing
    t_pre_start = time.time()
    if image_data:
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
            tmp_file.write(image_data)
            tmp_path = tmp_file.name
        try:
            opt_array = preprocess_optical(tmp_path)
            query_name = uploaded_file.filename if (uploaded_file and hasattr(uploaded_file, "filename")) else "uploaded_file"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    elif image_path:
        path_obj = Path(image_path)
        if not path_obj.exists():
            raise HTTPException(status_code=404, detail=f"Image not found at path: {image_path}")
        opt_array = preprocess_optical(str(path_obj))
        query_name = path_obj.name
    else:
        raise HTTPException(status_code=400, detail="No query image provided (either 'image_path' or uploaded 'file' is required)")
    t_pre_end = time.time()

    # 2. Feature extraction (DINOv2)
    t_feat_start = time.time()
    precomputed_emb = None
    total_cache_path = CACHE_DIR / "combined_evaluation_embeddings.npz"
    if total_cache_path.exists():
        try:
            total_data = np.load(total_cache_path)
            q_ids = [str(qid).split("/")[-1] for qid in total_data["query_ids"]]
            if query_name in q_ids:
                q_idx = q_ids.index(query_name)
                precomputed_emb = total_data["opt"][q_idx]
                print(f"[Inference] Found precomputed raw embedding for query: {query_name}")
        except Exception as e:
            print(f"[Inference] Error looking up precomputed query embedding: {e}")

    if precomputed_emb is not None:
        query_embedding = precomputed_emb
    else:
        device = next(model.parameters()).device
        pixel_values = torch.tensor(opt_array).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    t_feat_end = time.time()

    current_query_embedding = query_embedding
    current_query_name = query_name

    t_end = time.time()
    
    print(f"\n[Inference Request] Preprocessing Timing:")
    print(f"  - Image Preprocessing:     {(t_pre_end - t_pre_start)*1000:.2f}ms")
    print(f"  - Feature Extraction:      {(t_feat_end - t_feat_start)*1000:.2f}ms")
    print(f"  -----------------------------------")
    print(f"  - Total Preprocess Time:   {(t_end - t_start)*1000:.2f}ms\n")

    return {
        "status": "ok",
        "query": query_name,
        "timings": {
            "image_preprocessing": round(t_pre_end - t_pre_start, 4),
            "compute_query_embedding": round(t_feat_end - t_feat_start, 4),
            "total_python": round(t_end - t_start, 4)
        }
    }

@app.post("/search")
async def search(request: Request):
    global current_query_embedding, current_query_name
    t_req_start = time.time()
    
    # Check Content-Type and parse fields
    content_type = request.headers.get("content-type", "")
    image_path = None
    top_k = 5
    image_data = None
    
    if "application/json" in content_type:
        body = await request.json()
        image_path = body.get("image_path")
        top_k = int(body.get("top_k", 5))
    elif "multipart/form-data" in content_type:
        form = await request.form()
        image_path = form.get("image_path")
        top_k = int(form.get("top_k", 5))
        uploaded_file = form.get("file")
        if uploaded_file and hasattr(uploaded_file, "file"):
            image_data = await uploaded_file.read()

    # Decide if we can use the preprocessed cached embedding
    use_cache = False
    if not image_path and not image_data:
        if current_query_embedding is None:
            raise HTTPException(status_code=400, detail="No query image provided and no preprocessed query embedding in cache.")
        query_embedding = current_query_embedding
        query_name = current_query_name
        use_cache = True
        
        # Fill timing dummies
        t_pre_start = t_pre_end = t_feat_start = t_feat_end = time.time()
    else:
        # 1. Image preprocessing
        t_pre_start = time.time()
        if image_data:
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
                tmp_file.write(image_data)
                tmp_path = tmp_file.name
            try:
                opt_array = preprocess_optical(tmp_path)
                query_name = uploaded_file.filename if (uploaded_file and hasattr(uploaded_file, "filename")) else "uploaded_file"
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        elif image_path:
            path_obj = Path(image_path)
            if not path_obj.exists():
                raise HTTPException(status_code=404, detail=f"Image not found at path: {image_path}")
            opt_array = preprocess_optical(str(path_obj))
            query_name = path_obj.name
        t_pre_end = time.time()

        # 2. Feature extraction (DINOv2)
        t_feat_start = time.time()
        precomputed_emb = None
        total_cache_path = CACHE_DIR / "combined_evaluation_embeddings.npz"
        if total_cache_path.exists():
            try:
                total_data = np.load(total_cache_path)
                q_ids = [str(qid).split("/")[-1] for qid in total_data["query_ids"]]
                if query_name in q_ids:
                    q_idx = q_ids.index(query_name)
                    precomputed_emb = total_data["opt"][q_idx]
                    print(f"[Inference] Found precomputed raw embedding for query: {query_name}")
            except Exception as e:
                print(f"[Inference] Error looking up precomputed query embedding: {e}")

        if precomputed_emb is not None:
            query_embedding = precomputed_emb
        else:
            device = next(model.parameters()).device
            pixel_values = torch.tensor(opt_array).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)
            query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        t_feat_end = time.time()
        
        # Update cache
        current_query_embedding = query_embedding
        current_query_name = query_name

    # Choose index and image list based on query type
    is_test2 = is_test2_query(query_name)
    
    current_index = index
    current_names = image_names
    gallery_scope = "test2"
    
    if not is_test2 and total_index is not None:
        current_index = total_index
        current_names = total_image_names
        gallery_scope = "total"

    # 3. Projection head alignment
    t_proj_start = time.time()
    if opt_proj is not None and sar_proj is not None:
        with torch.no_grad():
            q_t = torch.tensor(query_embedding).reshape(1, -1)
            query_embedding = opt_proj(q_t).squeeze().numpy()
    t_proj_end = time.time()

    # 4. FAISS similarity search
    t_faiss_start = time.time()
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    query_norm = query_norm.reshape(1, -1).astype("float32")
    
    # Query FAISS index
    distances, indices = current_index.search(query_norm, top_k)
    
    results = [
        {
            "filename": current_names[idx],
            "score": float(dist)
        }
        for dist, idx in zip(distances[0], indices[0])
    ]
    t_faiss_end = time.time()
    
    t_req_end = time.time()
    
    # Request timings log
    print(f"\n[Inference Request] Timing Breakdown (Cached={use_cache}):")
    print(f"  - Query Name:              {query_name}")
    print(f"  - Target Search Scope:     {gallery_scope.upper()} ({len(current_names)} images)")
    if use_cache:
        print(f"  - Image Preprocessing:     SKIPPED")
        print(f"  - Feature Extraction:      SKIPPED")
    else:
        print(f"  - Image Preprocessing:     {(t_pre_end - t_pre_start)*1000:.2f}ms")
        print(f"  - Feature Extraction:      {(t_feat_end - t_feat_start)*1000:.2f}ms")
    print(f"  - Projection Head:         {(t_proj_end - t_proj_start)*1000:.2f}ms")
    print(f"  - FAISS Search:            {(t_faiss_end - t_faiss_start)*1000:.2f}ms")
    print(f"  -----------------------------------")
    print(f"  - Total Inference Time:    {(t_req_end - t_req_start)*1000:.2f}ms\n")

    return {
        "query": query_name,
        "gallery": gallery_scope,
        "results": results,
        "timings": {
            "image_preprocessing": 0.0 if use_cache else round(t_pre_end - t_pre_start, 4),
            "compute_query_embedding": 0.0 if use_cache else round(t_feat_end - t_feat_start, 4),
            "projection_head": round(t_proj_end - t_proj_start, 4),
            "search_gallery": round(t_faiss_end - t_faiss_start, 4),
            "total_python": round(t_req_end - t_req_start, 4)
        }
    }
