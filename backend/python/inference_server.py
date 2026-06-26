import time
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel, logging as transformers_logging
from PIL import Image
import faiss
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from typing import Optional

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

# Global variables loaded once during startup
processor = None
model = None
opt_proj = None
sar_proj = None
image_names = []
gallery_embeddings = None
index = None

# Active query cache variables kept resident in Python RAM
current_query_embedding = None
current_query_name = None

@app.on_event("startup")
def startup_event():
    global processor, model, opt_proj, sar_proj, image_names, gallery_embeddings, index
    
    print("\nStarting search inference server...")
    
    # 1. Load DINOv2
    t_model_start = time.time()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
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
        import io
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB").resize((224, 224))
            query_name = "uploaded_file"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    elif image_path:
        path_obj = Path(image_path)
        if not path_obj.exists():
            raise HTTPException(status_code=404, detail=f"Image not found at path: {image_path}")
        try:
            image = Image.open(path_obj).convert("RGB").resize((224, 224))
            query_name = path_obj.name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image at path: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="No query image provided (either 'image_path' or uploaded 'file' is required)")
    t_pre_end = time.time()

    # 2. Feature extraction (DINOv2)
    t_feat_start = time.time()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
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
            import io
            try:
                image = Image.open(io.BytesIO(image_data)).convert("RGB").resize((224, 224))
                query_name = "uploaded_file"
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        elif image_path:
            path_obj = Path(image_path)
            if not path_obj.exists():
                raise HTTPException(status_code=404, detail=f"Image not found at path: {image_path}")
            try:
                image = Image.open(path_obj).convert("RGB").resize((224, 224))
                query_name = path_obj.name
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image at path: {str(e)}")
        t_pre_end = time.time()

        # 2. Feature extraction (DINOv2)
        t_feat_start = time.time()
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        t_feat_end = time.time()
        
        # Update cache
        current_query_embedding = query_embedding
        current_query_name = query_name

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
    distances, indices = index.search(query_norm, top_k)
    
    results = [
        {
            "filename": image_names[idx],
            "score": float(dist)
        }
        for dist, idx in zip(distances[0], indices[0])
    ]
    t_faiss_end = time.time()
    
    t_req_end = time.time()
    
    # Request timings log
    print(f"\n[Inference Request] Timing Breakdown (Cached={use_cache}):")
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
        "gallery": str(GALLERY_DIR.relative_to(DATASET_ROOT)),
        "results": results,
        "timings": {
            "image_preprocessing": 0.0 if use_cache else round(t_pre_end - t_pre_start, 4),
            "compute_query_embedding": 0.0 if use_cache else round(t_feat_end - t_feat_start, 4),
            "projection_head": round(t_proj_end - t_proj_start, 4),
            "search_gallery": round(t_faiss_end - t_faiss_start, 4),
            "total_python": round(t_req_end - t_req_start, 4)
        }
    }
