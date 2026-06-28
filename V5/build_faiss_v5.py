from pathlib import Path

import faiss
import numpy as np

from v5_common import build_arg_parser, ensure_dirs, load_config
from evaluate_v5_against_all import evaluate


def main():
    parser = build_arg_parser("Build a separate V5 FAISS index from projected LoRA gallery embeddings")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.no_lora:
        cfg["use_lora"] = False
    ensure_dirs(cfg)

    gallery_path = Path(cfg["embedding_dir"]) / "combined_gallery_v5.npz"
    if not gallery_path.exists():
        print("V5 gallery embeddings not found; running full-gallery evaluation to create them first.")
        evaluate()

    if not gallery_path.exists():
        print("Error: Gallery embeddings could not be generated.")
        return

    data = np.load(gallery_path, allow_pickle=True)
    proj = data["proj"].astype("float32")
    names = data["names"]
    
    if len(proj) == 0:
        print("Error: Gallery projection dataset is empty. Cannot build index.")
        return

    # Normalize vectors for Inner Product search (equivalent to Cosine Similarity)
    proj = proj / np.linalg.norm(proj, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(proj.shape[1])
    index.add(proj)
    index_path = Path(cfg["embedding_dir"]) / "dinov2_lora_projected_gallery.index"
    names_path = Path(cfg["embedding_dir"]) / "dinov2_lora_gallery_names.npy"
    faiss.write_index(index, str(index_path))
    np.save(names_path, names)
    print(f"Saved V5 FAISS index to: {index_path}")
    print(f"Saved V5 gallery names to: {names_path}")


if __name__ == "__main__":
    main()
