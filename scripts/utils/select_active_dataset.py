import os
import sys
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATASET_DIR = os.path.join(WORKSPACE, "dataset")
CACHE_DIR = os.path.join(WORKSPACE, "backend", "cache")

def main():
    if len(sys.argv) < 2:
        print("Usage: python select_active_dataset.py [train|train2|test|test2]")
        sys.exit(1)
        
    choice = sys.argv[1].lower()
    source_dir = os.path.join(DATASET_DIR, choice)
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist.")
        sys.exit(1)
        
    opt_src = os.path.join(source_dir, "optical")
    sar_src = os.path.join(source_dir, "sar")
    
    if not os.path.exists(opt_src) or not os.path.exists(sar_src):
        print(f"Error: Could not find 'optical' or 'sar' folders inside {source_dir}.")
        sys.exit(1)
        
    # Active directories
    opt_dest = os.path.join(DATASET_DIR, "optical")
    sar_dest = os.path.join(DATASET_DIR, "sar")
    
    print(f"Switching active backend dataset to: {choice.upper()}...")
    
    # 1. Clean existing active folders
    if os.path.exists(opt_dest):
        shutil.rmtree(opt_dest)
    if os.path.exists(sar_dest):
        shutil.rmtree(sar_dest)
        
    # 2. Copy files (using shutil.copytree)
    print(f"Copying {choice}/optical -> dataset/optical...")
    shutil.copytree(opt_src, opt_dest)
    print(f"Copying {choice}/sar -> dataset/sar...")
    shutil.copytree(sar_src, sar_dest)
    
    # 3. Clear backend cache (only index files, keep model weights)
    for filename in ["gallery_embeddings.npy", "gallery_names.txt"]:
        path = os.path.join(CACHE_DIR, filename)
        if os.path.exists(path):
            os.remove(path)
        
    print(f"\nSuccessfully switched active dataset to {choice.upper()}!")
    print("Now when you search in the web app, it will search this dataset.")

if __name__ == "__main__":
    main()
