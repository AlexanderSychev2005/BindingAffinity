import os
import shutil
from tqdm import tqdm

def merge_directories(src_dirs, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    
    total_moved = 0
    for src in src_dirs:
        if not os.path.exists(src):
            print(f"Warning: Source directory {src} does not exist. Skipping.")
            continue
            
        folders = [f for f in os.listdir(src) if os.path.isdir(os.path.join(src, f))]
        print(f"Moving {len(folders)} folders from {src} to {dest_dir}...")
        
        for folder in tqdm(folders):
            src_path = os.path.join(src, folder)
            dest_path = os.path.join(dest_dir, folder)
            
            if not os.path.exists(dest_path):
                # Using move is instantly fast on the same drive
                shutil.move(src_path, dest_path)
                total_moved += 1
                
    print(f"Successfully moved {total_moved} unique folders into {dest_dir}")

if __name__ == "__main__":
    sources = ["refined-set", "PDBbind_v2020_other_PL"]
    destination = "general-set"
    merge_directories(sources, destination)
