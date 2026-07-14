import os
import torch
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.simplefilter('ignore', PDBConstructionWarning)

AA_MAP = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4, 
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9, 
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14, 
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19
}

def process_pocket(pdb_id, data_dir):
    folder = os.path.join(data_dir, pdb_id)
    pocket_path = os.path.join(folder, f"{pdb_id}_pocket.pdb")
    out_path = os.path.join(folder, "pocket_graph.pt")
    
    if os.path.exists(out_path):
        return True
        
    if not os.path.exists(pocket_path):
        return False
        
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, pocket_path)
    except Exception:
        return False
        
    coords = []
    features = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname().strip()
                if res_name not in AA_MAP:
                    continue
                    
                if 'CA' in residue:
                    ca_atom = residue['CA']
                    coords.append(ca_atom.get_coord())
                    features.append(AA_MAP[res_name])
                    
    if len(coords) == 0:
        return False
        
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    
    # 1-hot encode features
    feat_tensor = torch.zeros(len(features), 20, dtype=torch.float32)
    for i, f in enumerate(features):
        feat_tensor[i, f] = 1.0
        
    # Save as dictionary
    torch.save({"x": feat_tensor, "pos": coords_tensor}, out_path)
    return True

def main():
    data_dir = "general-set"
    pdb_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    success = 0
    print(f"Processing 3D pocket graphs for {len(pdb_ids)} complexes...")
    
    for pdb_id in tqdm(pdb_ids):
        if process_pocket(pdb_id, data_dir):
            success += 1
            
    print(f"Successfully processed {success}/{len(pdb_ids)} pockets.")

if __name__ == "__main__":
    main()
