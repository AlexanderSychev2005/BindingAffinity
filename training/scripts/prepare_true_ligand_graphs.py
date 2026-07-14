import os
import sys
from tqdm import tqdm
import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from dataset import get_atom_features

def process_true_ligand(pdb_id, data_dir):
    folder = os.path.join(data_dir, pdb_id)
    sdf_path = os.path.join(folder, f"{pdb_id}_ligand.sdf")
    mol2_path = os.path.join(folder, f"{pdb_id}_ligand.mol2")
    out_path = os.path.join(folder, "ligand_3d.pt")
    
    if os.path.exists(out_path):
        return True

    mol = None
    if os.path.exists(sdf_path):
        supplier = Chem.SDMolSupplier(sdf_path, sanitize=False)
        for m in supplier:
            if m is not None and m.GetNumConformers() > 0:
                mol = m
                break
                
    if mol is None and os.path.exists(mol2_path):
        mol = Chem.MolFromMol2File(mol2_path, sanitize=False)
        
    if mol is None or mol.GetNumConformers() == 0:
        return False

    # Get features
    try:
        atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    except Exception:
        return False
        
    x = torch.tensor(np.array(atom_features), dtype=torch.float)

    # Get positions
    conf = mol.GetConformer()
    positions = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])
    pos = torch.tensor(positions, dtype=torch.float)

    # Get edges
    edge_indexes = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indexes.append((i, j))
        edge_indexes.append((j, i))
        
    if len(edge_indexes) > 0:
        edge_index = torch.tensor(edge_indexes, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Create Data object directly
    data = Data(x=x, edge_index=edge_index, pos=pos)
    torch.save(data, out_path)
    return True

def main():
    data_dir = "general-set"
    pdb_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    success = 0
    print(f"Processing TRUE ligand graphs for {len(pdb_ids)} complexes...")
    
    for pdb_id in tqdm(pdb_ids):
        if process_true_ligand(pdb_id, data_dir):
            success += 1
            
    print(f"Successfully processed {success}/{len(pdb_ids)} ligands.")

if __name__ == "__main__":
    main()
