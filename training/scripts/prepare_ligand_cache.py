import os
import sys
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

# Add src to path to import get_atom_features
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from dataset import get_atom_features


import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="general-set")
    parser.add_argument("--csv-file", type=str, default="pdbbind_general_dataset.csv")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    csv_file = args.csv_file

    if not os.path.exists(csv_file):
        print(f"Dataset CSV {csv_file} not found!")
        return

    df = pd.read_csv(csv_file, dtype={"pdb_id": str})
    df.dropna(inplace=True)

    print(f"Found {len(df)} entries. Generating 3D ligand graphs...")

    success_count = 0
    fail_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        pdb_id = row["pdb_id"]
        if isinstance(pdb_id, str) and "E+" in pdb_id:
            try:
                parts = pdb_id.split("E+")
                pdb_id = str(int(float(parts[0]))) + "e" + parts[1]
            except:
                pass
        # Get SMILES
        smiles = None
        if "smiles" in row:
            smiles = row["smiles"]
            
        if pd.isna(smiles) or smiles is None:
            sdf_path = os.path.join(args.data_dir, pdb_id, f"{pdb_id}_ligand.sdf")
            mol2_path = os.path.join(args.data_dir, pdb_id, f"{pdb_id}_ligand.mol2")
            mol = None
            if os.path.exists(sdf_path):
                supplier = Chem.SDMolSupplier(sdf_path, sanitize=False)
                for m in supplier:
                    if m is not None:
                        mol = m
                        break
            if mol is None and os.path.exists(mol2_path):
                mol = Chem.MolFromMol2File(mol2_path, sanitize=False)
                
            if mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol)
                except:
                    pass
                    
        if pd.isna(smiles) or smiles is None:
            fail_count += 1
            continue

        target_path = os.path.join(data_dir, pdb_id, "ligand_3d.pt")

        # Skip if already cached
        if os.path.exists(target_path):
            success_count += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fail_count += 1
            continue

        mol = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if res != 0:
            fail_count += 1
            continue

        conf = mol.GetConformer()
        pos_ligand = torch.tensor(conf.GetPositions(), dtype=torch.float)

        atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
        x_ligand = torch.tensor(np.array(atom_features), dtype=torch.float)

        edge_indexes = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indexes.append((i, j))
            edge_indexes.append((j, i))

        if len(edge_indexes) == 0:
            edge_index_ligand = torch.empty((2, 0), dtype=torch.long)
            edge_attr_ligand = torch.empty((0, 1), dtype=torch.float)
        else:
            edge_index_ligand = (
                torch.tensor(edge_indexes, dtype=torch.long).t().contiguous()
            )
            edge_attr_ligand = torch.ones(
                (edge_index_ligand.size(1), 1), dtype=torch.float
            )

        data_ligand = Data(
            x=x_ligand,
            pos=pos_ligand,
            edge_index=edge_index_ligand,
            edge_attr=edge_attr_ligand,
        )
        torch.save(data_ligand, target_path)
        success_count += 1

    print(f"Finished! Successfully cached: {success_count} | Failed: {fail_count}")


if __name__ == "__main__":
    main()
