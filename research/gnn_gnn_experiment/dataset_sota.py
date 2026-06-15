import numpy as np
import torch
import pandas as pd
from rdkit import Chem, rdBase
from torch_geometric.data import Data
from torch.utils.data import Dataset
import os
from Bio.PDB import PDBParser
import esm
from scipy.spatial.transform import Rotation

rdBase.DisableLog("rdApp.*")

class SOTABindingDataset(Dataset):
    def __init__(self, dataframe, pdb_root, device='cpu', dist_cutoff=10.0, use_pocket=True, augment=True, esm_model_name="esm2_t33_650M_UR50D"):
        self.data = dataframe
        self.pdb_root = pdb_root
        self.dist_cutoff = dist_cutoff
        self.use_pocket = use_pocket
        self.augment = augment
        self.device = device
        
        # Load Larger ESM model for Colab
        print(f"Loading {esm_model_name}...")
        if esm_model_name == "esm2_t33_650M_UR50D":
            self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.repr_layer = 33
        elif esm_model_name == "esm2_t12_35M_UR50D":
            self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            self.repr_layer = 12
        else: # Default back to 8M for safety if not specified
            self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.repr_layer = 6
            
        self.esm_model = self.esm_model.to(device)
        self.esm_model.eval()
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()

    def __len__(self):
        return len(self.data)

    def get_atom_features(self, atom):
        symbols = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "Unknown"]
        def one_hot(x, allowable_set):
            if x not in allowable_set: x = allowable_set[-1]
            return [x == s for s in allowable_set]

        features = one_hot(atom.GetSymbol(), symbols)
        features += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        features += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        features += [atom.GetIsAromatic(), atom.IsInRing()]
        return np.array(features, dtype=np.float32)

    def get_bond_features(self, bond):
        bt = bond.GetBondType()
        features = [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing()
        ]
        return np.array(features, dtype=np.float32)

    def parse_protein(self, pdb_id):
        # Use pocket file if available and requested
        suffix = "pocket" if self.use_pocket else "protein"
        pdb_path = os.path.join(self.pdb_root, pdb_id, f"{pdb_id}_{suffix}.pdb")
        
        if not os.path.exists(pdb_path):
            # Fallback to protein if pocket is missing
            pdb_path = os.path.join(self.pdb_root, pdb_id, f"{pdb_id}_protein.pdb")
            if not os.path.exists(pdb_path): return None, None

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, pdb_path)
        
        ca_coords = []
        sequence = ""
        res_codes = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E', 'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'}
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id("CA") and residue.get_resname() in res_codes:
                        ca_coords.append(residue["CA"].get_coord())
                        sequence += res_codes[residue.get_resname()]
        
        return np.array(ca_coords), sequence

    def augment_coords(self, coords):
        if self.augment:
            # Random Rotation
            rotation = Rotation.from_euler('xyz', np.random.uniform(0, 360, 3), degrees=True)
            coords = rotation.apply(coords)
            # Small Noise
            coords += np.random.normal(0, 0.05, coords.shape)
        return coords

    @torch.no_grad()
    def get_esm_embeddings(self, sequence):
        _, _, batch_tokens = self.esm_batch_converter([("protein", sequence)])
        batch_tokens = batch_tokens.to(self.device)
        results = self.esm_model(batch_tokens, repr_layers=[self.repr_layer])
        return results["representations"][self.repr_layer][0, 1 : len(sequence) + 1].cpu()

    def build_protein_graph(self, coords):
        dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)
        adj = dist_matrix < self.dist_cutoff
        np.fill_diagonal(adj, False)
        edge_index = np.argwhere(adj).T
        
        # Edge features: Distances
        dists = dist_matrix[adj]
        # Normalize distance (simple RBF-like or inverse)
        edge_attr = np.exp(-dists / self.dist_cutoff).astype(np.float32).reshape(-1, 1)
        
        return torch.tensor(edge_index, dtype=torch.long), torch.tensor(edge_attr, dtype=torch.float)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pdb_id = row['pdb_id']
        
        # 1. Ligand
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is None: return None
        
        l_x = torch.tensor(np.array([self.get_atom_features(a) for a in mol.GetAtoms()]), dtype=torch.float)
        l_edges, l_attr = [], []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            feat = self.get_bond_features(b)
            l_edges.extend([(i, j), (j, i)])
            l_attr.extend([feat, feat])
        
        l_edge_index = torch.tensor(l_edges, dtype=torch.long).t().contiguous()
        l_edge_attr = torch.tensor(np.array(l_attr), dtype=torch.float)

        # 2. Protein
        coords, sequence = self.parse_protein(pdb_id)
        if coords is None or len(sequence) < 5: return None
        
        # Data Augmentation (for 3D robustness)
        coords = self.augment_coords(coords)
        
        p_x = self.get_esm_embeddings(sequence)
        p_edge_index, p_edge_attr = self.build_protein_graph(coords)

        return {
            'ligand': Data(x=l_x, edge_index=l_edge_index, edge_attr=l_edge_attr),
            'protein': Data(x=p_x, edge_index=p_edge_index, edge_attr=p_edge_attr),
            'y': torch.tensor([float(row['affinity'])], dtype=torch.float)
        }

def collate_fn_sota(batch):
    from torch_geometric.data import Batch
    batch = [x for x in batch if x is not None]
    if not batch: return None
    
    l_batch = Batch.from_data_list([i['ligand'] for i in batch])
    p_batch = Batch.from_data_list([i['protein'] for i in batch])
    y = torch.stack([i['y'] for i in batch])
    
    return l_batch, p_batch, y
