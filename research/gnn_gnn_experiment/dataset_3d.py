import os

import esm
import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from rdkit import Chem, rdBase
from torch.utils.data import Dataset
from torch_geometric.data import Data

rdBase.DisableLog("rdApp.*")


class ProteinLigand3DDataset(Dataset):
    def __init__(
        self, dataframe, pdb_root, device="cpu", dist_cutoff=8.0, max_prot_len=1000
    ):
        self.data = dataframe
        self.pdb_root = pdb_root
        self.dist_cutoff = dist_cutoff
        self.max_prot_len = max_prot_len
        self.device = device

        # Load ESM model
        print("Loading ESM-2 model...")
        self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.esm_model = self.esm_model.to(device)
        self.esm_model.eval()
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()

    def __len__(self):
        return len(self.data)

    def get_atom_features(self, atom):
        symbols_list = [
            "C",
            "N",
            "O",
            "S",
            "F",
            "Si",
            "P",
            "Cl",
            "Br",
            "Mg",
            "Na",
            "Ca",
            "Fe",
            "As",
            "Al",
            "I",
            "B",
            "V",
            "K",
            "Tl",
            "Yb",
            "Sb",
            "Sn",
            "Ag",
            "Pd",
            "Co",
            "Se",
            "Ti",
            "Zn",
            "H",
            "Li",
            "Ge",
            "Cu",
            "Au",
            "Ni",
            "Cd",
            "In",
            "Mn",
            "Zr",
            "Cr",
            "Pt",
            "Hg",
            "Pb",
            "Unknown",
        ]
        hybrid_list = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ]

        def one_hot(x, allowable_set):
            if x not in allowable_set:
                x = allowable_set[-1]
            return [x == s for s in allowable_set]

        features = one_hot(atom.GetSymbol(), symbols_list)
        features += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
        features += one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
        features += [atom.GetIsAromatic(), atom.IsInRing()]
        return np.array(features, dtype=np.float32)

    def parse_pdb_structure(self, pdb_id):
        pdb_path = os.path.join(self.pdb_root, pdb_id, f"{pdb_id}_protein.pdb")
        if not os.path.exists(pdb_path):
            return None, None

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, pdb_path)

        ca_coords = []
        sequence = ""
        res_codes = {
            "ALA": "A",
            "ARG": "R",
            "ASN": "N",
            "ASP": "D",
            "CYS": "C",
            "GLN": "Q",
            "GLE": "E",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LEU": "L",
            "LYS": "K",
            "MET": "M",
            "PHE": "F",
            "PRO": "P",
            "SER": "S",
            "THR": "T",
            "TRP": "W",
            "TYR": "Y",
            "VAL": "V",
            "GLU": "E",
        }

        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id("CA") and residue.get_resname() in res_codes:
                        ca_coords.append(residue["CA"].get_coord())
                        sequence += res_codes[residue.get_resname()]

        if not ca_coords:
            return None, None
        return np.array(ca_coords), sequence

    @torch.no_grad()
    def get_esm_embeddings(self, sequence):
        sequence = sequence[: self.max_prot_len]
        batch_labels, batch_strs, batch_tokens = self.esm_batch_converter(
            [("protein", sequence)]
        )
        batch_tokens = batch_tokens.to(self.device)

        results = self.esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
        token_representations = results["representations"][6]

        # Remove CLS and EOS tokens
        return token_representations[0, 1 : len(sequence) + 1].cpu()

    def build_3d_edges(self, coords):
        dist_matrix = np.linalg.norm(
            coords[:, np.newaxis] - coords[np.newaxis, :], axis=2
        )
        adj = dist_matrix < self.dist_cutoff
        np.fill_diagonal(adj, False)
        edge_index = np.argwhere(adj).T
        return torch.tensor(edge_index, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pdb_id = row["pdb_id"]
        smiles = row["smiles"]

        # 1. Ligand Graph
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        ligand_x = torch.tensor(
            np.array([self.get_atom_features(a) for a in mol.GetAtoms()]),
            dtype=torch.float,
        )
        ligand_edges = []
        for b in mol.GetBonds():
            ligand_edges.extend(
                [
                    (b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                    (b.GetEndAtomIdx(), b.GetBeginAtomIdx()),
                ]
            )
        ligand_edge_index = (
            torch.tensor(ligand_edges, dtype=torch.long).t().contiguous()
        )

        # 2. Protein 3D Graph + ESM
        coords, sequence = self.parse_pdb_structure(pdb_id)
        if coords is None:
            return None

        coords = coords[: self.max_prot_len]
        sequence = sequence[: self.max_prot_len]

        prot_x = self.get_esm_embeddings(sequence)  # [Seq_Len, 320] for esm2_t6_8M
        prot_edge_index = self.build_3d_edges(coords)

        return {
            "ligand_x": ligand_x,
            "ligand_edge_index": ligand_edge_index,
            "protein_x": prot_x,
            "protein_edge_index": prot_edge_index,
            "y": torch.tensor([float(row["affinity"])], dtype=torch.float),
            "pdb_id": pdb_id,
        }


def collate_fn(batch):
    from torch_geometric.data import Batch

    batch = [x for x in batch if x is not None]
    if not batch:
        return None

    l_list = [Data(x=i["ligand_x"], edge_index=i["ligand_edge_index"]) for i in batch]
    p_list = [Data(x=i["protein_x"], edge_index=i["protein_edge_index"]) for i in batch]
    y = torch.stack([i["y"] for i in batch])

    return Batch.from_data_list(l_list), Batch.from_data_list(p_list), y


if __name__ == "__main__":
    # Test
    df = pd.read_csv("research/pdbbind_refined_dataset.csv").head(5)
    pdb_path = "PDBbind_v2020_refined/refined-set"
    ds = ProteinLigand3DDataset(df, pdb_path)
    sample = ds[0]
    if sample:
        print(f"Ligand X: {sample['ligand_x'].shape}")
        print(f"Protein X (ESM): {sample['protein_x'].shape}")
        print(f"Protein Edges (3D): {sample['protein_edge_index'].shape}")
