import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase
from torch.utils.data import Dataset
from torch_geometric.data import Data

rdBase.DisableLog("rdApp.*")

# --- Ligand Utilities (from original dataset.py) ---


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_atom_features(atom):
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
    degrees_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    numhs_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    implicit_valences_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    formal_charge_list = [-2, -1, 0, 1, 2]
    chirality_list = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]

    return np.array(
        one_of_k_encoding(atom.GetSymbol(), symbols_list)
        + one_of_k_encoding(atom.GetDegree(), degrees_list)
        + one_of_k_encoding(atom.GetTotalNumHs(), numhs_list)
        + one_of_k_encoding(atom.GetImplicitValence(), implicit_valences_list)
        + one_of_k_encoding(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
                "other",
            ],
        )
        + [atom.GetIsAromatic()]
        + one_of_k_encoding(atom.GetFormalCharge(), formal_charge_list)
        + one_of_k_encoding(atom.GetChiralTag(), chirality_list)
        + [atom.IsInRing()]
    )


# --- Protein Utilities ---


def get_residue_features(char):
    """
    Basic features for an amino acid residue.
    One-hot encoding + some physical-chemical categories could be added here.
    """
    res_list = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
        "X",
    ]
    return one_of_k_encoding(char, res_list)


def build_protein_graph(sequence, max_len=1000):
    """
    Builds a graph where nodes are residues and edges are sequence proximity.
    """
    seq = sequence[:max_len]
    nodes = [get_residue_features(aa) for aa in seq]
    x = torch.tensor(np.array(nodes), dtype=torch.float)

    edge_index = []
    for i in range(len(seq) - 1):
        # Sequential edges (i -> i+1 and i+1 -> i)
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])

        # Optional: k-NN edges or local window edges
        if i < len(seq) - 2:
            edge_index.append([i, i + 2])
            edge_index.append([i + 2, i])

    if len(edge_index) == 0:  # Handle very short sequences
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return x, edge_index


class ProteinLigandGraphDataset(Dataset):
    def __init__(self, dataframe, max_prot_len=1000, use_esm=False):
        self.data = dataframe
        self.max_prot_len = max_prot_len
        self.use_esm = use_esm
        # In the future, we can load ESM embeddings here if they are pre-calculated

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row["smiles"]
        sequence = row["sequence"]
        affinity = row["affinity"]

        # 1. Ligand Graph
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        atom_feats = [get_atom_features(atom) for atom in mol.GetAtoms()]
        ligand_x = torch.tensor(np.array(atom_feats), dtype=torch.float)

        ligand_edges = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            ligand_edges.append((i, j))
            ligand_edges.append((j, i))
        ligand_edge_index = (
            torch.tensor(ligand_edges, dtype=torch.long).t().contiguous()
        )

        # 2. Protein Graph
        prot_x, prot_edge_index = build_protein_graph(sequence, self.max_prot_len)

        # 3. Target
        y = torch.tensor([float(affinity)], dtype=torch.float)

        return {
            "ligand_x": ligand_x,
            "ligand_edge_index": ligand_edge_index,
            "protein_x": prot_x,
            "protein_edge_index": prot_edge_index,
            "y": y,
            "pdb_id": row["pdb_id"],
        }


def collate_fn(batch):
    """
    Custom collate function for torch_geometric.data.Batch doesn't easily handle
    two different graphs in one Data object. We'll use a custom structure or Batch.from_data_list carefully.
    """
    from torch_geometric.data import Batch

    batch = [x for x in batch if x is not None]

    ligand_list = []
    protein_list = []
    y_list = []

    for item in batch:
        ligand_list.append(
            Data(x=item["ligand_x"], edge_index=item["ligand_edge_index"])
        )
        protein_list.append(
            Data(x=item["protein_x"], edge_index=item["protein_edge_index"])
        )
        y_list.append(item["y"])

    return (
        Batch.from_data_list(ligand_list),
        Batch.from_data_list(protein_list),
        torch.stack(y_list),
    )


if __name__ == "__main__":
    df = pd.read_csv("research/pdbbind_refined_dataset.csv").head(100)
    dataset = ProteinLigandGraphDataset(df)
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Ligand nodes: {sample['ligand_x'].shape}")
    print(f"Protein nodes: {sample['protein_x'].shape}")
