import numpy as np
import torch
import pandas as pd
from rdkit import Chem, rdBase
from torch_geometric.data import Data
from torch.utils.data import Dataset, random_split
import os

rdBase.DisableLog("rdApp.*")


def one_of_k_encoding(x, allowable_set):
    # last position - unknown
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
        # Type of atom (Symbol)
        one_of_k_encoding(atom.GetSymbol(), symbols_list)
        +
        # Number of neighbours (Degree)
        one_of_k_encoding(atom.GetDegree(), degrees_list)
        +
        # Number of hydrogen atoms (Implicit Hs) - bond donors
        one_of_k_encoding(atom.GetTotalNumHs(), numhs_list)
        +
        # Valence - chemical potential
        one_of_k_encoding(atom.GetImplicitValence(), implicit_valences_list)
        +
        # Hybridization - so important for 3d structure, sp2 - Trigonal planar, sp3 - Tetrahedral
        one_of_k_encoding(
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
        +
        # Aromaticity (Boolean)
        [atom.GetIsAromatic()]
        +
        # Formal Charge
        one_of_k_encoding(atom.GetFormalCharge(), formal_charge_list)
        +
        # Chirality (Geometry)
        one_of_k_encoding(atom.GetChiralTag(), chirality_list)
        +
        # Is in ring (Boolean)
        [atom.IsInRing()]
    )


def get_protein_features(char):
    prot_vocab = {
        "A": 1,
        "R": 2,
        "N": 3,
        "D": 4,
        "C": 5,
        "Q": 6,
        "E": 7,
        "G": 8,
        "H": 9,
        "I": 10,
        "L": 11,
        "K": 12,
        "M": 13,
        "F": 14,
        "P": 15,
        "S": 16,
        "T": 17,
        "W": 18,
        "Y": 19,
        "V": 20,
        "X": 21,
        "Z": 21,
        "B": 21,
        "PAD": 0,
        "UNK": 21,
    }
    return prot_vocab.get(char, prot_vocab["UNK"])


def parse_pdb_ca(pdb_file):
    coords = []
    aa_types = []

    # 3-letter to 1-letter mapping
    aa_map = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
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
    }

    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM  "):
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    res_name = line[17:20].strip()
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coords.append([x, y, z])
                    aa_types.append(get_protein_features(aa_map.get(res_name, "X")))

    if not coords:
        return None, None

    return torch.tensor(aa_types, dtype=torch.long), torch.tensor(
        coords, dtype=torch.float
    )


class BindingDataset(Dataset):
    def __init__(self, dataframe, max_seq_length=1000):
        self.data = dataframe
        self.max_seq_length = (
            max_seq_length  # Define a maximum sequence length for padding/truncation
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row["smiles"]
        sequence = row["sequence"]
        affinity = row["affinity"]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Ligand (Graph)
        # Nodes
        atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(np.array(atom_features), dtype=torch.float)

        # Edges
        edge_indexes = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indexes.append((i, j))
            edge_indexes.append((j, i))

        # t - transpose, [num_of_edges, 2] -> [2, num_of_edges]
        # contiguous - take the virtually transposed tensor and make its physical copy and lay bytes sequentially

        edge_index = torch.tensor(edge_indexes, dtype=torch.long).t().contiguous()

        # Protein (Sequence, tensor of integers)
        tokens = [get_protein_features(char) for char in sequence]
        if len(tokens) > self.max_seq_length:
            tokens = tokens[: self.max_seq_length]
        else:
            tokens.extend(
                [get_protein_features("PAD")] * (self.max_seq_length - len(tokens))
            )
        protein_tensor = torch.tensor(tokens, dtype=torch.long)

        # Affinity
        y = torch.tensor([affinity], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, protein_seq=protein_tensor, y=y)


if __name__ == "__main__":
    dataset = pd.read_csv("pdbbind_refined_dataset.csv")
    dataset = BindingDataset(dataset)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(len(train_dataset))
    print(len(test_dataset))


class BindingDataset3D(Dataset):
    def __init__(
        self, dataframe, data_dir="refined-set", max_nodes=1000, dist_threshold=8.0
    ):
        self.data = dataframe
        self.data_dir = data_dir
        self.max_nodes = max_nodes
        self.dist_threshold = dist_threshold

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pdb_id = row["pdb_id"]
        if isinstance(pdb_id, str) and "E+" in pdb_id:
            try:
                parts = pdb_id.split("E+")
                pdb_id = str(int(float(parts[0]))) + "e" + parts[1]
            except:
                pass
        smiles = row["smiles"]
        affinity = row["affinity"]

        # Ligand Cache
        ligand_cache_path = os.path.join(self.data_dir, pdb_id, "ligand_3d.pt")
        if not os.path.exists(ligand_cache_path):
            return self.__getitem__((idx + 1) % len(self))

        data_ligand = torch.load(ligand_cache_path, weights_only=False)

        # ESM-2 Cache
        esm_path = os.path.join(self.data_dir, pdb_id, "esm2_35m.pt")
        if not os.path.exists(esm_path):
            return self.__getitem__((idx + 1) % len(self))

        esm_emb = torch.load(esm_path, map_location="cpu", weights_only=True)

        # Protein (3D Graph)
        pdb_path = os.path.join(self.data_dir, pdb_id, f"{pdb_id}_protein.pdb")
        if not os.path.exists(pdb_path):
            return self.__getitem__((idx + 1) % len(self))

        x_protein, pos_protein = parse_pdb_ca(pdb_path)
        if x_protein is None or len(x_protein) != len(esm_emb):
            return self.__getitem__((idx + 1) % len(self))

        if len(esm_emb) > self.max_nodes:
            esm_emb = esm_emb[: self.max_nodes]
            pos_protein = pos_protein[: self.max_nodes]

        x_prot_features = esm_emb

        # Edges based on distance threshold
        dist_matrix = torch.cdist(pos_protein, pos_protein)
        edge_index_protein = (
            (dist_matrix < self.dist_threshold).nonzero(as_tuple=False).t().contiguous()
        )
        # Remove self-loops
        mask = edge_index_protein[0] != edge_index_protein[1]
        edge_index_protein = edge_index_protein[:, mask]

        # Edge attributes: log distance in primary sequence
        seq_dist = (
            torch.abs(edge_index_protein[0] - edge_index_protein[1])
            .float()
            .unsqueeze(1)
        )
        edge_attr_protein = torch.log1p(seq_dist)

        y = torch.tensor([affinity], dtype=torch.float)

        # We use separate Data objects for ligand and protein
        # so PyG's DataLoader can batch them properly.
        data_protein = Data(
            x=x_prot_features,
            pos=pos_protein,
            edge_index=edge_index_protein,
            edge_attr=edge_attr_protein,
        )

        return data_ligand, data_protein, y
