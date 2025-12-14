import numpy as np
import torch
import pandas as pd
from rdkit import Chem, rdBase
from torch_geometric.data import Data
from torch.utils.data import Dataset, random_split

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
    )

def get_protein_features(char):
    prot_vocab= {
            'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9,
            'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17,
            'W': 18, 'Y': 19, 'V': 20, 'X': 21, 'Z': 21, 'B': 21,
            'PAD': 0, 'UNK': 21
        }
    return prot_vocab.get(char, prot_vocab['UNK'])


class BindingDataset(Dataset):
    def __init__(self, dataframe, max_seq_length=1000):
        self.data = dataframe
        self.max_seq_length = max_seq_length  # Define a maximum sequence length for padding/truncation

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
            tokens = tokens[:self.max_seq_length]
        else:
            tokens.extend([get_protein_features("PAD")] * (self.max_seq_length - len(tokens)))
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

