import numpy as np
import torch
import pandas as pd
from rdkit import Chem, rdBase
from torch_geometric.data import Data
from torch.utils.data import Dataset

rdBase.DisableLog('rdApp.*')


def one_of_k_encoding(x, allowable_set):
    # last position - unknown
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_atom_features(atom):
    symbols_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    degrees_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    numhs_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    implicit_valences_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return np.array(
        # Type of atom (Symbol)
        one_of_k_encoding(atom.GetSymbol(), symbols_list) +
        # Number of neighbours (Degree)
        one_of_k_encoding(atom.GetDegree(), degrees_list) +
        # Number of hydrogen atoms (Implicit Hs) - bond donors
        one_of_k_encoding(atom.GetTotalNumHs(), numhs_list) +
        # Valence - chemical potential
        one_of_k_encoding(atom.GetImplicitValence(), implicit_valences_list) +
        # Hybridization - so important for 3d structure, sp2 - Trigonal planar, sp3 - Tetrahedral
        one_of_k_encoding(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            'other']) +
        # Aromaticity (Boolean)
        [atom.GetIsAromatic()]


    )



class SmilesDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row["smiles"]
        label = row["label"]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None

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


        # Label
        y = torch.tensor([label], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)


if __name__ == "__main__":
    columns = ["smiles", "label"]
    train_dataset = pd.read_csv(
        "dataset/classification/data_train.txt", sep=" ", header=None, names=columns
    )
    test_dataset = pd.read_csv(
        "dataset/classification/data_test.txt", sep=" ", header=None, names=columns
    )
    train_dataset.to_csv("dataset/classification/data_train.csv", index=False)
    test_dataset.to_csv("dataset/classification/data_test.csv", index=False)

    train_dataset = SmilesDataset(train_dataset)
    test_dataset = SmilesDataset(test_dataset)


    print(len(train_dataset))
    print(len(test_dataset))


