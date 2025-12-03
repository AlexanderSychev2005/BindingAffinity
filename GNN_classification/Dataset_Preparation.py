import torch
import pandas as pd

from rdkit import Chem
from torch_geometric.data import Data
from torch.utils.data import Dataset


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
        atom_features = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float)

        # Edges
        edge_indexes = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indexes.append((i, j))
            edge_indexes.append((j, i))

        # t - transpose, [num_of_edges, 2] -> [2, num_of_edges]
        # contiguous - take the virtually transposed tensor and make its physical copy and lay bytes sequentially
        if not edge_indexes:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
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

    train_dataset = SmilesDataset(train_dataset)
    test_dataset = SmilesDataset(test_dataset)

    print(len(train_dataset))
    print(len(test_dataset))


