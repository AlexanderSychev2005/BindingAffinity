import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from rdkit import Chem

from torch_geometric.loader import DataLoader

from Dataset_Preparation import SmilesDataset
from model import GNNClassifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)

        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def test(model, loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch)

            pred = out.argmax(dim=1)

            correct += (pred == batch.y).sum().item()

    acc = correct / len(loader.dataset)
    return acc


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

    num_node_features = train_dataset[0].x.shape[1]
    num_classes = 2

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Node features: {num_node_features}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    model = GNNClassifier(input_dim=1, output_dim=2, hidden_channels=16).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    EPOCHS = 20
    print("Start Training")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)

        train_acc = evaluate(model, train_loader)
        print(f"Epoch: {epoch}, Loss: {train_loss}, Train Accuracy: {train_acc}")
