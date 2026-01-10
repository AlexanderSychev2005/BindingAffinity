import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from dataset import BindingDataset
from model import BindingAffinityModel
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train_epoch(epoch, model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc=f"Training epoch: {epoch}"):
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch, batch.protein_seq)
        loss = criterion(out.squeeze(), batch.y.squeeze())

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(epoch, model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating epoch: {epoch}"):
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch, batch.protein_seq)
            loss = criterion(out.squeeze(), batch.y.squeeze())
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    # Load dataset
    dataframe = pd.read_csv('pdbbind_refined_dataset.csv')
    dataframe.dropna(inplace=True)
    print("Dataset loaded with {} samples".format(len(dataframe)))
    dataset = BindingDataset(dataframe)
    print("Dataset transformed with {} samples".format(len(dataset)))

    if len(dataset) == 0:
        print("Dataset is empty")
        return


    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    num_features = train_dataset[0].x.shape[1]
    print("Number of node features:", num_features)

    model = BindingAffinityModel(num_node_features=num_features, hidden_channels_gnn=128).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = nn.MSELoss()

    num_epochs = 20
    print(f"Starting training on {DEVICE}")
    for epoch in range(num_epochs):
        train_loss = train_epoch(epoch, model, train_loader, optimizer, criterion)
        test_loss = evaluate(epoch, model, test_loader, criterion)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    torch.save(model.state_dict(), './model.pth')


if __name__ == "__main__":
    main()