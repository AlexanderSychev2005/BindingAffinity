import random

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from dataset import BindingDataset
from model import BindingAffinityModel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LR = 0.0005
EPOCS = 30
LOG_DIR = f"runs/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    return torch.Generator().manual_seed(seed)


def train_epoch(epoch, model, loader, optimizer, criterion, writer):
    model.train()
    total_loss = 0

    loop = tqdm(loader, desc=f"Training epoch: {epoch}", leave=False)
    for i, batch in enumerate(loop):
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch, batch.protein_seq)
        loss = criterion(out.squeeze(), batch.y.squeeze())

        loss.backward()
        optimizer.step()
        current_loss = loss.item()
        total_loss += current_loss

        global_step = (epoch - 1) * len(loader) + i
        writer.add_scalar('Loss/Train_Step', current_loss, global_step)

        loop.set_postfix(loss = loss.item())

    avg_loss = total_loss / len(loader)
    return avg_loss

def evaluate(epoch, model, loader, criterion, writer):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating epoch: {epoch}", leave=False):
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch, batch.protein_seq)
            loss = criterion(out.squeeze(), batch.y.squeeze())
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    writer.add_scalar('Loss/Test', avg_loss, epoch)
    return avg_loss

def main():
    gen = set_seed(42)
    writer = SummaryWriter(LOG_DIR)
    print(f"Logging to {LOG_DIR}...")
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
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=gen)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    num_features = train_dataset[0].x.shape[1]
    print("Number of node features:", num_features)

    model = BindingAffinityModel(num_node_features=num_features, hidden_channels_gnn=128).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_test_loss = float('inf')

    print(f"Starting training on {DEVICE}")
    for epoch in range(1, EPOCS):
        train_loss = train_epoch(epoch, model, train_loader, optimizer, criterion, writer)
        test_loss = evaluate(epoch, model, test_loader, criterion, writer)


        print(f'Epoch {epoch:02d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), f'best_model_gat.pth')
            print(f'Best model saved with Test Loss MSE: {best_test_loss:.4f}')

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()