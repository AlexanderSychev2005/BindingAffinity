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
import os


BATCH_SIZE = 16
LR = 0.00064
WEIGHT_DECAY = 7.06e-6
EPOCS = 100
DROPOUT = 0.325
GAT_HEADS = 2
HIDDEN_CHANNELS = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = f"runs/experiment_scheduler{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TOP_K = 3
SAVES_DIR = LOG_DIR + "/models"


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
        writer.add_scalar("Loss/Train_Step", current_loss, global_step)

        loop.set_postfix(loss=loss.item())

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
    writer.add_scalar("Loss/Test", avg_loss, epoch)
    return avg_loss


def main():
    gen = set_seed(42)
    writer = SummaryWriter(LOG_DIR)

    if not os.path.exists(SAVES_DIR):
        os.makedirs(SAVES_DIR)
    print(f"Logging to {LOG_DIR}...")
    print(f"Model saves to {SAVES_DIR}...")
    # Load dataset
    dataframe = pd.read_csv("pdbbind_refined_dataset.csv")
    dataframe.dropna(inplace=True)
    print("Dataset loaded with {} samples".format(len(dataframe)))
    dataset = BindingDataset(dataframe, max_seq_length=1200)
    print("Dataset transformed with {} samples".format(len(dataset)))

    if len(dataset) == 0:
        print("Dataset is empty")
        return

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=gen
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    num_features = train_dataset[0].x.shape[1]
    print("Number of node features:", num_features)

    model = BindingAffinityModel(
        num_node_features=num_features,
        hidden_channels=HIDDEN_CHANNELS,
        gat_heads=GAT_HEADS,
        dropout=DROPOUT,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # factor of 0.5 means reducing lr to half when triggered
    # patience of 5 means wait for 5 epochs before reducing lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    top_models = []

    print(f"Starting training on {DEVICE}")
    for epoch in range(1, EPOCS + 1):
        train_loss = train_epoch(
            epoch, model, train_loader, optimizer, criterion, writer
        )
        test_loss = evaluate(epoch, model, test_loader, criterion, writer)

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(test_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        if new_lr != old_lr:
            print(
                f"\nEpoch {epoch}: Scheduler reduced LR from {old_lr:.6f} to {new_lr:.6f}!"
            )
        print(
            f"Epoch {epoch:02d} | LR: {new_lr:.6f} | Train: {train_loss:.4f} | Test: {test_loss:.4f}",
            end="",
        )

        filename = f"{SAVES_DIR}/model_ep{epoch:03d}_mse{test_loss:.4f}.pth"

        torch.save(model.state_dict(), filename)
        top_models.append({"loss": test_loss, "path": filename, "epoch": epoch})

        top_models.sort(key=lambda x: x["loss"])

        if len(top_models) > TOP_K:
            worst_model = top_models.pop()
            os.remove(worst_model["path"])

        if any(m["epoch"] == epoch for m in top_models):
            rank = [m["epoch"] for m in top_models].index(epoch) + 1
            print(f"-- Model saved (Rank: {rank})")
        else:
            print("")

    writer.close()
    print("Training finished.")
    print("Top models saved:")
    for i, m in enumerate(top_models):
        print(f"{i + 1}. {m['path']} (MSE: {m['loss']:.4f})")


if __name__ == "__main__":
    main()
