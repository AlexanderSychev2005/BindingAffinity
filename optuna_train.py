import optuna
import torch
import torch.nn as nn
import pandas as pd
import random
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from dataset import BindingDataset
from model import BindingAffinityModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_TRIALS = 20
EPOCHS_PER_TRIAL = 15

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return torch.Generator().manual_seed(seed)

dataframe = pd.read_csv('pdbbind_refined_dataset.csv')
dataframe.dropna(inplace=True)
dataset = BindingDataset(dataframe)

gen = set_seed(42)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=gen)
num_features = train_dataset[0].x.shape[1]

def train(model, loader, optimizer, criterion):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch, batch.protein_seq)
        loss = criterion(out.squeeze(), batch.y.squeeze())
        loss.backward()
        optimizer.step()


def test(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch, batch.protein_seq)
            loss = criterion(out.squeeze(), batch.y.squeeze())
            total_loss += loss.item()
    return total_loss / len(loader)


def objective(trial):
    # Architecture
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    gat_heads = trial.suggest_categorical("gat_heads", [2, 4, 8])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # Learning

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True) # Learning rate from 0.00001 to 0.01
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True) # Weight decay from 0.000001 to 0.001
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    model = BindingAffinityModel(num_node_features=num_features, hidden_channels=hidden_dim, gat_heads=gat_heads, dropout=dropout).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(EPOCHS_PER_TRIAL):
        train(model, train_loader, optimizer, criterion)
        val_loss = test(model, test_loader, criterion)

        print(f"Trial {trial.number} | Epoch {epoch + 1}/{EPOCHS_PER_TRIAL} | Val Loss: {val_loss:.4f}")

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return val_loss


if __name__ == "__main__":
    storage_name = "sqlite:///db.sqlite3"
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        storage=storage_name,
        study_name="binding_prediction_optimization",
        load_if_exists=True
    )
    print("Start hyperparameter optimization...")

    study.optimize(objective, n_trials=N_TRIALS)
    print("\n--- Optimization Finished ---")
    print("Best parameters found: ", study.best_params)
    print("Best Test MSE: ", study.best_value)

    df_results = study.trials_dataframe()
    df_results.to_csv("optuna_results.csv")
