import os

import pandas as pd
import torch
import torch.nn as nn
from dataset_3d import ProteinLigand3DDataset, collate_fn
from model_gnn import GNNGNNModel
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


def train():
    # 1. Hyperparameters
    batch_size = 16  # Reduced for ESM
    lr = 5e-5
    epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    csv_path = os.path.join(base_dir, "research", "pdbbind_refined_dataset.csv")
    pdb_path = os.path.join(base_dir, "PDBbind_v2020_refined", "refined-set")

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    dataset = ProteinLigand3DDataset(df, pdb_path, device=device)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # 3. Model
    # Get input dims from a valid sample
    print("Detecting input dimensions...")
    l_in, p_in = None, None
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample is not None:
            l_in = sample["ligand_x"].shape[1]
            p_in = sample["protein_x"].shape[1]
            break

    if l_in is None:
        raise ValueError(
            "Could not find a valid sample in the dataset to detect dimensions."
        )

    print(f"Detected dimensions: Ligand={l_in}, Protein={p_in}")
    model = GNNGNNModel(ligand_in_dim=l_in, protein_in_dim=p_in, hidden_dim=256).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 4. Training Loop
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            if batch is None:
                continue
            l_batch, p_batch, y = batch
            l_batch, p_batch, y = l_batch.to(device), p_batch.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(l_batch, p_batch)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                l_batch, p_batch, y = batch
                l_batch, p_batch, y = (
                    l_batch.to(device),
                    p_batch.to(device),
                    y.to(device),
                )
                pred = model(l_batch, p_batch)
                loss = criterion(pred, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(), "research/gnn_gnn_experiment/best_model_3d_esm.pth"
            )
            print("Model saved!")


if __name__ == "__main__":
    train()
