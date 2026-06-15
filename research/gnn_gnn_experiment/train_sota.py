import os

import pandas as pd
import torch
from dataset_sota import SOTABindingDataset, collate_fn_sota
from model_sota import SOTABindingModel, gaussian_nll_loss
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


def train():
    # 1. Hyperparameters
    batch_size = 4  # Small batch because ESM-650M is HUGE
    lr = 2e-5
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Paths
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    csv_path = os.path.join(base_dir, "research", "pdbbind_refined_dataset.csv")
    pdb_path = os.path.join(base_dir, "refined-set")

    df = pd.read_csv(csv_path)

    # 3. Dataset (ESM-650M for Colab)
    # If OOM in Colab, change to "esm2_t12_35M_UR50D"
    dataset = SOTABindingDataset(
        df, pdb_path, device=device, esm_model_name="esm2_t33_650M_UR50D"
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_sota
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_sota
    )

    # 4. Model Setup
    # Ligand: 23 features, 6 edge features
    # Protein: 1280 features (for ESM-650M), 1 edge feature (distance)
    model = SOTABindingModel(
        ligand_in_dim=23,
        protein_in_dim=1280,
        ligand_edge_dim=6,
        protein_edge_dim=1,
        hidden_dim=512,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Training
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
            pred, log_var = model(l_batch, p_batch)

            # Use Gaussian NLL Loss for Uncertainty
            loss = gaussian_nll_loss(pred, y, log_var)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

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
                pred, log_var = model(l_batch, p_batch)
                loss = F.mse_loss(pred, y)  # Validation using pure MSE for clarity
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch + 1}: Train NLL: {avg_train_loss:.4f}, Val MSE: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(), "research/gnn_gnn_experiment/best_model_sota.pth"
            )
            print("New best model saved!")


if __name__ == "__main__":
    train()
