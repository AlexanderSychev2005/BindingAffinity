import random
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import os
import sys
import argparse
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dataset import BindingDataset3D
from models.model_3d import BindingAffinity3DModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 3


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    return torch.Generator().manual_seed(seed)


def train_epoch(epoch, model, loader, optimizer, criterion, writer, scaler, args):
    model.train()
    total_loss = 0

    loop = tqdm(loader, desc=f"Training epoch: {epoch}", leave=False)
    for i, (batch_ligand, batch_protein, batch_y) in enumerate(loop):
        batch_ligand = batch_ligand.to(DEVICE)
        batch_protein = batch_protein.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        # Add Coordinate Noise (Data Augmentation) to prevent overfitting
        if args.noise_std > 0:
            noise_ligand = torch.randn_like(batch_ligand.pos) * args.noise_std
            batch_ligand.pos = batch_ligand.pos + noise_ligand

            noise_protein = torch.randn_like(batch_protein.pos) * args.noise_std
            batch_protein.pos = batch_protein.pos + noise_protein

        with torch.amp.autocast(device_type="cuda", enabled=args.fp16):
            out = model(
                batch_ligand.x,
                batch_ligand.pos,
                batch_ligand.edge_index,
                batch_ligand.edge_attr,
                batch_ligand.batch,
                batch_protein.x,
                batch_protein.pos,
                batch_protein.edge_index,
                batch_protein.edge_attr,
                batch_protein.batch,
            )
            loss = criterion(out.squeeze(), batch_y.squeeze())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        current_loss = loss.item()
        total_loss += current_loss

        global_step = (epoch - 1) * len(loader) + i
        writer.add_scalar("Loss/Train_Step", current_loss, global_step)

        loop.set_postfix(loss=current_loss)

    avg_loss = total_loss / len(loader)
    return avg_loss


def evaluate(epoch, model, loader, criterion, writer, args):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_ligand, batch_protein, batch_y in tqdm(
            loader, desc=f"Evaluating epoch: {epoch}", leave=False
        ):
            batch_ligand = batch_ligand.to(DEVICE)
            batch_protein = batch_protein.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            with torch.amp.autocast(device_type="cuda", enabled=args.fp16):
                out = model(
                    batch_ligand.x,
                    batch_ligand.pos,
                    batch_ligand.edge_index,
                    batch_ligand.edge_attr,
                    batch_ligand.batch,
                    batch_protein.x,
                    batch_protein.pos,
                    batch_protein.edge_index,
                    batch_protein.edge_attr,
                    batch_protein.batch,
                )
                loss = criterion(out.squeeze(), batch_y.squeeze())

            total_loss += loss.item()
            all_preds.extend(out.squeeze().cpu().numpy().tolist())
            all_targets.extend(batch_y.squeeze().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)

    # Calculate metrics
    rmse = root_mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    pearson_corr, _ = pearsonr(all_targets, all_preds)

    writer.add_scalar("Loss/Test", avg_loss, epoch)
    writer.add_scalar("Metrics/RMSE", rmse, epoch)
    writer.add_scalar("Metrics/MAE", mae, epoch)
    writer.add_scalar("Metrics/Pearson", pearson_corr, epoch)

    return avg_loss, rmse, mae, pearson_corr


def main():
    parser = argparse.ArgumentParser(
        description="Train 3D Binding Affinity Model on Colab"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/content/dataset_fast/refined-set",
        help="Path to unzipped refined-set",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="pdbbind_refined_dataset.csv",
        help="Path to csv dataset",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/content/drive/MyDrive/BindingAffinity_Runs",
        help="Directory for logs and models",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.02,
        help="Standard deviation of coordinate noise during training",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="L2 weight decay for optimizer"
    )
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden channels")
    parser.add_argument(
        "--fp16", action="store_true", help="Enable Automatic Mixed Precision (AMP)"
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Number of dataloader workers"
    )
    args = parser.parse_args()

    gen = set_seed(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"3d_egnn_{timestamp}")
    saves_dir = os.path.join(log_dir, "models")

    writer = SummaryWriter(log_dir)

    os.makedirs(saves_dir, exist_ok=True)
    print(f"Logging to {log_dir}...")
    print(f"Model saves to {saves_dir}...")

    # Load dataset
    dataframe = pd.read_csv(args.csv_file, dtype={"pdb_id": str})
    dataframe.dropna(inplace=True)

    print("Dataset loaded with {} samples".format(len(dataframe)))
    dataset = BindingDataset3D(
        dataframe, data_dir=args.data_dir, max_nodes=1000, dist_threshold=8.0
    )
    print("Dataset transformed with {} samples".format(len(dataset)))

    if len(dataset) == 0:
        print("Dataset is empty. Check --data-dir path!")
        return

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=gen
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    sample_ligand, sample_protein, _ = train_dataset[0]
    num_ligand_features = sample_ligand.x.shape[1]
    num_protein_features = 480  # ESM-2 35M embedding size

    print("Ligand features:", num_ligand_features)
    print("Protein features:", num_protein_features)

    model = BindingAffinity3DModel(
        num_ligand_features=num_ligand_features,
        num_protein_features=num_protein_features,
        hidden_channels=args.hidden,
        gat_heads=4,
        dropout=0.3,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )
    criterion = nn.MSELoss()

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    top_models = []

    print(f"Starting training on {DEVICE} (FP16 AMP: {args.fp16})")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            epoch, model, train_loader, optimizer, criterion, writer, scaler, args
        )
        test_loss, rmse, mae, pearson = evaluate(
            epoch, model, test_loader, criterion, writer, args
        )

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(test_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d} | LR: {new_lr:.6f} | Train: {train_loss:.4f} | "
            f"Test: {test_loss:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | Pearson: {pearson:.4f}"
        )

        filename = f"{saves_dir}/model_3d_ep{epoch:03d}_mse{test_loss:.4f}.pth"
        torch.save(model.state_dict(), filename)
        top_models.append({"loss": test_loss, "path": filename, "epoch": epoch})
        top_models.sort(key=lambda x: x["loss"])

        if len(top_models) > TOP_K:
            worst_model = top_models.pop()
            if os.path.exists(worst_model["path"]):
                os.remove(worst_model["path"])

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
