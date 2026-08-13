"""Control experiment: our GIGNStyleModel + our training loop, but fed
GIGN's own preprocessed graphs (from their Zenodo dataset) instead of our
RDKit/LP-PDBBind pipeline. If this reproduces ~their published numbers,
the earlier gap was in our data, not our code.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from models.gign import GIGNStyleModel


class PrebuiltGraphDataset(Dataset):
    def __init__(self, split_dir):
        self.paths = sorted(glob.glob(os.path.join(split_dir, "*", "*.pyg")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return torch.load(self.paths[idx], weights_only=False)

    @staticmethod
    def collate(batch):
        return Batch.from_data_list(batch)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        preds.append(model(batch).cpu().numpy())
        labels.append(batch.y.cpu().numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
    r = float(np.corrcoef(preds, labels)[0, 1])
    model.train()
    return rmse, r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to the unzipped GIGN_data folder")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--early-stop-patience", type=int, default=100)
    parser.add_argument("--lr-patience", type=int, default=20)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PrebuiltGraphDataset(os.path.join(args.data_dir, "train"))
    valid_ds = PrebuiltGraphDataset(os.path.join(args.data_dir, "valid"))
    test_ds = PrebuiltGraphDataset(os.path.join(args.data_dir, "test2016"))
    print(f"train={len(train_ds)} valid={len(valid_ds)} test2016={len(test_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=PrebuiltGraphDataset.collate)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=PrebuiltGraphDataset.collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=PrebuiltGraphDataset.collate)

    node_dim = train_ds[0].x.shape[1]
    model = GIGNStyleModel(node_dim=node_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience)
    loss_fn = torch.nn.MSELoss()

    best_val_rmse = float("inf")
    epochs_without_improvement = 0
    best_ckpt_path = os.path.join(args.run_dir, "best.pt")

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch.y)
            loss.backward()
            optimizer.step()

        val_rmse, val_r = evaluate(model, valid_loader, device)
        scheduler.step(val_rmse)
        print(f"epoch {epoch}: val_rmse={val_rmse:.4f} val_pearson={val_r:.4f}", flush=True)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    model.load_state_dict(torch.load(best_ckpt_path, weights_only=True))
    test_rmse, test_r = evaluate(model, test_loader, device)
    print(f"test2016 (GIGN's own data, never trained on): rmse={test_rmse:.4f} pearson={test_r:.4f}", flush=True)


if __name__ == "__main__":
    main()
