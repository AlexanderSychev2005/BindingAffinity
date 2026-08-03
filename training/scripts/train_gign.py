import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from dataset_pocket import NODE_DIM, PocketLigandDataset
from models.gign import GIGNStyleModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_loader(csv_name, data_dir, cache_subdir, batch_size, shuffle, data_root):
    ds = PocketLigandDataset(
        csv_path=os.path.join(data_root, csv_name),
        data_dir=data_dir,
        cache_dir=os.path.join(data_root, "cache", cache_subdir),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=PocketLigandDataset.collate)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        preds.append(pred.cpu().numpy())
        labels.append(batch.y.cpu().numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
    r = float(np.corrcoef(preds, labels)[0, 1])
    model.train()
    return rmse, r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "training", "data"))
    parser.add_argument("--general-set-dir", default=os.path.join(REPO_ROOT, "general-set"))
    parser.add_argument("--casf-coreset-dir", default=os.path.join(REPO_ROOT, "CASF-2016", "coreset"))
    parser.add_argument("--run-dir", default=os.path.join(REPO_ROOT, "runs", "gign"))
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--early-stop-patience", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.run_dir, exist_ok=True)

    train_loader = make_loader("train.csv", args.general_set_dir, "train", args.batch_size, True, args.data_root)
    val_loader = make_loader("val.csv", args.general_set_dir, "val", args.batch_size, False, args.data_root)
    test_loader = make_loader("test_casf2016.csv", args.casf_coreset_dir, "test_casf2016", args.batch_size, False, args.data_root)

    model = GIGNStyleModel(node_dim=NODE_DIM, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_val_rmse = float("inf")
    epochs_without_improvement = 0
    best_ckpt_path = os.path.join(args.run_dir, "best.pt")

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()

        val_rmse, val_r = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: val_rmse={val_rmse:.4f} val_pearson={val_r:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(best_ckpt_path, weights_only=True))
    test_rmse, test_r = evaluate(model, test_loader, device)
    print(f"CASF-2016 core (never trained on): rmse={test_rmse:.4f} pearson={test_r:.4f}")


if __name__ == "__main__":
    main()
