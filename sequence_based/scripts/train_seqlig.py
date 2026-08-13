import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from dataset_seqlig import SeqLigDataset, load_embeddings
from embeddings import LIGAND_EMBED_DIM
from models.seqlig import SeqLigBilinearModel, SeqLigModel
from train_utils import TopKCheckpoints, save_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for protein, ligand, y in loader:
        protein, ligand = protein.to(device), ligand.to(device)
        preds.append(model(protein, ligand).cpu().numpy())
        labels.append(y.numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
    r = float(np.corrcoef(preds, labels)[0, 1])
    model.train()
    return rmse, r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "sequence_based", "data"))
    parser.add_argument("--embed-dir", default=os.path.join(REPO_ROOT, "sequence_based", "data", "embeddings"))
    parser.add_argument("--run-dir", default=os.path.join(REPO_ROOT, "sequence_based", "runs", "seqlig"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="mlp", choices=["mlp", "bilinear"])
    parser.add_argument("--split-prefix", default="bindingdb", help="bindingdb (cold-both) or bindingdb_colddrug")
    parser.add_argument("--keep-top-k", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.run_dir, exist_ok=True)
    save_config(args.run_dir, args)
    writer = SummaryWriter(os.path.join(args.run_dir, "tensorboard"))

    protein_emb, ligand_emb = load_embeddings(args.embed_dir)
    protein_embed_dim = next(iter(protein_emb.values())).shape[0]
    print(f"protein embed dim (from cache): {protein_embed_dim}", flush=True)

    def make_loader(name, shuffle):
        ds = SeqLigDataset(os.path.join(args.data_dir, name), protein_emb, ligand_emb)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, collate_fn=SeqLigDataset.collate)

    train_loader = make_loader(f"{args.split_prefix}_train.csv", True)
    val_loader = make_loader(f"{args.split_prefix}_val.csv", False)
    test_loader = make_loader(f"{args.split_prefix}_test.csv", False)
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}", flush=True)

    model_cls = SeqLigBilinearModel if args.model == "bilinear" else SeqLigModel
    model = model_cls(protein_embed_dim, LIGAND_EMBED_DIM, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience)
    loss_fn = torch.nn.MSELoss()

    best_val_rmse = float("inf")
    epochs_without_improvement = 0
    checkpoints = TopKCheckpoints(args.run_dir, k=args.keep_top_k)

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for protein, ligand, y in train_loader:
            protein, ligand, y = protein.to(device), ligand.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(protein, ligand), y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * y.size(0)
            train_n += y.size(0)

        val_rmse, val_r = evaluate(model, val_loader, device)
        scheduler.step(val_rmse)
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch}: val_rmse={val_rmse:.4f} val_pearson={val_r:.4f}", flush=True)

        writer.add_scalar("train/loss", train_loss_sum / train_n, epoch)
        writer.add_scalar("val/rmse", val_rmse, epoch)
        writer.add_scalar("val/pearson", val_r, epoch)
        writer.add_scalar("lr", lr, epoch)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoints.offer(epoch, val_rmse, model.state_dict(), torch.save)
        if epochs_without_improvement >= args.early_stop_patience:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(torch.load(checkpoints.best_path, weights_only=True))
    test_rmse, test_r = evaluate(model, test_loader, device)
    print(f"{args.split_prefix} test (never trained on): rmse={test_rmse:.4f} pearson={test_r:.4f}", flush=True)
    writer.add_scalar("test/rmse", test_rmse, 0)
    writer.add_scalar("test/pearson", test_r, 0)
    writer.close()


if __name__ == "__main__":
    main()
