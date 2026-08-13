"""Cloud-only: end-to-end fine-tuning of ESM-2 + ChemBERTa (both unfrozen)
with the same fusion head as train_seqlig.py. Needs real GPU memory - this
is not something the local 4GB card runs.

Note: `torch.autocast(device_type="cuda", ...)` and `torch.cuda.*` below are
correct on ROCm too - PyTorch's ROCm build keeps the "cuda" namespace as a
drop-in alias for HIP, that's not a CUDA-only path.
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from dataset_seqlig_raw import RawSeqLigDataset, make_collate
from models.seqlig_finetune import SeqLigFinetuneModel
from train_utils import TopKCheckpoints, save_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for protein_tokens, ligand_inputs, y in loader:
        protein_tokens = protein_tokens.to(device)
        ligand_inputs = {k: v.to(device) for k, v in ligand_inputs.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(protein_tokens, ligand_inputs)
        preds.append(pred.float().cpu().numpy())
        labels.append(y.numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
    r = float(np.corrcoef(preds, labels)[0, 1])
    model.train()
    return rmse, r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "sequence_based", "data"))
    parser.add_argument("--run-dir", default=os.path.join(REPO_ROOT, "sequence_based", "runs", "seqlig_finetune"))
    parser.add_argument("--split-prefix", default="bindingdb", help="bindingdb (cold-both) or bindingdb_colddrug")
    parser.add_argument("--protein-model", default="esm2_t36_3B_UR50D",
                         choices=["esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D",
                                  "esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--micro-batch-size", type=int, default=4, help="per-step batch; tune to fit VRAM")
    parser.add_argument("--grad-accum-steps", type=int, default=64, help="effective batch = micro_batch * this")
    parser.add_argument("--max-protein-len", type=int, default=1000,
                         help="truncate residues beyond this - bounds worst-case O(L^2) attention memory per batch")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                         help="recompute ESM-2 layer activations in backward instead of storing them - big memory cut, ~30%% slower")
    parser.add_argument("--lr", type=float, default=2e-5, help="small - fine-tuning pretrained transformers, not training from scratch")
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--early-stop-patience", type=int, default=8,
                         help="in eval events (--eval-every-steps), not epochs - a full epoch over 341k rows can take hours")
    parser.add_argument("--lr-patience", type=int, default=3, help="in eval events")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-top-k", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every-steps", type=int, default=10, help="optimizer steps between train/loss log lines")
    parser.add_argument("--eval-every-steps", type=int, default=200,
                         help="optimizer steps between val runs + checkpoints - waiting for a full epoch to see any "
                              "signal isn't practical at this scale, so validation happens mid-epoch too")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.run_dir, exist_ok=True)
    save_config(args.run_dir, args)
    writer = SummaryWriter(os.path.join(args.run_dir, "tensorboard"))

    model = SeqLigFinetuneModel(
        device, args.protein_model, hidden_dim=args.hidden_dim, gradient_checkpointing=args.gradient_checkpointing
    ).to(device)
    collate = make_collate(model.protein_alphabet, model.ligand_tokenizer, max_protein_len=args.max_protein_len)

    def make_loader(split, shuffle):
        ds = RawSeqLigDataset(os.path.join(args.data_dir, f"{args.split_prefix}_{split}.csv"))
        return DataLoader(ds, batch_size=args.micro_batch_size, shuffle=shuffle, collate_fn=collate,
                           num_workers=args.num_workers, pin_memory=True)

    train_loader = make_loader("train", True)
    val_loader = make_loader("val", False)
    test_loader = make_loader("test", False)
    n_protein_params = sum(p.numel() for p in model.protein_model.parameters())
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}", flush=True)
    print(f"protein encoder: {args.protein_model} ({n_protein_params:,} params, unfrozen)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience)
    loss_fn = torch.nn.MSELoss()

    best_val_rmse = float("inf")
    evals_without_improvement = 0
    checkpoints = TopKCheckpoints(args.run_dir, k=args.keep_top_k)

    global_step = 0
    micro_step = 0
    running_loss, running_n = 0.0, 0
    stop = False

    model.train()
    for epoch in range(args.epochs):
        if stop:
            break
        for protein_tokens, ligand_inputs, y in train_loader:
            protein_tokens = protein_tokens.to(device)
            ligand_inputs = {k: v.to(device) for k, v in ligand_inputs.items()}
            y = y.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(protein_tokens, ligand_inputs)
                loss = loss_fn(pred, y) / args.grad_accum_steps
            loss.backward()
            running_loss += loss.item() * args.grad_accum_steps * y.size(0)
            running_n += y.size(0)
            micro_step += 1

            if micro_step % args.grad_accum_steps != 0:
                continue
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % args.log_every_steps == 0:
                train_loss = running_loss / max(running_n, 1)
                print(f"step {global_step} (epoch {epoch}): train_loss={train_loss:.4f}", flush=True)
                writer.add_scalar("train/loss", train_loss, global_step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
                running_loss, running_n = 0.0, 0

            if global_step % args.eval_every_steps != 0:
                continue
            val_rmse, val_r = evaluate(model, val_loader, device)
            scheduler.step(val_rmse)
            print(f"step {global_step} (epoch {epoch}): val_rmse={val_rmse:.4f} val_pearson={val_r:.4f}", flush=True)
            writer.add_scalar("val/rmse", val_rmse, global_step)
            writer.add_scalar("val/pearson", val_r, global_step)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1

            checkpoints.offer(global_step, val_rmse, model.state_dict(), torch.save)
            if evals_without_improvement >= args.early_stop_patience:
                print(f"early stopping at step {global_step}", flush=True)
                stop = True
                break

    model.load_state_dict(torch.load(checkpoints.best_path, weights_only=True))
    test_rmse, test_r = evaluate(model, test_loader, device)
    print(f"{args.split_prefix} test (never trained on): rmse={test_rmse:.4f} pearson={test_r:.4f}", flush=True)
    writer.add_scalar("test/rmse", test_rmse, 0)
    writer.add_scalar("test/pearson", test_r, 0)
    writer.close()


if __name__ == "__main__":
    main()
