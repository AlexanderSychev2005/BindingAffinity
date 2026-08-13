"""Compute and cache ESM-2 protein embeddings and ChemBERTa ligand embeddings
for every unique sequence/SMILES across the train/val/test split CSVs, so
training the fusion head never has to run either encoder again.
"""

import argparse
import os
import sys
import tempfile

import pandas as pd
import torch


def atomic_save(obj, path):
    """torch.save(obj, path) truncates the file before writing - if the
    process dies mid-write (e.g. mid-pickle on a dict with 1M+ tensors,
    which can take a while), the file is left empty and everything already
    cached is lost, not just the newest additions. Write to a temp file in
    the same directory first, then os.replace (atomic on both POSIX and
    Windows) so the real path only ever points at a complete file.
    """
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(fd)
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from embeddings import compute_ligand_embeddings, compute_protein_embeddings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_SPLITS = [
    "bindingdb_train.csv", "bindingdb_val.csv", "bindingdb_test.csv",
    "bindingdb_colddrug_train.csv", "bindingdb_colddrug_val.csv", "bindingdb_colddrug_test.csv",
]


def top_up(path, keys, compute_fn, device, save_interval_sec):
    """Load whatever's cached, compute embeddings only for keys missing from
    it, merge, and overwrite. Different split files (cold-both vs cold-drug)
    cover different subsets of the same underlying pool of sequences/SMILES,
    so the cache has to be the union across every split that's ever used,
    not just whichever split happened to be passed in on a given run.

    Writes into `existing` in place and re-saves it to `path` periodically
    (every save_interval_sec) while computing, not just at the very end -
    interrupting a long run (hundreds of thousands of ligands on a 4GB card
    can take over an hour) only loses the last partial interval, and a
    re-run picks up from the saved point instead of starting over.

    Note: switching --protein-model changes the embedding dimension, so a
    cache built with one protein model is incompatible with another - pass
    a fresh --out-dir when changing it.
    """
    existing = torch.load(path, weights_only=False) if os.path.exists(path) else {}
    missing = [k for k in keys if k not in existing]
    print(f"{path}: {len(existing)} cached, {len(missing)} missing", flush=True)

    def save_and_report():
        atomic_save(existing, path)
        print(f"  ...saved, {len(existing)} entries so far", flush=True)

    if missing:
        compute_fn(missing, device, existing, save_and_report, save_interval_sec)
        atomic_save(existing, path)
        print(f"wrote {path} ({len(existing)} entries)", flush=True)
    return existing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "sequence_based", "data"))
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "sequence_based", "data", "embeddings"),
                         help="pass a different path when using a non-default --protein-model - the cache dims won't match otherwise")
    parser.add_argument("--protein-model", default="esm2_t12_35M_UR50D")
    parser.add_argument("--protein-batch-size", type=int, default=1,
                         help="bump on a big-VRAM box; batch=1 is the safe default for a 4GB card")
    parser.add_argument("--ligand-batch-size", type=int, default=64)
    parser.add_argument("--stage", default="both", choices=["protein", "ligand", "both"],
                         help="run just one encoder, e.g. to monitor/interrupt them independently")
    parser.add_argument("--save-interval", type=int, default=60, help="seconds between incremental cache saves")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, protein model: {args.protein_model}", flush=True)

    dfs = [pd.read_csv(os.path.join(args.data_dir, name)) for name in args.splits]
    all_df = pd.concat(dfs, ignore_index=True)
    sequences = sorted(all_df["sequence"].unique())
    smiles_list = sorted(all_df["smiles"].unique())
    print(f"{len(sequences)} unique sequences, {len(smiles_list)} unique ligands", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.stage in ("protein", "both"):
        top_up(
            os.path.join(args.out_dir, "protein_esm2.pt"), sequences,
            lambda keys, dev, existing, save_cb, interval: compute_protein_embeddings(
                keys, dev, batch_size=args.protein_batch_size, model_name=args.protein_model,
                embeddings=existing, save_callback=save_cb, save_interval_sec=interval,
            ),
            device, args.save_interval,
        )
    if args.stage in ("ligand", "both"):
        top_up(
            os.path.join(args.out_dir, "ligand_chemberta.pt"), smiles_list,
            lambda keys, dev, existing, save_cb, interval: compute_ligand_embeddings(
                keys, dev, batch_size=args.ligand_batch_size,
                embeddings=existing, save_callback=save_cb, save_interval_sec=interval,
            ),
            device, args.save_interval,
        )


if __name__ == "__main__":
    main()
