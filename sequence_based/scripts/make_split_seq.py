"""Leak-proof splits for the sequence+ligand dataset, two flavors:

  --mode cold-both (default): both the protein (`target_key`, a UniProt ID)
    and the ligand (Bemis-Murcko scaffold) are cluster-split, and a row
    survives only if its target cluster AND its scaffold cluster agree on
    the same split. This is the hardest, most honest setting - "does this
    generalize to a target and chemistry the model has never seen" - but
    it throws away most rows to get there.

  --mode cold-drug: only the ligand scaffold is cluster-split; a target can
    appear in both train and test. This matches how BIOPTIC's own B1 case
    study was actually validated (LRRK2 - a target they already had data on
    - screened for *novel chemistry*, not a novel target). Answers "does
    this find new molecules for a target I already have some data on."

Report both - they answer different questions about the model.

ponytail: protein clustering is exact UniProt-ID grouping, not sequence-
identity clustering (no MMseqs2 here). This catches identical-protein
leakage but not homolog leakage. Upgrade path if val/test metrics look
too optimistic: cluster target_key by sequence identity (MMseqs2 at
~40%) the way LP-PDBBind does, same escalation as CL1->CL2->CL3.
"""

import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

rdBase.DisableLog("rdApp.*")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    return scaffold_smiles if scaffold_smiles else smiles  # no ring system: scaffold == itself


def cluster_split(cluster_ids, val_frac, test_frac, seed):
    rng = np.random.default_rng(seed)
    clusters = rng.permutation(sorted(set(cluster_ids)))
    n_val = int(len(clusters) * val_frac)
    n_test = int(len(clusters) * test_frac)
    test_clusters = set(clusters[:n_test])
    val_clusters = set(clusters[n_test:n_test + n_val])
    train_clusters = set(clusters[n_test + n_val:])
    return train_clusters, val_clusters, test_clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", default=os.path.join(REPO_ROOT, "sequence_based", "data", "bindingdb_clean.csv"))
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "sequence_based", "data"))
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="cold-both", choices=["cold-both", "cold-drug"])
    parser.add_argument("--out-prefix", default=None, help="defaults to bindingdb / bindingdb_colddrug - override to avoid clobbering an existing split, e.g. for a different --in-csv")
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)
    print(f"loaded {len(df)} rows")

    df["scaffold"] = df["smiles"].apply(murcko_scaffold)
    df = df.dropna(subset=["scaffold"])
    print(f"{len(df)} rows with a valid scaffold, {df['scaffold'].nunique()} unique scaffolds, "
          f"{df['target_key'].nunique()} unique targets")

    train_s, val_s, test_s = cluster_split(df["scaffold"], args.val_frac, args.test_frac, args.seed + 1)

    if args.mode == "cold-both":
        train_t, val_t, test_t = cluster_split(df["target_key"], args.val_frac, args.test_frac, args.seed)

        def assign(row):
            t_split = "train" if row["target_key"] in train_t else "val" if row["target_key"] in val_t else "test"
            s_split = "train" if row["scaffold"] in train_s else "val" if row["scaffold"] in val_s else "test"
            return t_split if t_split == s_split else None

        prefix = args.out_prefix or "bindingdb"
    else:
        def assign(row):
            return "train" if row["scaffold"] in train_s else "val" if row["scaffold"] in val_s else "test"

        prefix = args.out_prefix or "bindingdb_colddrug"

    df["split"] = df.apply(assign, axis=1)
    kept = df.dropna(subset=["split"])
    print(f"{args.mode} split kept {len(kept)}/{len(df)} rows ({len(kept) / len(df):.1%})")

    os.makedirs(args.out_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        part = kept[kept["split"] == split][["target_key", "sequence", "smiles", "affinity"]]
        part.to_csv(os.path.join(args.out_dir, f"{prefix}_{split}.csv"), index=False)
        print(f"{split}: {len(part)} rows, {part['target_key'].nunique()} targets, {part['smiles'].nunique()} ligands")


if __name__ == "__main__":
    main()
