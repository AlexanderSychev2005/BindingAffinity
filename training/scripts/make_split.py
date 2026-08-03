"""Build a leak-aware train/val/test split.

train/val: LP-PDBBind (https://github.com/THGLab/LP-PDBBind) CL1-filtered
general+refined complexes, with the CASF-2016 core set removed regardless
of LP-PDBBind's own assignment (LP-PDBBind puts ~40% of the core set into
its own train/val, which would leak our benchmark).

test: CASF-2016 core set (285 complexes), read straight from the official
CoreSet.dat, never used for training.
"""

import argparse
import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_casf2016_core(casf_dir):
    dat_path = os.path.join(casf_dir, "power_scoring", "CoreSet.dat")
    df = pd.read_csv(dat_path, sep=r"\s+", comment=None, skiprows=1,
                      names=["pdb_id", "resl", "year", "affinity", "Ka", "target"])
    return df[["pdb_id", "affinity"]]


def filter_existing(df, data_dir, id_col="pdb_id"):
    keep = df[id_col].apply(
        lambda pid: os.path.exists(os.path.join(data_dir, pid, f"{pid}_pocket.pdb"))
        and (
            os.path.exists(os.path.join(data_dir, pid, f"{pid}_ligand.sdf"))
            or os.path.exists(os.path.join(data_dir, pid, f"{pid}_ligand.mol2"))
        )
    )
    return df[keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lp-pdbbind-csv", default=os.path.join(REPO_ROOT, "LP-PDBBind", "dataset", "LP_PDBBind.csv"))
    parser.add_argument("--casf-dir", default=os.path.join(REPO_ROOT, "CASF-2016"))
    parser.add_argument("--general-set-dir", default=os.path.join(REPO_ROOT, "general-set"))
    parser.add_argument("--clean-level", default="CL1", choices=["CL1", "CL2", "CL3"])
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "training", "data"))
    args = parser.parse_args()

    lp = pd.read_csv(args.lp_pdbbind_csv, index_col=0)
    lp = lp.rename_axis("pdb_id").reset_index()

    non_core = lp[lp["category"] != "core"]
    clean = non_core[non_core[args.clean_level]]

    train = clean[clean["new_split"] == "train"][["pdb_id", "value"]].rename(columns={"value": "affinity"})
    val = clean[clean["new_split"] == "val"][["pdb_id", "value"]].rename(columns={"value": "affinity"})

    train = filter_existing(train, args.general_set_dir)
    val = filter_existing(val, args.general_set_dir)

    test = load_casf2016_core(args.casf_dir)
    core_ids = set(test["pdb_id"])

    leaked_train = train[train["pdb_id"].isin(core_ids)]
    leaked_val = val[val["pdb_id"].isin(core_ids)]
    if len(leaked_train) or len(leaked_val):
        train = train[~train["pdb_id"].isin(core_ids)]
        val = val[~val["pdb_id"].isin(core_ids)]

    os.makedirs(args.out_dir, exist_ok=True)
    train.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(args.out_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(args.out_dir, "test_casf2016.csv"), index=False)

    print(f"train: {len(train)} (dropped {len(leaked_train)} CASF-2016 leaks)")
    print(f"val:   {len(val)} (dropped {len(leaked_val)} CASF-2016 leaks)")
    print(f"test:  {len(test)} (CASF-2016 core, never trained on)")


if __name__ == "__main__":
    main()
