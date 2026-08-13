"""Parse the raw BindingDB_All.tsv dump into a clean
(target_key, sequence, smiles, affinity) table.

Keeps only single-chain targets with an exact (non-censored) Ki, Kd, or
(with --include-ic50) IC50 measurement - censored values ("<1000",
">10000") aren't usable for regression. Priority is Kd > Ki > IC50 (Kd/Ki
are direct binding constants; IC50 is assay-dependent). When --include-ic50
is set, IC50 values are divided by 2.3 before use - the conversion BIOPTIC's
own paper (arxiv 2406.14572) documents as "the common difference between
IC50 and Ki values in databases". KIBA similarly pools Kd/Ki/IC50 into one
score, so mixing them (with this adjustment) is standard field practice,
not a shortcut - see metrics_summary.csv notes for the Ki/Kd-only vs
+IC50 comparison this is meant to enable.

Duplicate (target, ligand) pairs are collapsed to the median affinity in
log space.
"""

import argparse
import os

import pandas as pd
from rdkit import Chem, rdBase
from tqdm import tqdm

rdBase.DisableLog("rdApp.*")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

USECOLS = [
    "Ligand SMILES",
    "Ki (nM)",
    "Kd (nM)",
    "IC50 (nM)",
    "Number of Protein Chains in Target (>1 implies a multichain complex)",
    "BindingDB Target Chain Sequence 1",
    "UniProt (SwissProt) Primary ID of Target Chain 1",
    "UniProt (TrEMBL) Primary ID of Target Chain 1",
]

MAX_SEQ_LEN = 1500
MIN_SEQ_LEN = 20
IC50_TO_KI_FACTOR = 2.3


def clean_affinity_nm(series):
    """Drop censored ("<x", ">x") and non-numeric values, return float nM."""
    s = series.astype(str).str.strip()
    s = s.where(~s.str.contains("[<>]", regex=True), None)
    return pd.to_numeric(s, errors="coerce")


def process_chunk(chunk, include_ic50):
    chunk = chunk[chunk["Number of Protein Chains in Target (>1 implies a multichain complex)"] == 1]

    kd = clean_affinity_nm(chunk["Kd (nM)"])
    ki = clean_affinity_nm(chunk["Ki (nM)"])
    ic50_adj = (
        clean_affinity_nm(chunk["IC50 (nM)"]) / IC50_TO_KI_FACTOR
        if include_ic50 else pd.Series(pd.NA, index=chunk.index, dtype="float64")
    )

    # priority Kd > Ki > IC50 - direct binding constants before assay-dependent ones
    value_nm = kd.where(kd.notna(), ki.where(ki.notna(), ic50_adj))
    source = pd.Series("Kd", index=chunk.index).where(kd.notna(), pd.Series("Ki", index=chunk.index).where(ki.notna(), "IC50"))

    chunk = chunk.assign(value_nm=value_nm, affinity_source=source)
    chunk = chunk[chunk["value_nm"] > 0]

    chunk = chunk.rename(columns={
        "Ligand SMILES": "smiles",
        "BindingDB Target Chain Sequence 1": "sequence",
    })
    chunk["target_key"] = chunk["UniProt (SwissProt) Primary ID of Target Chain 1"].where(
        chunk["UniProt (SwissProt) Primary ID of Target Chain 1"].notna(),
        chunk["UniProt (TrEMBL) Primary ID of Target Chain 1"],
    )
    chunk["target_key"] = chunk["target_key"].where(chunk["target_key"].notna(), chunk["sequence"])

    chunk = chunk.dropna(subset=["smiles", "sequence", "target_key"])
    chunk["sequence"] = chunk["sequence"].str.upper()
    # BindingDB's own curation occasionally has garbage in this field - e.g. residue-numbering
    # ("...ENSCK510520530540550ATGQ...") pasted straight into the sequence with no separator.
    # Letters-only, matching the amino acid alphabet, catches that and any other corruption.
    chunk = chunk[chunk["sequence"].str.match(r"^[A-Z]+$")]
    chunk = chunk[chunk["sequence"].str.len().between(MIN_SEQ_LEN, MAX_SEQ_LEN)]
    chunk["affinity"] = 9.0 - chunk["value_nm"].apply(lambda v: __import__("math").log10(v))

    return chunk[["target_key", "sequence", "smiles", "affinity", "affinity_source"]]


def valid_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv-path", default=os.path.join(REPO_ROOT, "sequence_based", "data", "raw", "BindingDB_All.tsv"))
    parser.add_argument("--out-csv", default=None, help="defaults to bindingdb_clean.csv, or bindingdb_clean_ic50.csv with --include-ic50")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--include-ic50", action="store_true",
                         help="also use IC50 (divided by 2.3) when no Kd/Ki is available - adds target diversity, see module docstring")
    args = parser.parse_args()
    out_csv = args.out_csv or os.path.join(
        REPO_ROOT, "sequence_based", "data", "bindingdb_clean_ic50.csv" if args.include_ic50 else "bindingdb_clean.csv"
    )

    kept_chunks = []
    total_rows = 0
    reader = pd.read_csv(
        args.tsv_path, sep="\t", usecols=USECOLS, chunksize=args.chunksize,
        quoting=3, on_bad_lines="skip", low_memory=False, encoding_errors="replace",
    )
    for chunk in tqdm(reader, desc="scanning BindingDB"):
        total_rows += len(chunk)
        kept = process_chunk(chunk, args.include_ic50)
        if len(kept):
            kept_chunks.append(kept)

    df = pd.concat(kept_chunks, ignore_index=True)
    measurement = "Ki/Kd/IC50" if args.include_ic50 else "Ki/Kd"
    print(f"scanned {total_rows} rows -> {len(df)} with a clean single-chain {measurement} measurement")

    tqdm.pandas(desc="validating SMILES")
    df = df[df["smiles"].progress_apply(valid_smiles)]
    print(f"{len(df)} rows with RDKit-valid SMILES")

    df = df.groupby(["target_key", "smiles"], as_index=False).agg(
        sequence=("sequence", "first"),
        affinity=("affinity", "median"),
        affinity_source=("affinity_source", "first"),
        n_measurements=("affinity", "size"),
    )
    print(f"{len(df)} unique (target, ligand) pairs after dedup")
    print(f"unique targets: {df['target_key'].nunique()}, unique ligands: {df['smiles'].nunique()}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
