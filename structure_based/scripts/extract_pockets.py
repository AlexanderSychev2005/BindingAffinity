"""Re-extract pockets straight from the full *_protein.pdb, at a fixed 5A
by-residue radius (matching GIGN's `byres <ligand> around 5` convention),
bypassing the ~8A pockets from the old (deleted) v2-3d-complex pipeline.

No PyMOL dependency - the pip wheel doesn't run on Windows (missing
bundled DLLs). Plain PDB-line parsing + numpy distance does the same job:
select whole residues with >=1 atom within `radius` of the ligand, write
them out as a smaller PDB.
"""

import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_ligand_positions(complex_dir, pdb_id):
    sdf_path = os.path.join(complex_dir, f"{pdb_id}_ligand.sdf")
    if os.path.exists(sdf_path):
        mol = next((m for m in Chem.SDMolSupplier(sdf_path, sanitize=False) if m is not None), None)
        if mol is not None:
            return mol.GetConformer().GetPositions()
    mol2_path = os.path.join(complex_dir, f"{pdb_id}_ligand.mol2")
    if os.path.exists(mol2_path):
        mol = Chem.MolFromMol2File(mol2_path, sanitize=False)
        if mol is not None:
            return mol.GetConformer().GetPositions()
    return None


def extract_pocket(protein_pdb_path, ligand_pos, radius=5.0):
    atom_lines, coords, res_keys = [], [], []
    with open(protein_pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            chain, res_seq, icode = line[21], line[22:26], line[26]
            atom_lines.append(line)
            coords.append((x, y, z))
            res_keys.append((chain, res_seq, icode))

    if not atom_lines:
        return None

    coords = np.array(coords)
    min_dist = np.linalg.norm(coords[:, None, :] - ligand_pos[None, :, :], axis=-1).min(axis=1)

    keep_residues = {res_keys[i] for i in range(len(res_keys)) if min_dist[i] < radius}
    kept_lines = [line for line, key in zip(atom_lines, res_keys) if key in keep_residues]
    return kept_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "structure_based", "general-set"))
    parser.add_argument("--splits", nargs="+", default=["training/data/ablation_ALL/train.csv", "training/data/ablation_ALL/val.csv"])
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--out-name", default="pocket_v2.pdb")
    args = parser.parse_args()

    pdb_ids = set()
    for split in args.splits:
        df = pd.read_csv(os.path.join(REPO_ROOT, split))
        pdb_ids.update(df["pdb_id"].tolist())

    ok, fail = 0, 0
    for pid in tqdm(sorted(pdb_ids)):
        complex_dir = os.path.join(args.data_dir, pid)
        out_path = os.path.join(complex_dir, args.out_name)
        if os.path.exists(out_path):
            ok += 1
            continue

        ligand_pos = load_ligand_positions(complex_dir, pid)
        protein_path = os.path.join(complex_dir, f"{pid}_protein.pdb")
        if ligand_pos is None or not os.path.exists(protein_path):
            fail += 1
            continue

        lines = extract_pocket(protein_path, ligand_pos, args.radius)
        if not lines:
            fail += 1
            continue

        with open(out_path, "w") as f:
            f.writelines(lines)
            f.write("END\n")
        ok += 1

    print(f"done: ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
