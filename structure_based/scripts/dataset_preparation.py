import os
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem
from tqdm import tqdm

PDBBIND_PATH = "refined-set"
INDEX_NAME = "INDEX_refined_data.2020"


def get_ligand_smiles(pdb_id, pdb_dir_path):
    """
    Get the SMILES representation of the ligand.
    """

    sdf_path = os.path.join(pdb_dir_path, f"{pdb_id}_ligand.sdf")
    mol2_path = os.path.join(pdb_dir_path, f"{pdb_id}_ligand.mol2")
    if os.path.exists(sdf_path):
        try:
            sfd_file = Chem.SDMolSupplier(sdf_path)
            if sfd_file:
                mol = sfd_file[0]
        except Exception:
            mol = None

    if mol is None and os.path.exists(mol2_path):
        try:
            mol = Chem.MolFromMol2File(mol2_path)
        except Exception:
            mol = None
    if mol is not None:
        smiles = Chem.MolToSmiles(mol)
        return smiles
    else:
        return None


def get_protein_sequence(pdb_id, pdb_dir_path):
    """
    Get the protein sequence of the protein.
    """
    protein_path = os.path.join(pdb_dir_path, f"{pdb_id}_protein.pdb")
    pdbparser = PDBParser()
    structure = pdbparser.get_structure(pdb_id, protein_path)
    sequences = []

    for model in structure:
        for chain in model:
            sequence = ""
            for residue in chain:
                if residue.get_id()[0] == " " and is_aa(
                    residue.get_resname(), standard=True
                ):
                    sequence += seq1(residue.get_resname())

            sequences.append(sequence)
    longest_sequence = max(sequences, key=len)
    return longest_sequence


def main():
    final_data = []

    index_data = {}

    index_file_path = os.path.join(PDBBIND_PATH, "index", INDEX_NAME)
    with open(index_file_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            pdb_id = parts[0]
            print(pdb_id)
            affinity = parts[3]

            index_data[pdb_id] = affinity
    print(f"Loaded index data for {len(index_data)} entries")

    for pdb_id, affinity in tqdm(index_data.items()):
        pdb_id_path = os.path.join(PDBBIND_PATH, pdb_id)

        smiles = get_ligand_smiles(pdb_id, pdb_id_path)
        sequence = get_protein_sequence(pdb_id, pdb_id_path)
        if smiles is not None or sequence is not None:
            final_data.append(
                {
                    "pdb_id": pdb_id,
                    "smiles": smiles,
                    "sequence": sequence,
                    "affinity": affinity,
                }
            )

    df = pd.DataFrame(final_data)
    df.to_csv("pdbbind_refined_dataset.csv", index=False)


# pdb_id = "1a1e"
# PDF_ID_PATH = os.path.join(PDBBIND_PATH, pdb_id)
#
# smiles = get_ligand_smiles(pdb_id, PDF_ID_PATH)
# print(smiles)
#
# sequence = get_protein_sequence(pdb_id, PDF_ID_PATH)
# print(sequence)

if __name__ == "__main__":
    main()
