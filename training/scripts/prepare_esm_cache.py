import torch
from transformers import AutoTokenizer, EsmModel
import pandas as pd
import os
from tqdm import tqdm


def parse_pdb_ca_seq(pdb_file):
    aa_map = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    seq = ""
    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM  "):
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    res_name = line[17:20].strip()
                    seq += aa_map.get(res_name, "X")
    return seq


def main():
    print("Loading ESM-2 model (facebook/esm2_t12_35M_UR50D)...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")

    print("Loading dataset...")
    df = pd.read_csv("pdbbind_refined_dataset.csv").dropna()
    unique_pdbs = df[["pdb_id"]].drop_duplicates()
    print(f"Found {len(unique_pdbs)} unique proteins to process.")

    os.makedirs("refined-set", exist_ok=True)

    for _, row in tqdm(unique_pdbs.iterrows(), total=len(unique_pdbs)):
        pdb_id = row["pdb_id"]

        dir_path = os.path.join("refined-set", pdb_id)
        if not os.path.exists(dir_path):
            continue

        cache_path = os.path.join(dir_path, "esm2_35m.pt")
        if os.path.exists(cache_path):
            continue

        pdb_path = os.path.join(dir_path, f"{pdb_id}_protein.pdb")
        if not os.path.exists(pdb_path):
            continue

        seq = parse_pdb_ca_seq(pdb_path)
        if len(seq) == 0:
            continue

        # ESM-2 handles up to 1024 tokens. Truncate sequence if it's too long.
        seq = seq[:1022]

        inputs = tokenizer(seq, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Strip <cls> and <eos>
        embeddings = outputs.last_hidden_state[0, 1:-1, :].cpu()

        torch.save(embeddings, cache_path)

    print("Finished computing ESM-2 embeddings!")


if __name__ == "__main__":
    main()
