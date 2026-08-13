import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class SeqLigDataset(Dataset):
    def __init__(self, csv_path, protein_emb, ligand_emb):
        df = pd.read_csv(csv_path)
        df = df[df["sequence"].isin(protein_emb) & df["smiles"].isin(ligand_emb)]
        self.rows = list(df[["sequence", "smiles", "affinity"]].itertuples(index=False, name=None))
        self.protein_emb = protein_emb
        self.ligand_emb = ligand_emb

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        sequence, smiles, affinity = self.rows[idx]
        return self.protein_emb[sequence], self.ligand_emb[smiles], torch.tensor(affinity, dtype=torch.float)

    @staticmethod
    def collate(batch):
        protein = torch.stack([b[0] for b in batch])
        ligand = torch.stack([b[1] for b in batch])
        y = torch.stack([b[2] for b in batch])
        return protein, ligand, y


def load_embeddings(embed_dir):
    protein_emb = torch.load(os.path.join(embed_dir, "protein_esm2.pt"), weights_only=False)
    ligand_emb = torch.load(os.path.join(embed_dir, "ligand_chemberta.pt"), weights_only=False)
    return protein_emb, ligand_emb
