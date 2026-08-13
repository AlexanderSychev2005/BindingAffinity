"""Dataset for end-to-end fine-tuning: unlike dataset_seqlig.py, this reads
raw sequence/SMILES strings and tokenizes them per-batch (both encoders run
live in the training loop, gradients on), instead of looking them up in a
precomputed frozen-embedding cache.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class RawSeqLigDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.rows = list(df[["sequence", "smiles", "affinity"]].itertuples(index=False, name=None))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def make_collate(esm_alphabet, ligand_tokenizer, max_ligand_len=256, max_protein_len=None):
    """max_protein_len: truncate residues beyond this (bounds worst-case
    self-attention memory - a batch with several near-max-length sequences
    can spike VRAM badly since attention is O(L^2) and everyone in a batch
    pads to the longest member). None = no truncation.
    """
    batch_converter = esm_alphabet.get_batch_converter()

    def collate(batch):
        sequences, smiles_list, affinities = zip(*batch)
        if max_protein_len is not None:
            sequences = [s[:max_protein_len] for s in sequences]
        _, _, protein_tokens = batch_converter([(str(i), s) for i, s in enumerate(sequences)])
        ligand_inputs = ligand_tokenizer(
            list(smiles_list), return_tensors="pt", padding=True, truncation=True, max_length=max_ligand_len
        )
        y = torch.tensor(affinities, dtype=torch.float)
        return protein_tokens, ligand_inputs, y

    return collate
