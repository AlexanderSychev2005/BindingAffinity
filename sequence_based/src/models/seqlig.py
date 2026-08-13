"""Fusion heads over frozen ESM-2 / ChemBERTa embeddings. No structure, no
graph - both models only ever see one pooled vector per protein and one
pooled vector per ligand (that's what's cached to disk).

Real per-residue/per-atom cross-attention isn't a "cheap" option here: the
embedding cache only stores post-pooling vectors (caching per-token/per-
residue representations for 205k ligands x up to 256 tokens x 768 dims, or
5.4k proteins x up to 1500 residues x 480 dims, would be tens to hundreds
of GB). Token-level cross-attention would mean running both encoders live
during training instead of reading a cache - at that point you're already
paying for fine-tuning, so it belongs with that step, not this one.

SeqLigBilinearModel is the cheap upgrade that's actually available at the
pooled-vector level: add the elementwise (Hadamard) product of the two
projected vectors as an explicit interaction term, on top of the plain
concatenation. Standard trick in two-tower DTA/recsys models for capturing
multiplicative interactions a plain concat+MLP struggles to learn.
"""

import torch.nn as nn
import torch


class SeqLigModel(nn.Module):
    def __init__(self, protein_dim, ligand_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(protein_dim + ligand_dim, hidden_dim),
            nn.LeakyReLU(), nn.Dropout(dropout), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(), nn.Dropout(dropout), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, protein_emb, ligand_emb):
        x = torch.cat([protein_emb, ligand_emb], dim=-1)
        return self.net(x).view(-1)


class SeqLigBilinearModel(nn.Module):
    def __init__(self, protein_dim, ligand_dim, proj_dim=256, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.protein_proj = nn.Linear(protein_dim, proj_dim)
        self.ligand_proj = nn.Linear(ligand_dim, proj_dim)
        self.net = nn.Sequential(
            nn.Linear(proj_dim * 3, hidden_dim),
            nn.LeakyReLU(), nn.Dropout(dropout), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(), nn.Dropout(dropout), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, protein_emb, ligand_emb):
        p = self.protein_proj(protein_emb)
        l = self.ligand_proj(ligand_emb)
        x = torch.cat([p, l, p * l], dim=-1)
        return self.net(x).view(-1)
