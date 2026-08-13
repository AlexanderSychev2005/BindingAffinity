"""End-to-end fine-tuned counterpart to models/seqlig.py: ESM-2 and
ChemBERTa are loaded with gradients on and trained jointly with the fusion
head, instead of being frozen feature extractors read from a cache. Needs
real GPU memory (cloud-only) - not something a 4GB local card can run.

Pooling conventions are kept identical to the frozen path (embeddings.py)
so results stay comparable: ligand pooling includes the tokenizer's special
tokens (mean over `attention_mask`, same as compute_ligand_embeddings);
protein pooling excludes BOS/EOS (same as compute_protein_embeddings's
`rep[1:true_len-1]`), just computed batched instead of per-sequence.
"""

import torch
import torch.nn as nn

from embeddings import LIGAND_EMBED_DIM, PROTEIN_EMBED_DIMS, load_ligand_encoder, load_protein_encoder


def protein_content_mask(tokens, padding_idx):
    """True for real-residue positions, False for BOS, EOS, and padding."""
    not_pad = tokens != padding_idx
    lengths = not_pad.sum(dim=1)
    mask = not_pad.clone()
    mask[:, 0] = False  # BOS
    mask[torch.arange(tokens.size(0)), lengths - 1] = False  # EOS
    return mask


def masked_mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)


def enable_gradient_checkpointing(esm_model):
    """Wrap each ESM-2 transformer layer so its activations are recomputed
    during backward instead of kept in memory. Self-attention is O(L^2) per
    layer; with 36 layers and sequences up to ~1500 residues, keeping every
    layer's activations for backward is what actually exhausts 192GB (not
    fragmentation) - this trades ~30% more compute for a large, predictable
    memory cut, the standard fix for fine-tuning transformers at this scale.
    """
    for layer in esm_model.layers:
        orig_forward = layer.forward

        def checkpointed_forward(*args, _orig=orig_forward, **kwargs):
            return torch.utils.checkpoint.checkpoint(_orig, *args, use_reentrant=False, **kwargs)

        layer.forward = checkpointed_forward


class SeqLigFinetuneModel(nn.Module):
    def __init__(self, device, protein_model_name, proj_dim=256, hidden_dim=256, dropout=0.2, gradient_checkpointing=False):
        super().__init__()
        self.protein_model, self.protein_alphabet = load_protein_encoder(device, protein_model_name, train_mode=True)
        self.protein_num_layers = self.protein_model.num_layers
        if gradient_checkpointing:
            enable_gradient_checkpointing(self.protein_model)
        self.ligand_tokenizer, self.ligand_model = load_ligand_encoder(train_mode=True)
        self.ligand_model.to(device)

        protein_dim = PROTEIN_EMBED_DIMS[protein_model_name]
        self.protein_proj = nn.Linear(protein_dim, proj_dim)
        self.ligand_proj = nn.Linear(LIGAND_EMBED_DIM, proj_dim)
        self.head = nn.Sequential(
            nn.Linear(proj_dim * 3, hidden_dim),
            nn.LeakyReLU(), nn.Dropout(dropout), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(), nn.Dropout(dropout), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, protein_tokens, ligand_inputs):
        protein_out = self.protein_model(
            protein_tokens, repr_layers=[self.protein_num_layers], return_contacts=False
        )["representations"][self.protein_num_layers]
        p_mask = protein_content_mask(protein_tokens, self.protein_alphabet.padding_idx)
        p = self.protein_proj(masked_mean_pool(protein_out, p_mask))

        ligand_out = self.ligand_model(**ligand_inputs).last_hidden_state
        l = self.ligand_proj(masked_mean_pool(ligand_out, ligand_inputs["attention_mask"]))

        x = torch.cat([p, l, p * l], dim=-1)
        return self.head(x).view(-1)
