import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.ligand_to_protein = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.protein_to_ligand = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm_l = nn.LayerNorm(feature_dim)
        self.norm_p = nn.LayerNorm(feature_dim)

        self.ff_l = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.ff_p = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.norm_ff_l = nn.LayerNorm(feature_dim)
        self.norm_ff_p = nn.LayerNorm(feature_dim)

    def forward(self, ligand_feat, protein_feat, l_mask=None, p_mask=None):
        p_padding_mask = ~p_mask if p_mask is not None else None
        l_padding_mask = ~l_mask if l_mask is not None else None

        l_attn, _ = self.ligand_to_protein(
            query=ligand_feat,
            key=protein_feat,
            value=protein_feat,
            key_padding_mask=p_padding_mask,
        )
        ligand_feat = self.norm_l(ligand_feat + l_attn)
        ligand_feat = self.norm_ff_l(ligand_feat + self.ff_l(ligand_feat))

        p_attn, _ = self.protein_to_ligand(
            query=protein_feat,
            key=ligand_feat,
            value=ligand_feat,
            key_padding_mask=l_padding_mask,
        )
        protein_feat = self.norm_p(protein_feat + p_attn)
        protein_feat = self.norm_ff_p(protein_feat + self.ff_p(protein_feat))

        return ligand_feat, protein_feat


class SOTABindingModel(nn.Module):
    def __init__(
        self,
        ligand_in_dim,
        protein_in_dim,
        ligand_edge_dim,
        protein_edge_dim,
        hidden_dim=512,
        n_heads=8,
        dropout=0.2,
    ):
        super().__init__()

        # Ligand Encoder with Edge Features
        self.l_conv1 = GATv2Conv(
            ligand_in_dim,
            hidden_dim,
            heads=n_heads,
            concat=False,
            edge_dim=ligand_edge_dim,
        )
        self.l_conv2 = GATv2Conv(
            hidden_dim,
            hidden_dim,
            heads=n_heads,
            concat=False,
            edge_dim=ligand_edge_dim,
        )

        # Protein Encoder with Edge Features (Distances)
        self.p_conv1 = GATv2Conv(
            protein_in_dim,
            hidden_dim,
            heads=n_heads,
            concat=False,
            edge_dim=protein_edge_dim,
        )
        self.p_conv2 = GATv2Conv(
            hidden_dim,
            hidden_dim,
            heads=n_heads,
            concat=False,
            edge_dim=protein_edge_dim,
        )

        # Interaction
        self.cross_attn = BidirectionalCrossAttention(
            hidden_dim, num_heads=n_heads, dropout=dropout
        )

        # Prediction Heads
        self.fc_shared = nn.Linear(hidden_dim * 2, hidden_dim)

        # Head 1: Affinity (pKd)
        self.fc_affinity = nn.Linear(hidden_dim, 1)

        # Head 2: Uncertainty (Log Variance)
        # We predict log(sigma^2) to ensure positivity
        self.fc_uncertainty = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, l_data, p_data):
        # 1. Ligand GNN
        lx, l_edge, l_edge_attr, l_batch = (
            l_data.x,
            l_data.edge_index,
            l_data.edge_attr,
            l_data.batch,
        )
        lx = F.elu(self.l_conv1(lx, l_edge, l_edge_attr))
        lx = F.elu(self.l_conv2(lx, l_edge, l_edge_attr))

        # 2. Protein GNN
        px, p_edge, p_edge_attr, p_batch = (
            p_data.x,
            p_data.edge_index,
            p_data.edge_attr,
            p_data.batch,
        )
        px = F.elu(self.p_conv1(px, p_edge, p_edge_attr))
        px = F.elu(self.p_conv2(px, p_edge, p_edge_attr))

        # 3. Interaction
        l_dense, l_mask = to_dense_batch(lx, l_batch)
        p_dense, p_mask = to_dense_batch(px, p_batch)
        l_dense, p_dense = self.cross_attn(l_dense, p_dense, l_mask, p_mask)

        # 4. Pooling
        l_pooled = (l_dense * l_mask.unsqueeze(-1)).sum(dim=1) / (
            l_mask.sum(dim=1, keepdim=True) + 1e-6
        )
        p_pooled = (p_dense * p_mask.unsqueeze(-1)).sum(dim=1) / (
            p_mask.sum(dim=1, keepdim=True) + 1e-6
        )

        # 5. Output
        combined = torch.cat([l_pooled, p_pooled], dim=-1)
        shared = F.relu(self.fc_shared(combined))
        shared = self.dropout(shared)

        affinity = self.fc_affinity(shared)
        uncertainty = self.fc_uncertainty(shared)

        return affinity, uncertainty


def gaussian_nll_loss(pred, target, log_var):
    """
    Gaussian Negative Log Likelihood Loss for uncertainty estimation.
    """
    precision = torch.exp(-log_var)
    return torch.mean(0.5 * precision * (target - pred) ** 2 + 0.5 * log_var)
