import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F


class EGNNConv(MessagePassing):
    """
    Equivariant Graph Convolutional Layer (EGNN).
    Operates on both node features (h) and 3D coordinates (x).
    """

    def __init__(self, in_channels, out_channels, edge_dim=0):
        super().__init__(aggr="mean")
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 1 + edge_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, 1, bias=False),
        )

    def forward(self, h, pos, edge_index, edge_attr=None):
        # Calculate distances
        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        # Propagate messages
        out = self.propagate(
            edge_index,
            h=h,
            pos=pos,
            coord_diff=coord_diff,
            radial=radial,
            edge_attr=edge_attr,
        )
        h_new, pos_new = out
        return h_new, pos_new

    def message(self, h_i, h_j, coord_diff, radial, edge_attr):
        if edge_attr is not None:
            feat = torch.cat([h_i, h_j, radial, edge_attr], dim=-1)
        else:
            feat = torch.cat([h_i, h_j, radial], dim=-1)

        m_ij = self.edge_mlp(feat)

        # coordinate message
        pos_m = coord_diff * self.coord_mlp(m_ij)
        return m_ij, pos_m

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        m_ij, pos_m = inputs
        # Aggregate node features
        m_i = super().aggregate(m_ij, index, ptr, dim_size)
        # Aggregate coordinate updates (mean aggregation for stability)
        pos_update = super().aggregate(pos_m, index, ptr, dim_size)
        return m_i, pos_update

    def update(self, aggr_out, h, pos):
        m_i, pos_update = aggr_out
        h_new = self.node_mlp(torch.cat([h, m_i], dim=-1))
        pos_new = pos + pos_update
        return h_new, pos_new


class CrossAttentionLayer(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.ff = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.norm_ff = nn.LayerNorm(feature_dim)
        self.last_attention_weights = None

    def forward(self, ligand_features, protein_features, key_padding_mask=None):
        attention_output, attn_weights = self.attention(
            query=ligand_features,
            key=protein_features,
            value=protein_features,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        self.last_attention_weights = attn_weights.detach().cpu()

        ligand_features = self.norm(ligand_features + attention_output)
        ff_output = self.ff(ligand_features)
        ligand_features = self.norm_ff(ligand_features + ff_output)

        return ligand_features


class BindingAffinity3DModel(nn.Module):
    def __init__(
        self,
        num_ligand_features,
        num_protein_features,
        hidden_channels=256,
        gat_heads=4,
        dropout=0.3,
    ):
        super().__init__()
        self.dropout = dropout
        self.hidden_channels = hidden_channels

        # Tower 1 - Ligand 3D GNN (EGNN)
        self.lig_embed = nn.Linear(num_ligand_features, hidden_channels)
        self.egnn_lig1 = EGNNConv(hidden_channels, hidden_channels, edge_dim=1)
        self.egnn_lig2 = EGNNConv(hidden_channels, hidden_channels, edge_dim=1)
        self.egnn_lig3 = EGNNConv(hidden_channels, hidden_channels, edge_dim=1)

        # Tower 2 - Protein 3D Pocket EGNN
        self.prot_embed = nn.Linear(num_protein_features, hidden_channels)
        self.egnn_prot1 = EGNNConv(hidden_channels, hidden_channels)
        self.egnn_prot2 = EGNNConv(hidden_channels, hidden_channels)
        self.egnn_prot3 = EGNNConv(hidden_channels, hidden_channels)

        # Cross-Attention Layer
        self.cross_attention = CrossAttentionLayer(
            feature_dim=hidden_channels, num_heads=4, dropout=dropout
        )

        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)

    def forward(
        self,
        x_ligand,
        pos_ligand,
        edge_index_ligand,
        edge_attr_ligand,
        batch_ligand,
        x_protein,
        pos_protein,
        edge_index_protein,
        edge_attr_protein,
        batch_protein,
    ):

        # 1. Ligand processing (3D EGNN)
        x_l = self.lig_embed(x_ligand)
        x_l, pos_l = self.egnn_lig1(
            x_l, pos_ligand, edge_index_ligand, edge_attr_ligand
        )
        x_l = F.elu(x_l)
        x_l = F.dropout(x_l, p=self.dropout, training=self.training)

        x_l, pos_l = self.egnn_lig2(x_l, pos_l, edge_index_ligand, edge_attr_ligand)
        x_l = F.elu(x_l)
        x_l = F.dropout(x_l, p=self.dropout, training=self.training)

        x_l, pos_l = self.egnn_lig3(x_l, pos_l, edge_index_ligand, edge_attr_ligand)

        ligand_dense, ligand_mask = to_dense_batch(x_l, batch_ligand)

        # 2. Protein processing (3D Pocket EGNN)
        x_p = self.prot_embed(x_protein)
        x_p, pos_p = self.egnn_prot1(x_p, pos_protein, edge_index_protein, edge_attr_protein)
        x_p = F.elu(x_p)
        x_p = F.dropout(x_p, p=self.dropout, training=self.training)
        
        x_p, pos_p = self.egnn_prot2(x_p, pos_p, edge_index_protein, edge_attr_protein)
        x_p = F.elu(x_p)
        x_p = F.dropout(x_p, p=self.dropout, training=self.training)
        
        x_p, pos_p = self.egnn_prot3(x_p, pos_p, edge_index_protein, edge_attr_protein)
        
        protein_dense, protein_mask = to_dense_batch(x_p, batch_protein)

        # MHA mask requires True where padding
        protein_pad_mask = ~protein_mask

        # 3. Cross-Attention
        x_cross = self.cross_attention(
            ligand_dense, protein_dense, key_padding_mask=protein_pad_mask
        )

        mask_expanded = ligand_mask.unsqueeze(-1)
        x_cross = x_cross * mask_expanded

        sum_features = torch.sum(x_cross, dim=1)
        num_atoms = torch.sum(mask_expanded, dim=1)
        pooled_x = sum_features / (num_atoms + 1e-6)

        # 4. MLP Head
        out = F.relu(self.fc1(pooled_x))
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.fc2(out)
        return out
