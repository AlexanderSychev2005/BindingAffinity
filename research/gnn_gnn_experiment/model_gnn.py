import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
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
        # ligand_feat: [B, N_l, D], protein_feat: [B, N_p, D]
        # MHA masks: True means IGNORE
        p_padding_mask = ~p_mask if p_mask is not None else None
        l_padding_mask = ~l_mask if l_mask is not None else None

        # 1. Ligand attends to Protein
        l_attn, _ = self.ligand_to_protein(
            query=ligand_feat,
            key=protein_feat,
            value=protein_feat,
            key_padding_mask=p_padding_mask,
        )
        ligand_feat = self.norm_l(ligand_feat + l_attn)
        ligand_feat = self.norm_ff_l(ligand_feat + self.ff_l(ligand_feat))

        # 2. Protein attends to Ligand
        p_attn, _ = self.protein_to_ligand(
            query=protein_feat,
            key=ligand_feat,
            value=ligand_feat,
            key_padding_mask=l_padding_mask,
        )
        protein_feat = self.norm_p(protein_feat + p_attn)
        protein_feat = self.norm_ff_p(protein_feat + self.ff_p(protein_feat))

        return ligand_feat, protein_feat


class GNNGNNModel(nn.Module):
    def __init__(
        self, ligand_in_dim, protein_in_dim, hidden_dim=256, n_heads=4, dropout=0.2
    ):
        super().__init__()

        # Ligand Encoder
        self.l_conv1 = GATv2Conv(ligand_in_dim, hidden_dim, heads=n_heads, concat=False)
        self.l_conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=n_heads, concat=False)
        self.l_conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=n_heads, concat=False)

        # Protein Encoder
        self.p_conv1 = GATv2Conv(
            protein_in_dim, hidden_dim, heads=n_heads, concat=False
        )
        self.p_conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=n_heads, concat=False)
        self.p_conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=n_heads, concat=False)

        # Interaction Layer
        self.cross_attn = BidirectionalCrossAttention(
            hidden_dim, num_heads=n_heads, dropout=dropout
        )

        # Prediction Head
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, l_data, p_data):
        # 1. Ligand GNN
        lx, l_edge, l_batch = l_data.x, l_data.edge_index, l_data.batch
        lx = F.elu(self.l_conv1(lx, l_edge))
        lx = F.elu(self.l_conv2(lx, l_edge))
        lx = F.elu(self.l_conv3(lx, l_edge))

        # 2. Protein GNN
        px, p_edge, p_batch = p_data.x, p_data.edge_index, p_data.batch
        px = F.elu(self.p_conv1(px, p_edge))
        px = F.elu(self.p_conv2(px, p_edge))
        px = F.elu(self.p_conv3(px, p_edge))

        # 3. Prepare for Cross-Attention (Dense Batch)
        l_dense, l_mask = to_dense_batch(lx, l_batch)
        p_dense, p_mask = to_dense_batch(px, p_batch)

        # 4. Cross-Attention
        l_dense, p_dense = self.cross_attn(l_dense, p_dense, l_mask, p_mask)

        # 5. Pooling
        # Masked mean pooling for ligand
        l_pooled = (l_dense * l_mask.unsqueeze(-1)).sum(dim=1) / (
            l_mask.sum(dim=1, keepdim=True) + 1e-6
        )
        # Masked mean pooling for protein
        p_pooled = (p_dense * p_mask.unsqueeze(-1)).sum(dim=1) / (
            p_mask.sum(dim=1, keepdim=True) + 1e-6
        )

        # 6. Concatenate and Predict
        out = torch.cat([l_pooled, p_pooled], dim=-1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out


if __name__ == "__main__":
    # Test with dummy data
    from torch_geometric.data import Batch, Data

    l_in, p_in = 79, 21  # Example dims from dataset
    model = GNNGNNModel(l_in, p_in)

    # Fake batch
    l1 = Data(
        x=torch.randn(10, l_in),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    l2 = Data(
        x=torch.randn(15, l_in),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    l_batch = Batch.from_data_list([l1, l2])

    p1 = Data(
        x=torch.randn(100, p_in),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    p2 = Data(
        x=torch.randn(120, p_in),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    p_batch = Batch.from_data_list([p1, p2])

    output = model(l_batch, p_batch)
    print(f"Output shape: {output.shape}")
