"""Pocket-graph binding affinity model (GIGN-style: arxiv/jpclett.2c03906).

One joint graph over ligand + pocket atoms, with two edge types handled by
separate message passing: covalent (chemical bonds, within ligand or within
pocket) and noncovalent (distance-based cross edges between ligand and
pocket atoms). No coordinate updates and no attention - geometry only enters
as an RBF distance embedding that gates each message.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool


def rbf_expand(dist, d_min=0.0, d_max=6.0, num_bases=9):
    centers = torch.linspace(d_min, d_max, num_bases, device=dist.device)
    sigma = (d_max - d_min) / num_bases
    return torch.exp(-((dist.unsqueeze(-1) - centers) / sigma) ** 2)


class HeteroInteractionLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rbf=9, dropout=0.1):
        super().__init__(aggr="add")
        self.node_mlp_cov = nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.Dropout(dropout),
            nn.LeakyReLU(), nn.BatchNorm1d(out_channels),
        )
        self.node_mlp_noncov = nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.Dropout(dropout),
            nn.LeakyReLU(), nn.BatchNorm1d(out_channels),
        )
        self.gate_cov = nn.Sequential(nn.Linear(num_rbf, in_channels), nn.SiLU())
        self.gate_noncov = nn.Sequential(nn.Linear(num_rbf, in_channels), nn.SiLU())
        self.num_rbf = num_rbf

    def forward(self, x, pos, edge_index_cov, edge_index_noncov):
        msg_cov = self._propagate_typed(x, pos, edge_index_cov, self.gate_cov)
        msg_noncov = self._propagate_typed(x, pos, edge_index_noncov, self.gate_noncov)
        return self.node_mlp_cov(x + msg_cov) + self.node_mlp_noncov(x + msg_noncov)

    def _propagate_typed(self, x, pos, edge_index, gate_mlp):
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        gate = gate_mlp(rbf_expand(dist, num_bases=self.num_rbf))
        return self.propagate(edge_index, x=x, gate=gate)

    def message(self, x_j, gate):
        return x_j * gate


class GIGNStyleModel(nn.Module):
    def __init__(self, node_dim, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.SiLU())
        self.layers = nn.ModuleList(
            [HeteroInteractionLayer(hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout), nn.LeakyReLU(), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout), nn.LeakyReLU(), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x = self.embed(data.x)
        for layer in self.layers:
            x = layer(x, data.pos, data.edge_index_intra, data.edge_index_inter)
        x = global_add_pool(x, data.batch)
        return self.head(x).view(-1)
