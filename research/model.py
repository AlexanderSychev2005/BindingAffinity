import math
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (Seq_len, 1)
        # Compute the positional encodings once in log space.
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the cos to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, Seq_len, d_model) batch dimension
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LigandGNN(nn.Module):  # GCN CONV
    def __init__(self, input_dim, hidden_channels, dropout):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.dropout(x)

        # Averaging nodes and got the molecula vector
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        return x


# class LigandGNN(nn.Module):
#     def __init__(self, input_dim, hidden_channels, heads=4, dropout=0.2):
#         super().__init__()
#         # Heads=4 means we use 4 attention heads
#         # Concat=False, we average the heads instead of concatenating them, to keep the output dimension same as hidden_channels
#         self.conv1 = GATConv(input_dim, hidden_channels, heads=heads, concat=False)
#         self.conv2 = GATConv(
#             hidden_channels, hidden_channels, heads=heads, concat=False
#         )
#         self.conv3 = GATConv(
#             hidden_channels, hidden_channels, heads=heads, concat=False
#         )
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, edge_index, batch):
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = self.dropout(x)
#
#         x = self.conv2(x, edge_index)
#         x = x.relu()
#         x = self.dropout(x)
#
#         x = self.conv3(x, edge_index)
#
#         # Global Mean Pooling
#         x = global_mean_pool(x, batch)
#         return x


class ProteinTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, N=2, h=4, output_dim=128, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=h, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N)

        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_len]
        padding_mask = x == 0  # mask for PAD tokens
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        mask = (~padding_mask).float().unsqueeze(-1)
        x = x * mask

        sum_x = x.sum(dim=1)  # Global average pooling
        token_counts = mask.sum(dim=1).clamp(min=1e-9)
        x = sum_x / token_counts
        x = self.fc(x)
        return x


class BindingAffinityModel(nn.Module):
    def __init__(
        self, num_node_features, hidden_channels=128, gat_heads=4, dropout=0.2
    ):
        super().__init__()
        # Tower 1 - Ligand GNN
        self.ligand_gnn = LigandGNN(
            input_dim=num_node_features,
            hidden_channels=hidden_channels,
            # heads=gat_heads,
            dropout=dropout,
        )
        # Tower 2 - Protein Transformer
        self.protein_transformer = ProteinTransformer(
            vocab_size=26,
            d_model=hidden_channels,
            output_dim=hidden_channels,
            dropout=dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x, edge_index, batch, protein_seq):
        ligand_vec = self.ligand_gnn(x, edge_index, batch)
        batch_size = batch.max().item() + 1
        protein_seq = protein_seq.view(batch_size, -1)

        protein_vec = self.protein_transformer(protein_seq)
        combined = torch.cat([ligand_vec, protein_vec], dim=1)
        return self.head(combined)
