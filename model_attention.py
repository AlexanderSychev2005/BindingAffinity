import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super().__init__()
        # Main attention layer
        # Feature dim is the dimension of the hidden features
        self.attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Normalization layer for stabilizing training
        self.norm = nn.LayerNorm(feature_dim)

        # Feedforward network for further processing, classical transformer style
        self.ff = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(), # GELU works better with transformers
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.norm_ff = nn.LayerNorm(feature_dim)
        self.last_attention_weights = None

    def forward(self, ligand_features, protein_features, key_padding_mask=None):
        # ligand_features: [Batch, Atoms, Dim] - atoms
        # protein_features: [Batch, Residues, Dim] - amino acids
        # Cross attention:
        # Query = Ligand (What we want to find out)
        # Key, Value = Protein (Where we look for information)
        # Result: "Ligand enriched with knowledge about proteins"
        attention_output, attn_weights = self.attention(
            query=ligand_features,
            key=protein_features,
            value=protein_features,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        self.last_attention_weights = attn_weights.detach().cpu()

        # Residual connection (x + attention(x)) and normalization
        ligand_features = self.norm(ligand_features + attention_output)

        # Feedforward network with residual connection and normalization
        ff_output = self.ff(ligand_features)
        ligand_features = self.norm_ff(ligand_features + ff_output)

        return ligand_features


class BindingAffinityModel(nn.Module):
    def __init__(
        self, num_node_features, hidden_channels=256, gat_heads=2, dropout=0.3
    ):
        super().__init__()
        self.dropout = dropout
        self.hidden_channels = hidden_channels

        # Tower 1 - Ligand GNN with GAT layers, using 3 GAT layers, so that every atom can "see" up to 3 bonds away,
        # Attention allows to measure the importance of the neighbours
        self.gat1 = GATConv(
            num_node_features, hidden_channels, heads=gat_heads, concat=False
        )
        self.gat2 = GATConv(
            hidden_channels, hidden_channels, heads=gat_heads, concat=False
        )
        self.gat3 = GATConv(
            hidden_channels, hidden_channels, heads=gat_heads, concat=False
        )

        # Tower 2 - Protein Transformer, 22 = 21 amino acids + 1 padding token PAD
        self.protein_embedding = nn.Embedding(22, hidden_channels)
        # Additional positional encoding (simple linear) to give the model information about the order
        self.prot_conv = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )

        # Cross-Attention Layer, atoms attending to amino acids
        self.cross_attention = CrossAttentionLayer(
            feature_dim=hidden_channels, num_heads=4, dropout=dropout
        )

        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)  # Final output for regression, pKd

    def forward(self, x, edge_index, batch, protein_seq):
        # Ligand GNN forward pass (Graph -> Node Embeddings)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.gat2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.gat3(x, edge_index))  # [Total_Atoms, Hidden_Channels]

        # Convert graph into tensor [Batch, Max_Atoms, Hidden_Channels]
        # to_dense_batch adds zeros paddings where necessary to the size of the largest graph in the batch
        ligand_dense, ligand_mask = to_dense_batch(x, batch)
        # ligand_dense: [Batch, Max_Atoms, Hidden_Channels]
        # ligand_mask: [Batch, Max_Atoms] True where there is real atom, False where there is padding

        batch_size = ligand_dense.size(0)
        protein_seq = protein_seq.view(batch_size, -1)  # [Batch, Seq_Len]

        # Protein forward pass protein_seq: [Batch, Seq_Len]
        p = self.protein_embedding(protein_seq)  # [Batch, Seq_Len, Hidden_Channels]

        # A simple convolution to understand local context in amino acids
        p = p.permute(0, 2, 1)  # Change to [Batch, Hidden_Channels, Seq_Len] for Conv1d
        p = F.relu(self.prot_conv(p))
        p = p.permute(0, 2, 1)  # [Batch, Seq, Hidden_Channels]

        # Mask for protein (where PAD=0, True, but MHA needs True where IGNOREME)
        # In Pytorch MHA, the key_padding_mask should be True where we want to ignore
        protein_pad_mask = protein_seq == 0

        # Cross-Attention
        x_cross = self.cross_attention(
            ligand_dense, p, key_padding_mask=protein_pad_mask
        )

        # Pooling over atoms to get a single vector per molecule, considering only real atoms, ignoring paddings
        # ligand mask True where real atom, False where padding
        mask_expanded = ligand_mask.unsqueeze(-1)  # [Batch, Max_Atoms, 1]

        # Zero out the padded atom features
        x_cross = x_cross * mask_expanded

        # Sum the features of real atoms / number of real atoms to get the mean
        sum_features = torch.sum(x_cross, dim=1)  # [Batch, Hidden_Channels]
        num_atoms = torch.sum(mask_expanded, dim=1)  # [Batch, 1]
        pooled_x = sum_features / (num_atoms + 1e-6)  # Avoid division by zero

        # MLP Head
        out = F.relu(self.fc1(pooled_x))
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.fc2(out)
        return out
