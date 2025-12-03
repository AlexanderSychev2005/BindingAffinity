import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from rdkit import Chem

from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

class GNNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.lin = nn.Linear(hidden_channels, output_dim) # classification task 0 or 1

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # Averaging nodes and got the molecula vector
        x = global_mean_pool(x, batch) # [batch_size, hidden_channels]

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x