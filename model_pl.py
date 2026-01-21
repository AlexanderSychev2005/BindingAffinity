from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import Adam

from model import LigandGNN, ProteinTransformer

class BindingAffinityModelPL(pl.LightningModule):
    def __init__(self, num_node_features, hidden_channels_gnn, lr):
        super().__init__()
        self.save_hyperparameters() # Save hyperparameters for easy access
        self.lr = lr

        self.ligand_gnn = LigandGNN(input_dim=num_node_features, hidden_channels=hidden_channels_gnn)
        self.protein_transformer = ProteinTransformer(vocab_size=26)
        self.head = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        self.criterion = nn.MSELoss()

    def forward(self, x, edge_index, batch, protein_seq):
        ligand_vec = self.ligand_gnn(x, edge_index, batch)
        batch_size = batch.max().item() + 1
        protein_seq = protein_seq.view(batch_size, -1)

        protein_vec = self.protein_transformer(protein_seq)
        combined = torch.cat([ligand_vec, protein_vec], dim=1)
        return self.head(combined)

    def training_step(self, batch, batch_idx):
        # We don't need .to(device), zero_grad, backward, PL handles that
        out = self(batch.x, batch.edge_index, batch.batch, batch.protein_seq)
        loss = self.criterion(out.squeeze(), batch.y.squeeze())

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index, batch.batch, batch.protein_seq)
        loss = self.criterion(out.squeeze(), batch.y.squeeze())

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
