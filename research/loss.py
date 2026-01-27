import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        squared_errors = (prediction - target) ** 2
        weights = torch.ones_like(target)

        weights[target >= 6.0] = 2.0  # Fine x2 pKd > 6 good binding
        weights[target >= 7.0] = 5.0  # Fine x5 pKd > 7 great binding
        weights[target >= 8.0] = 10.0  # Fine x10 pKd > 8 super binding

        weighted_loss = squared_errors * weights
        return torch.mean(weighted_loss)
