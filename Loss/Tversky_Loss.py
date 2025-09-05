import torch
from torch import nn


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W], softmax applied
        # targets: one-hot [B, C, H, W]
        dims = (0, 2, 3)
        TP = torch.sum(inputs * targets, dims)
        FP = torch.sum(inputs * (1 - targets), dims)
        FN = torch.sum((1 - inputs) * targets, dims)

        tversky = (TP + 1e-6) / (TP + self.alpha * FP + self.beta * FN + 1e-6)
        loss = torch.pow((1 - tversky), self.gamma).mean()
        return loss