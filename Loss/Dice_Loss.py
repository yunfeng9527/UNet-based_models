import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    # labels: [B, H, W] → [B, C, H, W]
    return F.one_hot(labels, num_classes).permute(0, 3, 1, 2).float()

class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, epsilon=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == self.num_classes
        inputs = torch.softmax(inputs, dim=1)
        targets_one_hot = make_one_hot(targets, self.num_classes)

        total_loss = 0.0
        valid_classes = 0

        for c in range(self.num_classes):
            pred = inputs[:, c]  # [B, H, W]
            gt = targets_one_hot[:, c]  # [B, H, W]

            if gt.sum() == 0:
                continue  # 忽略未出现的类

            intersection = 2 * (pred * gt).sum()
            denominator = pred.sum() + gt.sum()
            dice = (intersection + self.epsilon) / (denominator + self.epsilon)
            total_loss += 1 - dice
            valid_classes += 1

        if valid_classes == 0:
            return torch.tensor(0.0, device=inputs.device)
        return total_loss