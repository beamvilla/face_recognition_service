import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        p_dist, n_dist = input
        return torch.mean(torch.max(p_dist - n_dist + self.alpha, torch.zeros_like(p_dist)), 0)