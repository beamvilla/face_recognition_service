from typing import Tuple, List
import torch
import torch.nn as nn

from models.get_distance import get_features_distance


class TripletNet(nn.Module):
    def __init__(self, feature_extractor_module: nn.Module, device: torch.device):
        super().__init__()
        self.feature_extractor = feature_extractor_module.to(device)
        self.device = device

    def get_distance(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        img1, img2 = img1.to(self.device), img2.to(self.device)
        img1_feat, img2_feat = self.feature_extractor(img1), self.feature_extractor(img2)
        return get_features_distance(img1_feat, img2_feat)

    def forward(self, input: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        anchor, pos, neg = input
        p_dist = self.get_distance(anchor, pos)
        n_dist = self.get_distance(anchor, neg)
        return (p_dist, n_dist)