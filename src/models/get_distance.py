import torch
import torch.nn.functional as F


def get_features_distance(
    feature_1: torch.Tensor,
    feature_2: torch.Tensor
) -> torch.Tensor:
    return F.pairwise_distance(feature_1, feature_2, p=2.0, keepdim=True)