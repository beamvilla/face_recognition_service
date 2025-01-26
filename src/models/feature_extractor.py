import torch
import torchvision.models as models
import torch.nn as nn
from typing import Optional, List


def get_fc_layers(fc_sizes: List[int], ps: List[float]):
    fc_layers_list = []
    for ni,nf,p in zip(fc_sizes[:-1], fc_sizes[1:], ps):
        fc_layers_list.append(nn.Linear(ni, nf))
        fc_layers_list.append(nn.ReLU(inplace=True))
        fc_layers_list.append(nn.BatchNorm1d(nf))
        fc_layers_list.append(nn.Dropout(p=p))
    return nn.Sequential(*fc_layers_list)


class Resnet34FeatureExtractor(nn.Module):
    def __init__(
        self, 
        n_chanel: int = 3, 
        feat_dim: int = 128, 
        weights: Optional[models.ResNet34_Weights] = None
    ):
        super().__init__()
        assert n_chanel in [1, 3]  # Validate input channel

        self.feat_dim = feat_dim
        resnet34 = models.resnet34(weights=weights)
        # Change input channel according to the input data
        resnet34.conv1 = nn.Conv2d(n_chanel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.body = nn.Sequential(*nn.ModuleList(resnet34.children())[:-1])
        self.dense = get_fc_layers(fc_sizes=[512, 1024, 512, feat_dim], ps=[.5,.5,.5])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.body(input)
        output = torch.flatten(output, 1)
        output = self.dense(output)
        return output
    