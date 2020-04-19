import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizeScale(nn.Module):

    def __init__(self, dim, init_norm=20):
        super(NormalizeScale, self).__init__()
        self.init_norm = init_norm
        self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

    def forward(self, bottom):
        bottom_normalized = F.normalize(bottom, p=2, dim=2)
        bottom_normalized_scaled = bottom_normalized * self.weight
        return bottom_normalized_scaled