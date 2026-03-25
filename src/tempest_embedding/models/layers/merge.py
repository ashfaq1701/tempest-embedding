import torch
import torch.nn as nn


class MergeLayer(nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, non_linear=True):
        super().__init__()
        self.fc1 = nn.Linear(dim1 + dim2, dim3)
        self.fc2 = nn.Linear(dim3, dim4)
        self.act = nn.ReLU()
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        self.non_linear = non_linear
        if not non_linear:
            assert dim1 == dim2
            self.fc = nn.Linear(dim1, 1)
            torch.nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x1, x2):
        z_walk = None
        if self.non_linear:
            x = torch.cat([x1, x2], dim=-1)
            h = self.act(self.fc1(x))
            z = self.fc2(h)
        else:
            x = torch.cat([x1, x2], dim=-2)
            z_walk = self.fc(x).squeeze(-1)
            z = z_walk.sum(dim=-1, keepdim=True)
        return z, z_walk
