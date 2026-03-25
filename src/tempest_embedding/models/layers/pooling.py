import torch
import torch.nn as nn


class SetPooler(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, linear_out=False):
        super(SetPooler, self).__init__()

        self.linear_out = linear_out
        self.dropout = nn.Dropout(dropout)

        if linear_out:
            self.out_proj = nn.Linear(in_dim, out_dim)
        else:
            self.out_proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                self.dropout
            )

    def forward(self, x, agg='mean'):
        """
        x: (batch, n_walks, dim)
        """

        if self.linear_out:
            return self.out_proj(x)

        if agg == 'sum':
            pooled = x.sum(dim=1)
        elif agg == 'mean':
            pooled = x.mean(dim=1)
        else:
            raise ValueError(f"Unsupported aggregation: {agg}")

        return self.out_proj(pooled)
