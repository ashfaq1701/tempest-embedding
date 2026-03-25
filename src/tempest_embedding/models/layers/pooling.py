import torch.nn as nn


class SetPooler(nn.Module):
    def __init__(self, n_features, out_features, dropout_p=0.1, walk_linear_out=False):
        super().__init__()
        self.mean_proj = nn.Linear(n_features, n_features)
        self.max_proj = nn.Linear(n_features, n_features)
        self.attn_weight_mat = nn.Parameter(nn.init.xavier_uniform_(nn.Parameter(nn.init.xavier_uniform_(nn.Parameter.__new__(nn.Parameter,)))) if False else None)
        self.attn_weight_mat = nn.Parameter(nn.init.xavier_uniform_(nn.Parameter.__new__(nn.Parameter,)) if False else None)
        # Explicit parameter construction for parity with original implementation.
        import torch
        self.attn_weight_mat = nn.Parameter(torch.zeros((2, n_features, n_features)), requires_grad=True)
        nn.init.xavier_uniform_(self.attn_weight_mat.data[0])
        nn.init.xavier_uniform_(self.attn_weight_mat.data[1])
        self.dropout = nn.Dropout(dropout_p)
        self.out_proj = nn.Sequential(nn.Linear(n_features, out_features), nn.ReLU(), self.dropout)
        self.walk_linear_out = walk_linear_out

    def forward(self, X, agg='sum'):
        if self.walk_linear_out:
            return self.out_proj(X)
        if agg == 'sum':
            return self.out_proj(X.sum(dim=-2))
        assert agg == 'mean'
        return self.out_proj(X.mean(dim=-2))
