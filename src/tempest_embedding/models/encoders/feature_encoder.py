import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.lin_xr = nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_xz = nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_xn = nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        r = torch.sigmoid(self.lin_xr(x) + self.lin_hr(h))
        z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h))
        g = torch.tanh(self.lin_xn(x) + self.lin_hn(r * h))
        return z * h + (1 - z) * g


class GRUODECell(nn.Module):
    def __init__(self, hidden_size, bias=True):
        super().__init__()
        self.lin_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, t, h):
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        g = torch.tanh(x + self.lin_hn(r * h))
        return (1 - z) * (g - h)


class FeatureEncoder(nn.Module):
    start_time = 0.0
    end_time = 1.0

    def __init__(self, in_features, hidden_features, dropout_p=0.1, solver='rk4', step_size=0.125):
        super().__init__()
        self.hidden_dim = hidden_features
        if self.hidden_dim == 0:
            return
        self.gru = GRUCell(in_features, hidden_features)
        self.odefun = GRUODECell(hidden_features)
        self.dropout = nn.Dropout(dropout_p)
        self.solver = solver
        if self.solver in {'euler', 'rk4'}:
            self.step_size = step_size

    def integrate(self, t_records, X, mask=None):
        batch, n_walk, len_walk, feat_dim = X.shape
        X = X.view(batch * n_walk, len_walk, feat_dim)
        t_records = t_records.view(batch * n_walk, len_walk, 1)
        h = torch.zeros(batch * n_walk, self.hidden_dim).type_as(X)
        for i in range(X.shape[1] - 1):
            h = self.gru(X[:, i, :], h)
            t0 = t_records[:, i + 1, :]
            t1 = t_records[:, i, :]
            delta_t = torch.log10(torch.abs(t1 - t0) + 1.0) + 0.01
            state = (torch.zeros_like(t0), delta_t, h)
            ts = torch.tensor([self.start_time, self.end_time]).type_as(X)
            if self.solver in {'euler', 'rk4'}:
                solution = odeint(self, state, ts, method=self.solver, options=dict(step_size=self.step_size))
            elif self.solver == 'dopri5':
                solution = odeint(self, state, ts, method=self.solver)
            else:
                raise NotImplementedError(f'{self.solver} solver is not implemented.')
            _, _, h = tuple(s[-1] for s in solution)
        encoded_features = self.gru(X[:, -1, :], h)
        encoded_features = encoded_features.view(batch, n_walk, self.hidden_dim)
        return self.dropout(encoded_features)

    def forward(self, s, state):
        t0, t1, x = state
        ratio = (t1 - t0) / (self.end_time - self.start_time)
        t = (s - self.start_time) * ratio + t0
        dx = self.odefun(t, x)
        dx = dx * ratio
        return torch.zeros_like(t0), torch.zeros_like(t1), dx
