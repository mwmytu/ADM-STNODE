import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Whether use adjoint method or not.
adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

class AdaptiveODEFunc(nn.Module):
    def __init__(self, feature_dim, temporal_dim, adj, max_num_layers):
        super(AdaptiveODEFunc, self).__init__()
        self.adj = adj
        self.x0 = None
        self.max_num_layers = max_num_layers
        self.ode_layers = nn.ModuleList([ODEFunc(feature_dim, temporal_dim, adj) for _ in range(max_num_layers)])
        self.decision_network = nn.Sequential(
            nn.Linear(feature_dim, max_num_layers),
            nn.Softmax(dim=1)
        )

    def forward(self, t, x):
        layer_scores = self.decision_network(x)

        layer_scores = layer_scores.squeeze(0) if layer_scores.dim() > 1 else layer_scores.squeeze()
        layer_scores = layer_scores.view(-1)

        num_layers = torch.multinomial(layer_scores, 1) + 1

        num_layers = min(num_layers, self.max_num_layers)

        x_output = x.clone().detach()
        for i in range(num_layers):
            x_output = self.ode_layers[i](t, x_output)

        self.x0 = self.x0 if self.x0 is not None else x.mean()

        f = x_output + self.x0
        return f


class ODEFunc(nn.Module):
    def __init__(self, feature_dim, temporal_dim, adj):
        super(ODEFunc, self).__init__()
        self.adj = adj
        self.x0 = None
        self.alpha = nn.Parameter(0.8 * torch.ones(adj.shape[1]))
        self.beta = 0.6
        self.w = nn.Parameter(torch.eye(feature_dim))
        self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        self.w2 = nn.Parameter(torch.eye(temporal_dim))
        self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)

    def forward(self, t, x):
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).clone()
        xa = torch.einsum('ij, kjlm->kilm', self.adj.clone(), x.clone())

        # ensure the eigenvalues to be less than 1
        d = torch.clamp(self.d.clone(), min=0, max=1)
        w = torch.mm(self.w.clone() * d, torch.t(self.w.clone()))
        xw = torch.einsum('ijkl, lm->ijkm', x.clone(), w)

        d2 = torch.clamp(self.d2.clone(), min=0, max=1)
        w2 = torch.mm(self.w2.clone() * d2, torch.t(self.w2.clone()))
        xw2 = torch.einsum('ijkl, km->ijml', x.clone(), w2)

        self.x0 = self.x0 if self.x0 is not None else x.mean()

        f = alpha / 2 * xa - x + xw - x + xw2 - x + self.x0
        return f





class ODEblock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0,1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x):
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t, method='euler')[1]
        return z


class ODEG(nn.Module):
    def __init__(self, feature_dim, temporal_dim, adj, time, max_num_layers):
        super(ODEG, self).__init__()
        self.odeblock = ODEblock(AdaptiveODEFunc(feature_dim, temporal_dim, adj, max_num_layers), t=torch.tensor([0, time]))

    def forward(self, x):
        self.odeblock.set_x0(x)
        z = self.odeblock(x)
        return F.relu(z)