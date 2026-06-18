import math

import torch
import torch.nn as nn
from torch_scatter import scatter


class PhaseInterferenceModule(nn.Module):
    """Query-aware phase interference branch for edge message aggregation."""

    def __init__(self, hidden_dim, attn_dim, tau=1.0, act=lambda x: x):
        super(PhaseInterferenceModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.tau = tau
        self.act = act

        self.phase_proj = nn.Linear(hidden_dim * 3, attn_dim)
        self.phase_score = nn.Linear(attn_dim, 1)
        self.W_re = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_im = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_amp = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, hs, hr, h_qr, weighted_message, obj, n_node):
        phase_input = torch.cat([hs, hr, h_qr], dim=-1)
        theta_logits = self.phase_score(torch.relu(self.phase_proj(phase_input))).squeeze(-1)
        theta = math.pi * torch.tanh(theta_logits / self.tau)

        cos_theta = torch.cos(theta).unsqueeze(-1)
        sin_theta = torch.sin(theta).unsqueeze(-1)

        real_message = weighted_message * cos_theta
        imag_message = weighted_message * sin_theta

        real_agg = scatter(real_message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        imag_agg = scatter(imag_message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        amp_agg = torch.sqrt(real_agg.pow(2) + imag_agg.pow(2) + 1e-12)

        return self.act(
            self.W_re(real_agg) + self.W_im(imag_agg) + self.W_amp(amp_agg)
        )
