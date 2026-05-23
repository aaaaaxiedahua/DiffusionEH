import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_relation_line_graph(triples, n_ent, n_rel, topk=10, threshold=0.0, include_inverse=True):
    """Build a relation transition graph from two-hop paths in fact triples.

    The returned edge index uses the same relation id space as DiffusionE:
    original relations [0, n_rel), inverse relations [n_rel, 2*n_rel),
    and the identity relation id 2*n_rel.
    """
    rel_count = 2 * n_rel + 1
    adjacency = [[] for _ in range(n_ent)]
    for h, r, t in triples:
        adjacency[h].append((r, t))
        if include_inverse:
            adjacency[t].append((r + n_rel, h))

    counts = {}
    for mid_edges in adjacency:
        if len(mid_edges) <= 1:
            continue
        in_rels = [r for r, _ in mid_edges]
        out_rels = in_rels
        for r_i in in_rels:
            for r_j in out_rels:
                counts[(r_i, r_j)] = counts.get((r_i, r_j), 0) + 1

    rows, cols, weights = [], [], []
    by_source = {}
    for (src, dst), count in counts.items():
        by_source.setdefault(src, []).append((dst, count))

    for src, dst_counts in by_source.items():
        total = float(sum(count for _, count in dst_counts))
        scored = [(dst, count / total) for dst, count in dst_counts]
        scored = [(dst, score) for dst, score in scored if score >= threshold]
        scored.sort(key=lambda item: item[1], reverse=True)
        if topk is not None and topk > 0:
            scored = scored[:topk]
        for dst, score in scored:
            rows.append(src)
            cols.append(dst)
            weights.append(score)

    for rel in range(rel_count):
        rows.append(rel)
        cols.append(rel)
        weights.append(1.0)

    edge_index = torch.LongTensor([rows, cols])
    edge_weight = torch.FloatTensor(weights)
    return edge_index, edge_weight


class RelationCoDiffusion(nn.Module):
    def __init__(
        self,
        hidden_dim,
        attn_dim,
        n_rel,
        edge_index,
        edge_weight,
        tau=1.0,
        residual_alpha=0.5,
        dropout=0.1,
        layers_per_gnn=1,
        act=lambda x: x,
    ):
        super(RelationCoDiffusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.n_rel = n_rel
        self.rel_count = 2 * n_rel + 1
        self.tau = tau
        self.residual_alpha = residual_alpha
        self.layers_per_gnn = layers_per_gnn
        self.act = act

        self.register_buffer("rel_edge_index", edge_index)
        self.register_buffer("rel_edge_weight", edge_weight)

        self.W_src = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.W_dst = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.W_q = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.W_w = nn.Linear(1, attn_dim, bias=False)
        self.w_alpha = nn.Linear(attn_dim, 1, bias=False)
        self.W_msg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, rel_state, query_rel_embed):
        if rel_state.dim() == 2:
            rel_state = rel_state.unsqueeze(0).expand(query_rel_embed.size(0), -1, -1)
        state = rel_state
        base_state = rel_state
        src = self.rel_edge_index[0]
        dst = self.rel_edge_index[1]
        edge_log_weight = torch.log(self.rel_edge_weight.clamp_min(1e-8)).unsqueeze(-1)

        for _ in range(self.layers_per_gnn):
            hs = state[:, src, :]
            ht = state[:, dst, :]
            hq = query_rel_embed.unsqueeze(1).expand_as(hs)
            edge_weight_feat = edge_log_weight.unsqueeze(0).expand(hs.size(0), -1, -1)
            alpha_logits = self.w_alpha(
                torch.relu(self.W_src(hs) + self.W_dst(ht) + self.W_q(hq) + self.W_w(edge_weight_feat))
            ).squeeze(-1)
            alpha = self._edge_softmax(src, alpha_logits / self.tau)
            messages = alpha.unsqueeze(-1) * self.W_msg(ht)
            agg = torch.zeros_like(state)
            agg.index_add_(1, src, messages)
            diffused = self.act(agg)
            diffused = self.dropout(diffused)
            state = self.residual_alpha * base_state + (1.0 - self.residual_alpha) * diffused

        return state

    def _edge_softmax(self, src, logits):
        batch_size = logits.size(0)
        stable_logits = logits - logits.max(dim=1, keepdim=True)[0]
        exp_logits = torch.exp(stable_logits)
        denom = torch.zeros((batch_size, self.rel_count), device=logits.device)
        denom.index_add_(1, src, exp_logits)
        return exp_logits / denom[:, src].clamp_min(1e-12)
