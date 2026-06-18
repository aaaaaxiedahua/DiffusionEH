import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from   torch_scatter import scatter
from   collections import defaultdict
from   relation_codiffusion import RelationCoDiffusion
from   phase_interference import PhaseInterferenceModule

class GNNLayer(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        attn_dim,
        n_rel,
        n_ent,
        n_node_topk=-1,
        n_edge_topk=-1,
        tau=1.0,
        act=lambda x:x,
        use_phase_interference=False,
        phase_tau=1.0,
        phase_weight=0.3,
    ):
        super(GNNLayer, self).__init__()
        self.n_rel       = n_rel
        self.n_ent       = n_ent
        self.in_dim      = in_dim
        self.out_dim     = out_dim
        self.attn_dim    = attn_dim
        self.act         = act
        self.n_node_topk = n_node_topk
        self.n_edge_topk = n_edge_topk
        self.tau         = tau
        self.use_phase_interference = use_phase_interference
        self.phase_weight = phase_weight
        self.rela_embed  = nn.Embedding(2*n_rel+1, in_dim)
        self.Ws_attn     = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn     = nn.Linear(in_dim, attn_dim, bias=False)
        self.Ws1_attn     = nn.Linear(in_dim, in_dim, bias=False)
        self.Wr1_attn     = nn.Linear(in_dim, in_dim, bias=False)
        
        self.Wqr_attn    = nn.Linear(in_dim, attn_dim)
        self.w_alpha     = nn.Linear(attn_dim, 1)
        self.W_h         = nn.Linear(in_dim, out_dim, bias=False)
        self.W_samp      = nn.Linear(in_dim, 1, bias=False)
        if self.use_phase_interference:
            self.phase_module = PhaseInterferenceModule(
                in_dim,
                attn_dim,
                tau=phase_tau,
                act=act,
            )
        else:
            self.phase_module = None
        
    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        if self.training and self.tau > 0: 
            self.softmax = lambda x : F.gumbel_softmax(x, tau=self.tau, hard=False)
        else:
            self.softmax = lambda x : F.softmax(x, dim=1)
        for module in self.children():
            module.train(mode)
        return self

    def forward(self, q_sub, q_rel, hidden, edges, nodes, old_nodes_new_idx, batchsize, rel_state=None, rel_diff_weight=1.0):
        sub     = edges[:,4]
        rel     = edges[:,2]
        obj     = edges[:,5]
        hs      = hidden[sub]
        base_hr = self.rela_embed(rel)
        r_idx   = edges[:,0]
        if rel_state is None:
            hr = base_hr
        else:
            dynamic_hr = rel_state[r_idx, rel]
            hr = (1.0 - rel_diff_weight) * base_hr + rel_diff_weight * dynamic_hr
        h_qr    = self.rela_embed(q_rel)[r_idx]
        n_node  = nodes.shape[0]
        message = nn.ReLU()(self.Ws1_attn(hs+hr) - self.Wr1_attn(hs+h_qr))
        if self.n_edge_topk > 0:
            alpha          = self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)-  self.Ws1_attn(sub) -self.Wr1_attn(rel) )).squeeze(-1)
            edge_prob      = F.gumbel_softmax(alpha, tau=1, hard=False)
            topk_index     = torch.argsort(edge_prob, descending=True)[:self.n_edge_topk]
            edge_prob_hard = torch.zeros((alpha.shape[0])).cuda()
            edge_prob_hard[topk_index] = 1
            alpha *= (edge_prob_hard - edge_prob.detach() + edge_prob)
            alpha = torch.sigmoid(alpha).unsqueeze(-1)
            
        else:
            alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)))) # 
        weighted_message = alpha * message
        message_agg = scatter(weighted_message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        hidden_real = self.act(self.W_h(message_agg))
        if self.phase_module is not None:
            hidden_phase = self.phase_module(hs, hr, h_qr, weighted_message, obj, n_node)
            hidden_new = hidden_real + self.phase_weight * hidden_phase
        else:
            hidden_new = hidden_real
        hidden_new  = hidden_new.clone()
        
        if self.n_node_topk <= 0:
            return hidden_new

        tmp_diff_node_idx = torch.ones(n_node)
        tmp_diff_node_idx[old_nodes_new_idx] = 0
        bool_diff_node_idx = tmp_diff_node_idx.bool()
        diff_node = nodes[bool_diff_node_idx]

        diff_node_logit  = self.W_samp(hidden_new[bool_diff_node_idx]).squeeze(-1) 
        
        node_scores = torch.ones((batchsize, self.n_ent)).cuda() * float('-inf')
        node_scores[diff_node[:,0], diff_node[:,1]] = diff_node_logit

        node_scores = self.softmax(node_scores) 
        topk_index = torch.topk(node_scores, self.n_node_topk, dim=1).indices.reshape(-1)
        topk_batchidx  = torch.arange(batchsize).repeat(self.n_node_topk,1).T.reshape(-1)
        batch_topk_nodes = torch.zeros((batchsize, self.n_ent)).cuda()
        batch_topk_nodes[topk_batchidx, topk_index] = 1

        bool_sampled_diff_nodes_idx = batch_topk_nodes[diff_node[:,0], diff_node[:,1]].bool()
        bool_same_node_idx = ~bool_diff_node_idx.cuda()
        bool_same_node_idx[bool_diff_node_idx] = bool_sampled_diff_nodes_idx

        diff_node_prob_hard = batch_topk_nodes[diff_node[:,0], diff_node[:,1]]
        diff_node_prob = node_scores[diff_node[:,0], diff_node[:,1]]
        hidden_new[bool_diff_node_idx] *= (diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob).unsqueeze(-1)
        
        new_nodes  = nodes[bool_same_node_idx]
        hidden_new = hidden_new[bool_same_node_idx]

        return hidden_new, new_nodes, bool_same_node_idx

class GNNModel(torch.nn.Module):
    def __init__(self, params, loader):
        super(GNNModel, self).__init__()
        self.n_layer     = params.n_layer
        self.hidden_dim  = params.hidden_dim
        self.attn_dim    = params.attn_dim
        self.n_ent       = params.n_ent
        self.n_rel       = params.n_rel
        self.n_node_topk = params.n_node_topk
        self.n_edge_topk = params.n_edge_topk
        self.loader      = loader
        self.use_rel_codiffusion = getattr(params, 'use_rel_codiffusion', False)
        self.rel_diff_weight = getattr(params, 'rel_diff_weight', 0.5)
        self.use_phase_interference = getattr(params, 'use_phase_interference', False)
        self.phase_tau = getattr(params, 'phase_tau', 1.0)
        self.phase_weight = getattr(params, 'phase_weight', 0.3)
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act  = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            i_n_node_topk = self.n_node_topk if 'int' in str(type(self.n_node_topk)) else self.n_node_topk[i]
            self.gnn_layers.append(
                GNNLayer(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.attn_dim,
                    self.n_rel,
                    self.n_ent,
                    n_node_topk=i_n_node_topk,
                    n_edge_topk=self.n_edge_topk,
                    tau=params.tau,
                    act=act,
                    use_phase_interference=self.use_phase_interference,
                    phase_tau=self.phase_tau,
                    phase_weight=self.phase_weight,
                )
            )

        self.gnn_layers = nn.ModuleList(self.gnn_layers)       
        if self.use_rel_codiffusion:
            self.rel_codiffusion = RelationCoDiffusion(
                self.hidden_dim,
                self.attn_dim,
                self.n_rel,
                loader.rel_edge_index,
                loader.rel_edge_weight,
                tau=params.rel_tau,
                residual_alpha=params.rel_residual_alpha,
                dropout=params.rel_dropout,
                layers_per_gnn=params.rel_layers_per_gnn,
                act=act,
            )
        else:
            self.rel_codiffusion = None
        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        self.gate    = nn.GRU(self.hidden_dim, self.hidden_dim)

    def updateTopkNums(self, topk_list):
        assert len(topk_list) == self.n_layer
        for idx in range(self.n_layer):
            self.gnn_layers[idx].n_node_topk = topk_list[idx]

    def fixSamplingWeight(self):
        def freeze(m):
            m.requires_grad=False
        for i in range(self.n_layer):
            self.gnn_layers[i].W_samp.apply(freeze)

    def forward(self, subs, rels, mode='train'):
        n      = len(subs)                                                                
        q_sub  = torch.LongTensor(subs).cuda()                                           
        q_rel  = torch.LongTensor(rels).cuda()                                           
        h0     = torch.zeros((1, n, self.hidden_dim)).cuda()                              
        nodes  = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)  
        hidden = torch.zeros(n, self.hidden_dim).cuda()                                  
    
        for i in range(self.n_layer):
            rel_state = None
            if self.rel_codiffusion is not None:
                query_rel_embed = self.gnn_layers[i].rela_embed(q_rel)
                rel_state = self.gnn_layers[i].rela_embed.weight
                rel_state = self.rel_codiffusion(rel_state, query_rel_embed)
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), n, mode=mode)
            n_node  = nodes.size(0)
            hidden, nodes, sampled_nodes_idx = self.gnn_layers[i](
                q_sub,
                q_rel,
                hidden,
                edges,
                nodes,
                old_nodes_new_idx,
                n,
                rel_state=rel_state,
                rel_diff_weight=self.rel_diff_weight,
            )
            
            h0          = torch.zeros(1, n_node, hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            h0          = h0[0, sampled_nodes_idx, :].unsqueeze(0)
            
            hidden      = self.dropout(hidden)
            hidden, h0  = self.gate(hidden.unsqueeze(0), h0)
            hidden      = hidden.squeeze(0)
        scores     = self.W_final(hidden).squeeze(-1)                   
        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()         
        scores_all[[nodes[:,0], nodes[:,1]]] = scores                   
        
        return scores_all
