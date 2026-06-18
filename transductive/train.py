import random
import os
import argparse
import json
from datetime import datetime
from uuid import uuid4
import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel
from utils import *

''' main script of DiffusionE'''
parser = argparse.ArgumentParser(description="Parser for DiffusionE")
parser.add_argument('--data_path', type=str, default='data/family/')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--topk', type=int, default=-1)
parser.add_argument('--layers', type=int, default=-1)
parser.add_argument('--sampling', type=str, default='incremental')
parser.add_argument('--weight', type=str, default=None)
parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--loss_in_each_layer', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--HPO', action='store_true')
parser.add_argument('--eval_with_node_usage', action='store_true')
parser.add_argument('--scheduler', type=str, default='exp')
parser.add_argument('--remove_1hop_edges', action='store_true')
parser.add_argument('--fact_ratio', type=float, default=0.9)
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--eval_interval', type=int, default=1)
parser.add_argument('--use_rel_codiffusion', action='store_true')
parser.add_argument('--rel_line_topk', type=int, default=10)
parser.add_argument('--rel_edge_threshold', type=float, default=0.001)
parser.add_argument('--rel_tau', type=float, default=1.0)
parser.add_argument('--rel_residual_alpha', type=float, default=0.5)
parser.add_argument('--rel_diff_weight', type=float, default=0.5)
parser.add_argument('--rel_dropout', type=float, default=0.1)
parser.add_argument('--rel_layers_per_gnn', type=int, default=1)
parser.add_argument('--rel_include_inverse', action='store_true')
parser.add_argument('--use_phase_interference', action='store_true')
parser.add_argument('--phase_tau', type=float, default=1.0)
parser.add_argument('--phase_weight', type=float, default=0.3)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--decay_rate', type=float, default=None)
parser.add_argument('--lamb', type=float, default=None)
parser.add_argument('--hidden_dim', type=int, default=None)
parser.add_argument('--attn_dim', type=int, default=None)
parser.add_argument('--dropout', type=float, default=None)
parser.add_argument('--act', type=str, default=None, choices=['relu', 'tanh', 'idd'])
parser.add_argument('--n_batch', type=int, default=None)
parser.add_argument('--trial_id', type=int, default=None)
args = parser.parse_args()
CLI_OVERRIDE_NAMES = ['lr', 'decay_rate', 'lamb', 'hidden_dim', 'attn_dim', 'dropout', 'act', 'n_batch']


def build_log_file(dataset, trial_id=None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_uuid = uuid4().hex[:8]
    if trial_id is None:
        filename = f'{dataset}_{timestamp}_{run_uuid}.log'
    else:
        filename = f'{dataset}_trial{trial_id}_{timestamp}_{run_uuid}.log'
    return os.path.join('results', dataset, 'logs', filename)

if __name__ == '__main__':
    opts = args
    cli_overrides = {name: getattr(args, name) for name in CLI_OVERRIDE_NAMES}
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    
    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]
    
    torch.cuda.set_device(opts.gpu)
    print('==> gpu:', opts.gpu)
    loader = DataLoader(opts)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    
    if dataset == 'family':
        opts.lr = 0.0036
        opts.decay_rate = 0.999
        opts.lamb = 0.000017
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.dropout = 0.29
        opts.act = 'relu'
        opts.n_node_topk = [opts.topk] * opts.layers
        opts.n_edge_topk = -1
        opts.n_layer = opts.layers
        opts.n_batch = opts.n_tbatch = 20
        
    elif dataset == 'umls':
        opts.lr = 0.0012
        opts.decay_rate = 0.998
        opts.lamb = 0.00014
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.dropout = 0.01
        opts.act = 'tanh'
        opts.n_node_topk = [opts.topk] * opts.layers
        opts.n_edge_topk = -1
        opts.n_layer = opts.layers
        opts.n_batch = opts.n_tbatch = 10
        
    elif dataset == 'WN18RR':
        opts.lr = 0.0030
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.n_node_topk = [opts.topk] * opts.layers
        opts.n_edge_topk = -1
        opts.n_layer = opts.layers
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = opts.n_tbatch = 50
        
    elif dataset == 'fb15k-237':
        opts.lr = 0.0009
        opts.decay_rate = 0.9938
        opts.lamb = 0.000080
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_node_topk = [opts.topk] * opts.layers
        opts.n_edge_topk = -1
        opts.n_layer = opts.layers
        opts.dropout = 0.0391
        opts.act = 'idd'
        opts.n_batch = opts.n_tbatch = 6 
        
    elif dataset == 'nell':
        opts.lr = 0.0011
        opts.decay_rate = 0.9938
        opts.lamb = 0.000089
        opts.hidden_dim = 128
        opts.attn_dim = 64
        opts.dropout = 0.2593
        opts.act = 'idd'
        opts.n_node_topk = [opts.topk] * opts.layers
        opts.n_edge_topk = -1
        opts.n_layer = opts.layers
        opts.n_batch = opts.n_tbatch = 10
        
    elif dataset == 'YAGO':
        opts.lr = 0.001
        opts.decay_rate = 0.9429713470775948
        opts.lamb = 0.000946516892415447
        opts.hidden_dim = 64
        opts.attn_dim = 2
        opts.dropout = 0.19456805575101324
        opts.act = 'relu'
        opts.n_node_topk = [opts.topk] * opts.layers
        opts.n_edge_topk = -1
        opts.n_layer = opts.layers
        opts.n_batch = opts.n_tbatch = 5

    for name in CLI_OVERRIDE_NAMES:
        value = cli_overrides[name]
        if value is not None:
            if name == 'n_batch':
                opts.n_batch = opts.n_tbatch = value
            else:
                setattr(opts, name, value)
    
    checkPath('./results/')
    checkPath(f'./results/{dataset}/')
    checkPath(f'./results/{dataset}/logs/')
    checkPath(f'{loader.task_dir}/saveModel/')

    model = BaseModel(opts, loader)
    opts.perf_file = build_log_file(dataset, trial_id=opts.trial_id)
    print(f'==> perf_file: {opts.perf_file}')
    
    config = {
        'dataset': dataset,
        'trial_id': opts.trial_id,
        'perf_file': opts.perf_file,
        'lr': opts.lr,
        'decay_rate': opts.decay_rate,
        'lamb': opts.lamb,
        'hidden_dim': opts.hidden_dim,
        'attn_dim': opts.attn_dim,
        'n_layer': opts.n_layer,
        'n_batch': opts.n_batch,
        'dropout': opts.dropout,
        'act': opts.act,
        'topk': opts.topk,
        'fact_ratio': opts.fact_ratio,
        'tau': opts.tau,
        'remove_1hop_edges': opts.remove_1hop_edges,
        'use_rel_codiffusion': opts.use_rel_codiffusion,
        'use_phase_interference': opts.use_phase_interference,
    }
    if opts.use_rel_codiffusion:
        config.update({
            'rel_line_topk': opts.rel_line_topk,
            'rel_edge_threshold': opts.rel_edge_threshold,
            'rel_tau': opts.rel_tau,
            'rel_residual_alpha': opts.rel_residual_alpha,
            'rel_diff_weight': opts.rel_diff_weight,
            'rel_dropout': opts.rel_dropout,
            'rel_layers_per_gnn': opts.rel_layers_per_gnn,
            'rel_include_inverse': opts.rel_include_inverse,
        })
    if opts.use_phase_interference:
        config.update({
            'phase_tau': opts.phase_tau,
            'phase_weight': opts.phase_weight,
        })
    legacy_config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act)
    config_str = '[CONFIG] ' + json.dumps(config, sort_keys=True) + '\n'
    print(legacy_config_str)
    print(config_str)
    with open(opts.perf_file, 'a+') as f:
        f.write(legacy_config_str)
        f.write(config_str)  

    if args.weight != None:
        model.loadModel(args.weight)
        model._update()
        model.model.updateTopkNums(opts.n_node_topk)

    if opts.train:
        best_v_mrr = 0
        for epoch in range(opts.epoch):
            model.train_batch()
            if (epoch+1) % args.eval_interval == 0:
                result_dict, out_str = model.evaluate(eval_val=True, eval_test=True)
                v_mrr, t_mrr = result_dict['v_mrr'], result_dict['t_mrr']
                print(out_str)
                with open(opts.perf_file, 'a+') as f:
                    f.write(out_str)
                if v_mrr > best_v_mrr:
                    best_v_mrr = v_mrr
                    best_str = out_str
                    print(str(epoch) + '\t' + best_str)
                    BestMetricStr = f'ValMRR_{str(v_mrr)[:5]}_TestMRR_{str(t_mrr)[:5]}'
                    model.saveModelToFiles(BestMetricStr, deleteLastFile=False)
        
        print(best_str)
        
    if opts.eval:
        result_dict, out_str = model.evaluate(eval_val=False, eval_test=True, verbose=True)
        print(result_dict, '\n', out_str)
        with open(opts.perf_file, 'a+') as f:
            f.write(out_str)
        
