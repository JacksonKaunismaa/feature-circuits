from typing import Literal
import numpy as np
import torch as t
import matplotlib.pyplot as plt

from dictionary_learning.dictionary import AutoEncoder
from graph_utils import dfs

def normalize_path(path):
    parts = path.split('.')
    try:
        layer = int(parts[-1])   # if the last part is an integer, then we are dealing with a resid
        component = 'resid'
    except ValueError:
        component = parts[-1]
        if component == 'embed_in':
            return 'embed'
        if component == 'attention':
            component = 'attn'
        layer = int(parts[-2])

    return f'{component}_{layer}'

def get_submod_repr(submod):
    if hasattr(submod, '_module_path'):
        path =  submod._module_path
    elif hasattr(submod, 'path'):
        path = submod.path
    else:
        raise ValueError('submod does not have _module_path or path attribute')
    return normalize_path(path)


class HistAggregator:
    def __init__(self, model_str='gpt2', seq_len=64, n_bins=5000, act_min=-10, act_max=10):
        self.n_bins = n_bins
        self.nnz_min = 0
        self.nnz_max = {} # will be set later
        self.act_min = act_min   # power of 10
        self.act_max = act_max
        self.seq_len = seq_len
        self.model_str = model_str

        self.node_nnz = {}
        self.node_acts = {}

        self.tracing_edge_nnz = []  # to deal with nnsight tracing
        self.tracing_edge_act = []
        # self.saved_ws = []

        self.edge_nnz = {}
        self.edge_acts = {}

    def compute_node_hist(self, submod, w: t.Tensor):
        # w: [N, seq_len, n_feats]
        submod = get_submod_repr(submod)
        if submod not in self.node_nnz:
            n_feats = w.shape[2] - 1  # -1 to avoid "error" term
            self.nnz_max[submod] = np.log10(n_feats)
            self.node_nnz[submod] = t.zeros(self.n_bins).cuda()
            self.node_acts[submod] = t.zeros(self.n_bins).cuda()

        w_late = w[:, self.seq_len//2:, :-1]   # -1 to avoid error term
        nnz = (w_late != 0).sum(dim=2).flatten()   # [N, seq_len//2].flatten()
        abs_w = abs(w_late)
        self.node_nnz[submod] += t.histc(t.log10(nnz), bins=self.n_bins, min=self.nnz_min, max=self.nnz_max[submod])
        self.node_acts[submod] += t.histc(t.log10(abs_w[abs_w != 0]), bins=self.n_bins, min=self.act_min, max=self.act_max)


    def compute_edge_hist(self, up_submod, down_submod, w: t.Tensor):
        # w: [N, seq_len, n_feat_down]
        w_late = w[:, self.seq_len//2:, :-1]  # -1 to avoid "error" term
        nnz = (w_late != 0).sum(dim=2).flatten()
        abs_w = abs(w_late)
        nnz_hist = t.histc(t.log10(nnz), bins=self.n_bins, min=self.nnz_min,  # this will implicitly ignore cases where there are no non-zero elements
                                                           max=self.nnz_max[up_submod])
        act_hist = t.histc(t.log10(abs_w[abs_w != 0]), bins=self.n_bins, min=self.act_min,
                                                           max=self.act_max)
        return nnz_hist, act_hist


    def trace_edge_hist(self, up_submod, down_submod, vjv):
        # self.saved_ws.append(vjv.save())
        # if up_submod._module_path != '.gpt_neox.layers.5.attention' or down_submod._module_path != '.gpt_neox.layers.5.mlp':
        up_submod = get_submod_repr(up_submod)
        down_submod = get_submod_repr(down_submod)

        nnz, act = self.compute_edge_hist(up_submod, down_submod, vjv)
        self.tracing_edge_nnz.append(nnz.save())
        self.tracing_edge_act.append(act.save())


    def aggregate_edge_hist(self, up_submod, down_submod):
        up_submod = get_submod_repr(up_submod)
        down_submod = get_submod_repr(down_submod)

        if up_submod not in self.edge_nnz:
            self.edge_nnz[up_submod] = {}
            self.edge_acts[up_submod] = {}

        if down_submod not in self.edge_nnz[up_submod]:
            self.edge_nnz[up_submod][down_submod] = t.zeros(self.n_bins).cuda()
            self.edge_acts[up_submod][down_submod] = t.zeros(self.n_bins).cuda()

        for nnz_hist, act_hist in zip(self.tracing_edge_nnz, self.tracing_edge_act):
            self.edge_nnz[up_submod][down_submod] += nnz_hist.value
            self.edge_acts[up_submod][down_submod] += act_hist.value

        self.tracing_edge_nnz.clear()
        self.tracing_edge_act.clear()
        # self.saved_ws.clear()

    def cpu(self):
        for k in self.node_nnz:
            self.node_nnz[k] = self.node_nnz[k].cpu().numpy()
            self.node_acts[k] = self.node_acts[k].cpu().numpy()
        for up in self.edge_nnz:
            for down in self.edge_nnz[up]:
                self.edge_nnz[up][down] = self.edge_nnz[up][down].cpu().numpy()
                self.edge_acts[up][down] = self.edge_acts[up][down].cpu().numpy()

    def save(self, path):
        t.save({'node_nnz': self.node_nnz,
                'node_acts': self.node_acts,
                'edge_nnz': self.edge_nnz,
                'edge_acts': self.edge_acts,
                'nnz_max': self.nnz_max,
                'act_min_max': (self.act_min, self.act_max),
                'model_str': self.model_str
                }, path)

    def load(self, path_or_dict):
        if isinstance(path_or_dict, str):
            data = t.load(path_or_dict)
        else:
            data = path_or_dict
        self.node_nnz = data['node_nnz']
        self.node_acts = data['node_acts']
        self.edge_nnz = data['edge_nnz']
        self.edge_acts = data['edge_acts']
        self.nnz_max = data['nnz_max']
        self.model_str = data['model_str']
        self.act_min, self.act_max = data['act_min_max']
        self.cpu()
        self.n_bins = list(self.node_nnz.values())[0].shape[0]


    def get_hist_for_node_effect(self, layer, component, acts_or_nnz):
        mod_name = f'{component}_{layer}'
        if acts_or_nnz == 'acts':
            hist = self.node_acts[mod_name]
        else:
            hist = self.node_nnz[mod_name]

        feat_size = int(np.round(10**self.nnz_max[mod_name]))
        return hist, feat_size

    def get_hist_settings(self, hist, n_feats, acts_or_nnz='acts', thresh=None, as_sparsity=False):
        if acts_or_nnz == 'acts':
            min_val = self.act_min
            max_val = self.act_max
            xlabel = 'log10(Activation magnitude)'
            bins = np.linspace(min_val, max_val, self.n_bins)
            if thresh is not None:
                if not as_sparsity:
                    thresh_loc = np.searchsorted(bins, np.log(thresh))
                else:
                    percentile_hist = np.cumsum(hist) / hist.sum()
                    thresh_loc = np.searchsorted(percentile_hist, 1-thresh)
                hist = hist.copy()
                hist[:thresh_loc-1] = 0
        else:
            if thresh is not None:
                raise ValueError("thresh can only be computed for hist")
            min_val = 0
            xlabel = 'NNZ'
            max_val = np.log10(n_feats)
            bins = 10 ** (np.linspace(min_val, max_val, self.n_bins))
            max_index = np.nonzero(hist)[0].max()
            max_val = bins[max_index]
            bins = bins[:max_index+1]
            hist = hist[:max_index+1]
        return hist, bins, xlabel

    def plot_hist(self, hist, bins, ax, xlabel, title):
        value_hist_color = 'blue'
        ax.set_xlabel(xlabel, color=value_hist_color)
        ax.set_ylabel('Frequency', color=value_hist_color)
        ax.plot(bins, hist, color=value_hist_color)
        ax.tick_params(axis='x', colors=value_hist_color)
        ax.tick_params(axis='y', colors=value_hist_color)
        # ax.set_xlim(min(min_nnz, min_val), max(max_nnz, max_val))
        # compute median value of activations
        median_idx = (hist.cumsum() >= hist.sum() / 2).nonzero()[0][0]
        median_val = bins[median_idx]
        # compute variance of activations
        total = hist.sum()
        mean = (bins * hist).sum() / total
        std = np.sqrt(((bins - mean)**2 * hist).sum() / total)
        ax.set_title(f'{title} : (log10(total) = {np.log10(total):.2f})')
        # vertical line at mean
        ax.axvline(median_val, color='r', linestyle='--')
        # add text with mean
        ax.text(median_val+0.5, hist.max(), f'{median_val:.2f} +- {std:.2f}', color='r')

    def plot(self, n_layers, nodes_or_edges: Literal['nodes', 'edges']='nodes',
             acts_or_nnz:Literal['acts', 'nnz'] ='acts', thresh=None, as_sparsity=False):
        if nodes_or_edges == 'nodes':
            fig, axs = plt.subplots(n_layers, 3, figsize=(18, 3.6*n_layers))

            for layer in range(n_layers):
                for i, component in enumerate(['resid', 'attn', 'mlp']):
                    hist, feat_size = self.get_hist_for_node_effect(layer, component, acts_or_nnz)
                    hist, bins, xlabel = self.get_hist_settings(hist, feat_size, acts_or_nnz, thresh, as_sparsity)
                    self.plot_hist(hist, bins, axs[layer, i], xlabel, f'{self.model_str} {component} layer {layer}')

        elif nodes_or_edges == 'edges':
            edges_per_layer = 6 if self.model_str == 'gpt2' else 5
            first_layer_type = 'attn' if self.model_str == 'gpt2' else 'embed'
            total_edges = len(self.edge_acts)
            fig, axs = plt.subplots(n_layers, edges_per_layer, figsize=(6*edges_per_layer, 3.6*n_layers))

            edges_hists = self.edge_acts if acts_or_nnz == 'acts' else self.edge_nnz

            for layer in range(n_layers):
                if layer == 0:
                    if first_layer_type == 'embed':
                        start_node = 'embed'
                    else:
                        start_node = f'{first_layer_type}_{layer}'
                    end_node = f'resid_0'
                else:
                    start_node = f'resid_{layer-1}'
                    end_node = f'resid_{layer}'

                layer_edges = dfs(edges_hists, start_node, end_node)
                for i, (up, down) in enumerate(layer_edges):
                    hist = edges_hists[up][down]
                    hist, bins, xlabel = self.get_hist_settings(hist, self.nnz_max[up], acts_or_nnz, thresh, as_sparsity)
                    self.plot_hist(hist, bins, axs[layer, i], xlabel, f'{self.model_str} layer {layer} edge {(up, down)}')


        plt.tight_layout()
        plt.show()