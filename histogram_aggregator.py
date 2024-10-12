from collections import defaultdict
import os
from typing import Literal
import warnings
import numpy as np
import torch as t
import matplotlib.pyplot as plt
from enum import Enum

from dictionary_learning.dictionary import AutoEncoder
from graph_utils import dfs, iterate_edges

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
    """Return canonical string represention of a submod (e.g. attn_0, mlp_5, ...)"""
    if isinstance(submod, str):
        return submod
    if hasattr(submod, '_module_path'):
        path =  submod._module_path
    elif hasattr(submod, 'path'):
        path = submod.path
    else:
        raise ValueError('submod does not have _module_path or path attribute')
    return normalize_path(path)

class ThresholdType(Enum):
    THRESH = 'thresh'
    SPARSITY = 'sparsity'
    Z_SCORE = 'z_score'
    PEAK_MATCH = 'peak_match'


class HistAggregator:
    def __init__(self, model_str='gpt2', seq_len=64, n_bins=5000, act_min=-10, act_max=10):
        self.n_bins = n_bins
        self.nnz_min = 0
        self.nnz_max = {} # will be set later
        self.act_min = act_min   # power of 10
        self.act_max = act_max
        self.seq_len = seq_len
        self.model_str = model_str
        self.n_samples = defaultdict(int)

        self.node_nnz = {}
        self.node_acts = {}

        self.tracing_edge_nnz = []  # to deal with nnsight tracing
        self.tracing_edge_act = []
        # self.saved_ws = []

        self.edge_nnz = {}
        self.edge_acts = {}

    @t.no_grad()
    def compute_node_hist(self, submod, w: t.Tensor):
        # w: [N, seq_len, n_feats]
        submod = get_submod_repr(submod)
        if submod not in self.node_nnz:
            n_feats = w.shape[2] - 1  # -1 to avoid "error" term
            self.nnz_max[submod] = np.log10(n_feats)
            self.node_nnz[submod] = t.zeros(self.n_bins).cuda()
            self.node_acts[submod] = t.zeros(self.n_bins).cuda()

        self.n_samples[submod] += 1
        w_late = w[:, self.seq_len//2:, :-1]   # -1 to avoid error term
        nnz = (w_late != 0).sum(dim=2).flatten()   # [N, seq_len//2].flatten()
        abs_w = abs(w_late)
        self.node_nnz[submod] += t.histc(t.log10(nnz), bins=self.n_bins, min=self.nnz_min, max=self.nnz_max[submod])
        self.node_acts[submod] += t.histc(t.log10(abs_w[abs_w != 0]), bins=self.n_bins, min=self.act_min, max=self.act_max)

    @t.no_grad()
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

    @t.no_grad()
    def trace_edge_hist(self, up_submod, down_submod, vjv):
        # self.saved_ws.append(vjv.save())
        # if up_submod._module_path != '.gpt_neox.layers.5.attention' or down_submod._module_path != '.gpt_neox.layers.5.mlp':
        up_submod = get_submod_repr(up_submod)
        down_submod = get_submod_repr(down_submod)

        nnz, act = self.compute_edge_hist(up_submod, down_submod, vjv)
        self.tracing_edge_nnz.append(nnz.save())
        self.tracing_edge_act.append(act.save())

    @t.no_grad()
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
        return self

    def save(self, path):
        t.save({'node_nnz': self.node_nnz,
                'node_acts': self.node_acts,
                'edge_nnz': self.edge_nnz,
                'edge_acts': self.edge_acts,
                'nnz_max': self.nnz_max,
                'act_min_max': (self.act_min, self.act_max),
                'model_str': self.model_str,
                'n_samples': self.n_samples
                }, path)

    def load(self, path_or_dict, map_location=None):
        if isinstance(path_or_dict, str):
            if os.path.exists(path_or_dict):
                data = t.load(path_or_dict, map_location=map_location)
            else:
                warnings.warn(f'Tried to load histogram, but file "{path_or_dict}" was not found...')
                return None
        else:
            data = path_or_dict
        self.node_nnz = data['node_nnz']
        self.node_acts = data['node_acts']
        self.edge_nnz = data['edge_nnz']
        self.edge_acts = data['edge_acts']
        self.nnz_max = data['nnz_max']
        self.model_str = data['model_str']
        self.act_min, self.act_max = data['act_min_max']
        self.n_bins = list(self.node_nnz.values())[0].shape[0]
        self.n_samples = data['n_samples']
        if isinstance(path_or_dict, str):
            print("Successfully loaded histograms at", path_or_dict)
        return self


    def get_hist_for_node_effect(self, layer, component, acts_or_nnz):
        mod_name = f'{component}_{layer}'
        if acts_or_nnz == 'acts':
            hist = self.node_acts[mod_name]
        else:
            hist = self.node_nnz[mod_name]

        feat_size = int(np.round(10**self.nnz_max[mod_name]))
        return hist, feat_size

    def get_mean_median_std(self, hist, bins):
        total = hist.sum()
        median_idx = (hist.cumsum() >= total / 2).nonzero()[0][0]
        median_val = bins[median_idx]
        mean = (bins * hist).sum() / total
        # compute variance of activations
        std = np.sqrt(((bins - mean)**2 * hist).sum() / total)
        return mean, median_val, std

    def get_threshold(self, hist, bins, thresh, thresh_type):
        match thresh_type:
            case ThresholdType.THRESH:
                thresh_loc = np.searchsorted(bins, np.log10(thresh))
            case ThresholdType.SPARSITY:
                percentile_hist = np.cumsum(hist) / hist.sum()
                thresh_loc = np.searchsorted(percentile_hist, 1-thresh)
            case ThresholdType.Z_SCORE:
                mean, _, std = self.get_mean_median_std(hist, bins)
                thresh_loc = np.searchsorted(bins, mean + thresh * std)
            case ThresholdType.PEAK_MATCH:
                thresh_loc = None
                best_diff = np.inf
                for b in range(1, len(hist)):
                    thresh_peak = hist[b:].max()
                    hist_diff = np.abs(thresh_peak - thresh)
                    if hist_diff < best_diff:
                        best_diff = hist_diff
                        thresh_loc = b
        return thresh_loc

    def get_hist_settings(self, hist, n_feats, acts_or_nnz='acts', thresh=None, thresh_type=None):
        if acts_or_nnz == 'acts':
            min_val = self.act_min
            max_val = self.act_max
            xlabel = 'log10(Activation magnitude)'
            bins = np.linspace(min_val, max_val, self.n_bins)
        else:
            min_val = 0
            xlabel = 'NNZ'
            max_val = np.log10(n_feats)
            bins = 10 ** (np.linspace(min_val, max_val, self.n_bins))
            max_index = np.nonzero(hist)[0].max()
            max_val = bins[max_index]
            bins = bins[:max_index+1]
            hist = hist[:max_index+1]

        if thresh is not None:
            if acts_or_nnz == 'nnz':
                raise ValueError("Cannot compute threshold for nnz")
            else:
                thresh_loc = self.get_threshold(hist, bins, thresh, thresh_type)
                hist = hist.copy()
                hist[:thresh_loc-1] = 0

        _, median_val, std = self.get_mean_median_std(hist, bins)
        return hist, bins, xlabel, median_val, std

    def plot_hist(self, hist, median, std, bins, ax, xlabel, title):
        value_hist_color = 'blue'
        ax.set_xlabel(xlabel, color=value_hist_color)
        ax.set_ylabel('Frequency', color=value_hist_color)
        ax.plot(bins, hist, color=value_hist_color)
        ax.tick_params(axis='x', colors=value_hist_color)
        ax.tick_params(axis='y', colors=value_hist_color)
        # ax.set_xlim(min(min_nnz, min_val), max(max_nnz, max_val))
        # compute median value of activations
        ax.set_title(f'{title}')
        # vertical line at mean
        ax.axvline(median, color='r', linestyle='--')
        # add text with mean
        ax.text(median+0.5, hist.max(), f'{median:.2f} +- {std:.2f}', color='r')

    def compute_edge_thresholds(self, thresh: float, thresh_type: ThresholdType):
        thresh_vals = {}
        for up in self.edge_acts:
            thresh_vals[up] = {}
            for down in self.edge_acts[up]:
                hist = self.edge_acts[up][down]
                feat_size = 10**self.nnz_max[up]
                hist, bins, _, _, _ = self.get_hist_settings(hist, feat_size, 'acts', thresh, thresh_type)
                thresh_loc = self.get_threshold(hist, bins, thresh, thresh_type)
                thresh_vals[up][down] = 10**bins[thresh_loc]
        return thresh_vals

    def compute_node_thresholds(self, thresh: float, thresh_type: ThresholdType):
        thresh_vals = {}
        for mod_name in self.node_acts:
            hist = self.node_acts[mod_name]
            feat_size = 10**self.nnz_max[mod_name]
            hist, bins, _, _, _ = self.get_hist_settings(hist, feat_size, 'acts', thresh, thresh_type)
            thresh_loc = self.get_threshold(hist, bins, thresh, thresh_type)
            thresh_vals[mod_name] = 10**bins[thresh_loc]
        return thresh_vals


    def plot(self, n_layers, nodes_or_edges: Literal['nodes', 'edges']='nodes',
             acts_or_nnz:Literal['acts', 'nnz'] ='acts', thresh=None, thresh_type=ThresholdType.THRESH):
        if nodes_or_edges == 'nodes':
            fig, axs = plt.subplots(n_layers, 3, figsize=(18, 3.6*n_layers))

            for layer in range(n_layers):
                for i, component in enumerate(['resid', 'attn', 'mlp']):
                    hist, feat_size = self.get_hist_for_node_effect(layer, component, acts_or_nnz)
                    hist, bins, xlabel, median, std = self.get_hist_settings(hist, feat_size, acts_or_nnz, thresh, thresh_type)
                    self.plot_hist(hist, median, std, bins, axs[layer, i], xlabel, f'{self.model_str} {component} layer {layer}')

        elif nodes_or_edges == 'edges':
            edges_per_layer = 6 if self.model_str == 'gpt2' else 5
            first_component = 'attn_0' if self.model_str == 'gpt2' else 'embed'
            fig, axs = plt.subplots(n_layers, edges_per_layer, figsize=(6*edges_per_layer, 3.6*n_layers))

            edges_hists = self.edge_acts if acts_or_nnz == 'acts' else self.edge_nnz

            for layer in range(n_layers):
                for i, (up, down) in enumerate(iterate_edges(edges_hists, layer, first_component)):
                    hist = edges_hists[up][down]
                    hist, bins, xlabel, median, std = self.get_hist_settings(hist, 10**self.nnz_max[up], acts_or_nnz, thresh, thresh_type)
                    self.plot_hist(hist, median, std, bins, axs[layer, i], xlabel, f'{self.model_str} layer {layer} edge {(up, down)}')


        plt.tight_layout()
        plt.show()
