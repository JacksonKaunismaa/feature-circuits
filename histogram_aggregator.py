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
    THRESH = 'thresh'  # raw threshold that activations must exceed
    SPARSITY = 'sparsity'   # the number of features to choose per sequence position
    Z_SCORE = 'z_score'   # how many stds above the mean must the activation be to be considered
    PEAK_MATCH = 'peak_match'  # uhh....
    PERCENTILE = 'percentile'   # compute a threshold based on histograms that will say to keep the top thresh % of weights

class Histogram:
    def __init__(self, n_bins, act_min, act_max, nnz_min, nnz_max, model_str):
        self.n_bins = n_bins
        self.act_min = act_min
        self.act_max = act_max
        self.nnz_min = nnz_min
        self.nnz_max = nnz_max  # nnz max for the largest case
        self.model_str = model_str
        self.n_samples = 0

        self.nnz = t.zeros(n_bins).cuda()
        self.acts = t.zeros(n_bins).cuda()

        self.error_nnz = t.zeros(n_bins).cuda()
        self.error_acts = t.zeros(n_bins).cuda()

        self.first_nnz = t.zeros(n_bins).cuda()
        self.first_acts = t.zeros(n_bins).cuda()

        self.error_first_nnz = t.zeros(2).cuda()  # 2 bins since it can only be 0 or 1
        self.error_first_acts = t.zeros(n_bins).cuda()

        self.tracing_nnz = []
        self.tracing_acts = []

    @t.no_grad()
    def compute_hist(self, w: t.Tensor):
        abs_w = abs(w)
        acts_hist = t.histc(t.log10(abs_w[abs_w != 0]), bins=self.n_bins, min=self.act_min, max=self.act_max)
        nnz = (w != 0).sum(dim=2).flatten()
        nnz_hist = t.histc(t.log10(nnz), bins=self.n_bins, min=self.nnz_min, max=self.nnz_max)
        return nnz_hist, acts_hist


    @t.no_grad()
    def compute_hist(self, w: t.Tensor, trace=False):
        self.n_samples += 1

        w_late = w[:, 1:, :-1]
        nnz, acts = self.compute_hist(w_late)

        w_first = w[:, :1, :-1]  # -1 to avoid "error" term
        nnz_first, acts_first = self.compute_hist(w_first)

        w_error = w[:, 1:, -1:]
        nnz_error, acts_error = self.compute_hist(w_error)

        w_first_error = w[:, :1, -1:]
        _, acts_first_error = self.compute_hist(w_first_error)

        if trace:
            self.tracing_nnz.append(nnz.save(), nnz_error.save(), nnz_first.save(), w_first_error.save())
            self.tracing_acts.append(acts.save(), acts_error.save(), acts_first.save(), acts_first_error.save())
        else:
            self.nnz += nnz
            self.acts += acts

            self.first_nnz += nnz_first
            self.first_acts += acts_first

            self.error_nnz += nnz_error
            self.error_acts += acts_error


    def aggregate_traced(self):
        for nnz, nnz_error, nnz_first, w_first_error in self.tracing_nnz:
            self.nnz += nnz.value
            self.error_nnz += nnz_error.value
            self.first_nnz += nnz_first.value
            self.error_first_nnz[0] += (w_first_error.value == 0).sum()
            self.error_first_nnz[1] += (w_first_error.value != 0).sum()

        for acts, acts_error, acts_first, acts_first_error in self.tracing_acts:
            self.acts += acts.value
            self.error_acts += acts_error.value
            self.first_acts += acts_first.value
            self.error_first_acts += acts_first_error.value

        self.tracing_nnz.clear()
        self.tracing_acts.clear()

    def cpu(self):
        if isinstance(self.nnz, t.Tensor):
            self.nnz = self.nnz.cpu().numpy()
            self.acts = self.acts.cpu().numpy()
            self.error_nnz = self.error_nnz.cpu().numpy()
            self.error_acts = self.error_acts.cpu().numpy()
            self.first_nnz = self.first_nnz.cpu().numpy()
            self.first_acts = self.first_acts.cpu().numpy()
            self.error_first_nnz = self.error_first_nnz.cpu().numpy()
            self.error_first_acts = self.error_first_acts.cpu().numpy()
        return self



class HistAggregator:
    def __init__(self, model_str='gpt2', seq_len=64, n_bins=10_000, act_min=-10, act_max=5):
        self.n_bins = n_bins
        self.nnz_min = 0
        self.nnz_max = {} # will be set later
        self.act_min = act_min   # power of 10
        self.act_max = act_max
        self.seq_len = seq_len
        self.model_str = model_str
        self.n_samples = defaultdict(int)

        self.nodes: dict[str, Histogram] = {}
        self.edges: dict[dict[str, Histogram]] = {}


    @t.no_grad()
    def compute_node_hist(self, submod, w: t.Tensor):
        # w: [N, seq_len, n_feats]
        submod = get_submod_repr(submod)
        if submod not in self.nodes:
            n_feats = w.shape[2] - 1  # -1 to avoid "error" term
            self.nnz_max[submod] = np.log10(n_feats)
            self.nodes[submod] = Histogram(self.n_bins, self.act_min, self.act_max, self.nnz_min,
                                              self.nnz_max[submod], self.model_str)

        self.nodes[submod].compute_hist(w)


    @t.no_grad()
    def trace_edge_hist(self, up_submod, down_submod, vjv):
        up_submod = get_submod_repr(up_submod)
        down_submod = get_submod_repr(down_submod)

        if up_submod not in self.edges:
            self.edges[up_submod] = {}

        if down_submod not in self.edges[up_submod]:
            self.edges[up_submod][down_submod] = Histogram(self.n_bins, self.act_min, self.act_max, self.nnz_min,
                                                            self.nnz_max[up_submod], self.model_str)
        self.edges[up_submod][down_submod].compute_hist(vjv, trace=True)

    @t.no_grad()
    def aggregate_edge_hist(self, up_submod, down_submod):
        up_submod = get_submod_repr(up_submod)
        down_submod = get_submod_repr(down_submod)

        self.edges[up_submod][down_submod].aggregate_traced()


    def cpu(self):
        for n in self.nodes.values():
            n.cpu()
        for up in self.edges:
            for e in self.edges[up].values():
                e.cpu()
        return self

    def save(self, path):
        t.save({'nodes': self.nodes,
                'edges': self.edges,
                'nnz_max': self.nnz_max,
                'act_min_max': (self.act_min, self.act_max),
                'model_str': self.model_str,
                'n_samples': self.n_samples,
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
        self.nodes = data['nodes']
        self.edges = data['edges']
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
            hist = self.nodes[mod_name].acts
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

            case ThresholdType.PERCENTILE:
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
                hist, bins, _, _, _ = self.get_hist_settings(hist, feat_size, 'acts')
                thresh_loc = self.get_threshold(hist, bins, thresh, thresh_type)
                thresh_vals[up][down] = 10**bins[thresh_loc]
        return thresh_vals

    def compute_node_thresholds(self, thresh: float, thresh_type: ThresholdType):
        thresh_vals = {}
        for mod_name in self.node_acts:
            hist = self.node_acts[mod_name]
            feat_size = 10**self.nnz_max[mod_name]
            hist, bins, _, _, _ = self.get_hist_settings(hist, feat_size, 'acts')
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
