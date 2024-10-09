import numpy as np
import torch as t

from dictionary_learning.dictionary import AutoEncoder

class HistAggregator:
    def __init__(self, seq_len, n_bins=1500):
        self.n_bins = n_bins
        self.nnz_min = 0
        self.nnz_max = {} # will be set later
        self.act_min = -10
        self.act_max = 10
        self.seq_len = seq_len

        self.node_nnz = {}
        self.node_acts = {}

        self.edge_nnz = {}
        self.edge_acts = {}

    def compute_node_hist(self, submod, w: t.Tensor):
        # w: [N, seq_len, n_feats]
        if submod not in nnz:
            n_feats = w.shape[2]
            self.nnz_max[submod] = np.log10(n_feats)
            self.node_nnz[submod] = t.zeros(self.n_bins).cuda()
            self.node_acts[submod] = t.zeros(self.n_bins).cuda

        w_late = w[:, self.seq_len//2:, :]
        nnz = (w_late != 0).sum(dim=2).flatten()   # [N, seq_len//2].flatten()
        abs_w = abs(w_late)
        self.node_nnz[submod] += t.histc(t.log10(nnz), bins=self.n_bins, min=self.nnz_min, max=self.nnz_max[submod])
        self.node_acts[submod] += t.histc(abs_w[abs_w != 0], bins=self.n_bins, min=self.act_min, max=self.act_max)


    def compute_edge_hist(self, up_submod, down_submod, w: t.Tensor):
        # w: [N, seq_len, n_feat_down]
        if (up_submod, down_submod) not in nnz:
            n_feat_down = w.shape[2]
            self.nnz_max[(up_submod, down_submod)] = np.log10(n_feat_down)
            self.edge_nnz[(up_submod, down_submod)] = t.zeros(self.n_bins).cuda()
            self.edge_acts[(up_submod, down_submod)] = t.zeros(self.n_bins).cuda

        w_late = w[:, self.seq_len//2:, :]
        nnz = (w_late != 0).sum(dim=2).flatten()
        abs_w = abs(w_late)
        self.edge_nnz[(up_submod, down_submod)] += t.histc(t.log10(nnz), bins=self.n_bins, min=self.nnz_min,
                                                           max=self.nnz_max[(up_submod, down_submod)])
        self.edge_acts[(up_submod, down_submod)] += t.histc(abs_w[abs_w != 0], bins=self.n_bins, min=self.act_min,
                                                           max=self.act_max)

    def save(self, path):
        t.save({'node_nnz': self.node_nnz,
                'node_acts': self.node_acts,
                'edge_nnz': self.edge_nnz,
                'edge_acts': self.edge_acts}, path)

