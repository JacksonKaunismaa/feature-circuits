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


        self.tracing_edge_nnz = []  # to deal with nnsight tracing
        self.tracing_edge_act = []
        # self.saved_ws = []

        self.edge_nnz = {}
        self.edge_acts = {}

    def compute_node_hist(self, submod, w: t.Tensor):
        # w: [N, seq_len, n_feats]
        submod = submod._module_path
        if submod not in self.node_nnz:
            n_feats = w.shape[2] - 1  # -1 to avoid "error" term
            self.nnz_max[submod] = np.log10(n_feats)
            self.node_nnz[submod] = t.zeros(self.n_bins).cuda()
            self.node_acts[submod] = t.zeros(self.n_bins).cuda()

        w_late = w[:, self.seq_len//2:, :-1]   # -1 to avoid error term
        nnz = (w_late != 0).sum(dim=2).flatten()   # [N, seq_len//2].flatten()
        abs_w = abs(w_late)
        self.node_nnz[submod] += t.histc(t.log10(nnz), bins=self.n_bins, min=self.nnz_min, max=self.nnz_max[submod])
        self.node_acts[submod] += t.histc(abs_w[abs_w != 0], bins=self.n_bins, min=self.act_min, max=self.act_max)


    def compute_edge_hist(self, up_submod, down_submod, w: t.Tensor):
        # w: [N, seq_len, n_feat_down]
        w_late = w[:, self.seq_len//2:, :-1]  # -1 to avoid "error" term
        nnz = (w_late != 0).sum(dim=2).flatten()
        abs_w = abs(w_late)
        nnz_hist = t.histc(t.log10(nnz), bins=self.n_bins, min=self.nnz_min,  # this will implicitly ignore cases where there are no non-zero elements
                                                           max=self.nnz_max[up_submod])
        act_hist = t.histc(abs_w[abs_w != 0], bins=self.n_bins, min=self.act_min,
                                                           max=self.act_max)
        return nnz_hist, act_hist


    def trace_edge_hist(self, up_submod, down_submod, vjv):
        # self.saved_ws.append(vjv.save())
        # if up_submod._module_path != '.gpt_neox.layers.5.attention' or down_submod._module_path != '.gpt_neox.layers.5.mlp':
        up_submod = up_submod._module_path
        down_submod = down_submod._module_path

        nnz, act = self.compute_edge_hist(up_submod, down_submod, vjv)
        self.tracing_edge_nnz.append(nnz.save())
        self.tracing_edge_act.append(act.save())


    def aggregate_edge_hist(self, up_submod, down_submod):
        up_submod = up_submod._module_path
        down_submod = down_submod._module_path

        if (up_submod, down_submod) not in self.edge_nnz:
            self.edge_nnz[(up_submod, down_submod)] = t.zeros(self.n_bins).cuda()
            self.edge_acts[(up_submod, down_submod)] = t.zeros(self.n_bins).cuda()

        for nnz_hist, act_hist in zip(self.tracing_edge_nnz, self.tracing_edge_act):
            self.edge_nnz[(up_submod, down_submod)] += nnz_hist.value
            self.edge_acts[(up_submod, down_submod)] += act_hist.value

        self.tracing_edge_nnz.clear()
        self.tracing_edge_act.clear()
        # self.saved_ws.clear()

    def save(self, path):
        t.save({'node_nnz': self.node_nnz,
                'node_acts': self.node_acts,
                'edge_nnz': self.edge_nnz,
                'edge_acts': self.edge_acts}, path)

