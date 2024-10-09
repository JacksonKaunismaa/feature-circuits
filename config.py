import dataclasses
from typing import Literal
from argparse import Namespace

@dataclasses.dataclass
class Config:
    # Algorithm controls
    node_sparsity: float = 0.0
    edge_sparsity: float = 0.0
    as_threshold: bool = False  # if True, edge_sparsity is treated as a threshold, instead of a sparsity amount
    nodes_only: bool = False
    method: Literal['ig', 'attrib', 'exact'] = 'ig'
    aggregation: Literal['none', 'sum'] = 'sum'
    prune_method: Literal['none', 'source-sink', 'sink-backwards', 'first-layer-sink'] = 'none'
    collect_hists: int = 0  # if > 0 then collect histograms for the first collect_hists examples

    # Data
    dataset: str = 'simple_train'
    num_examples: int = 100
    batch_size: int = 32
    example_length: int | None = None
    data_type: Literal['nopair', 'regular', 'hf'] = 'regular'

    # Plotting controls
    edge_thickness_normalization: Literal['linear', 'log'] = 'linear'
    pen_thickness: float = 0.5
    plot_only: bool = False

    # Model
    model: str = 'gpt2'
    has_embed: bool = True
    device: str = 'cuda'
    d_model: int = 512
    resid_posn: Literal['post', 'mid', 'pre'] = 'post'
    layers: int = 6
    first_component: Literal['embed', 'attn_0', 'resid_0'] = 'embed'
    parallel_attn: bool = False
    dict_id: str = 'gpt2'
    annotations_path: str = ''

    # Miscellaneous
    disable_tqdm: bool = False
    seed: int = 42

    def update(self, args: Namespace):
        for k, v in vars(args).items():
            if hasattr(self, k):
                setattr(self, k, v)


    def as_fname(self):
        return f'dict{self.dict_id}_node{self.node_sparsity}_edge{self.edge_sparsity}_n{self.num_examples}_agg{self.aggregation}_thresh{self.as_threshold}_method{self.method}_prune{self.prune_method}_model{self.model.replace("/", "_")}'
