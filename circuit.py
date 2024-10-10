import argparse
import gc
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch as t
from tqdm import tqdm
from nnsight import LanguageModel

from dictionary_learning import AutoEncoder
from dictionary_learning.dictionary import IdentityDict

from histogram_aggregator import HistAggregator
from activation_utils import SparseAct
from attribution import patching_effect, jvp, threshold_effects
from circuit_plotting import plot_circuit, plot_circuit_posaligned
from loading_utils import load_examples, load_examples_nopair, load_examples_hf
from config import Config


###### utilities for dealing with sparse COO tensors ######
def flatten_index(idxs, shape):
    """
    index : a tensor of shape [n, len(shape)]
    shape : a shape
    return a tensor of shape [n] where each element is the flattened index
    """
    idxs = idxs.t()
    # get strides from shape
    strides = [1]
    for i in range(len(shape)-1, 0, -1):
        strides.append(strides[-1]*shape[i])
    strides = list(reversed(strides))
    strides = t.tensor(strides).to(idxs.device)
    # flatten index
    return (idxs * strides).sum(dim=1).unsqueeze(0)

def prod(l):
    out = 1
    for x in l: out *= x
    return out

def sparse_flatten(x):
    x = x.coalesce()
    return t.sparse_coo_tensor(
        flatten_index(x.indices(), x.shape),
        x.values(),
        (prod(x.shape),)
    )

def reshape_index(index, shape):
    """
    index : a tensor of shape [n]
    shape : a shape
    return a tensor of shape [n, len(shape)] where each element is the reshaped index
    """
    multi_index = []
    for dim in reversed(shape):
        multi_index.append(index % dim)
        index //= dim
    multi_index.reverse()
    return t.stack(multi_index, dim=-1)

def sparse_reshape(x, shape):
    """
    x : a sparse COO tensor
    shape : a shape
    return x reshaped to shape
    """
    # first flatten x
    x = sparse_flatten(x).coalesce()
    new_indices = reshape_index(x.indices()[0], shape)
    return t.sparse_coo_tensor(new_indices.t(), x.values(), shape)

def sparse_mean(x, dim):
    if isinstance(dim, int):
        return x.sum(dim=dim) / x.shape[dim]
    else:
        return x.sum(dim=dim) / prod(x.shape[d] for d in dim)


######## end sparse tensor utilities ########


def aggregate_nodes_edges(nodes, edges, dims, divisor_dim=None):
    for child in edges:
        for parent in edges[child]:

            if divisor_dim is not None:
                divisor = nodes[child].shape[divisor_dim]
            else:
                divisor = 1.

            weight_matrix = edges[child][parent]
            if weight_matrix.shape[0] == 0:
                continue

            if parent == 'y':
                weight_matrix = weight_matrix.sum(dim=dims[0]) / divisor
            else:
                weight_matrix = weight_matrix.sum(dim=dims) / divisor
            edges[child][parent] = weight_matrix
    for node in nodes:
        if node != 'y':
            nodes[node] = nodes[node].sum(dim=dims[0]) / divisor



def get_circuit(
        clean,
        patch,
        model,
        embed,
        attns,
        mlps,
        resids,
        dictionaries,
        cfg: Config,
        hist_agg: HistAggregator,
        metric_fn,
        metric_kwargs=dict(),
):

    all_submods = ([embed] if embed is not None else []) + \
        [submod for layer_submods in zip(mlps, attns, resids) for submod in layer_submods]

    # first get the patching effect of everything on y
    effects, deltas, grads, total_effect = patching_effect(
        clean,
        patch,
        model,
        all_submods,
        dictionaries,
        metric_fn,
        metric_kwargs=metric_kwargs,
        method=cfg.method # get better approximations for early layers by using ig
    )

    t.cuda.empty_cache()  # helps a bit with memory management

    features_by_submod = {}
    for submod in all_submods:
        effect = effects[submod].to_tensor()
        if cfg.collect_hists > 0:
            hist_agg.compute_node_hist(submod, effect)
        features_by_submod[submod] = threshold_effects(effect, cfg.node_sparsity, cfg.as_threshold)

    # submodule -> list of indices

    n_layers = len(resids)

    nodes = {'y' : total_effect}
    if embed is not None:
        nodes['embed'] = effects[embed]
    for i in range(n_layers):
        nodes[f'attn_{i}'] = effects[attns[i]]
        nodes[f'mlp_{i}'] = effects[mlps[i]]
        nodes[f'resid_{i}'] = effects[resids[i]]

    if cfg.nodes_only:
        if cfg.aggregation == 'sum':
            for k in nodes:
                if k != 'y':
                    nodes[k] = nodes[k].sum(dim=1)
        nodes = {k : v.mean(dim=0) for k, v in nodes.items()}
        return nodes, None

    edges = defaultdict(lambda:{})
    edges[f'resid_{len(resids)-1}'] = { 'y' : effects[resids[-1]].to_tensor().to_sparse() }

    def N(upstream, downstream, intermediate_stop_grads=None):
        return jvp(
            clean,
            model,
            dictionaries,
            downstream,  # downstream submod
            features_by_submod[downstream],  # downstream features
            upstream,   # upstream submod
            grads[downstream],  # left_vec
            deltas[upstream],   # right_vec
            cfg,
            hist_agg,
            intermediate_stop_grads=intermediate_stop_grads
        )

    # now we work backward through the model to get the edges
    for layer in reversed(range(len(resids))):
        resid = resids[layer]
        mlp = mlps[layer]
        attn = attns[layer]

        MR_effect = N(mlp, resid)
        AR_effect = N(attn, resid, [mlp])


        edges[f'mlp_{layer}'][f'resid_{layer}'] = MR_effect
        edges[f'attn_{layer}'][f'resid_{layer}'] = AR_effect

        if not cfg.parallel_attn:
            AM_effect = N(attn, mlp)
            edges[f'attn_{layer}'][f'mlp_{layer}'] = AM_effect


        if layer > 0:
            prev_resid = resids[layer-1]
            upstream_name = f'resid_{layer-1}'
        else:
            if embed is not None:
                prev_resid = embed
                upstream_name = 'embed'
            else:
                return

        RM_effect = N(prev_resid, mlp, [attn])
        RA_effect = N(prev_resid, attn)
        RR_effect = N(prev_resid, resid, [mlp, attn])

        edges[upstream_name][f'mlp_{layer}'] = RM_effect
        edges[upstream_name][f'attn_{layer}'] = RA_effect
        edges[upstream_name][f'resid_{layer}'] = RR_effect
        print("layer done", layer)

    if cfg.aggregation == 'sum':
        # aggregate across sequence position
        aggregate_nodes_edges(nodes, edges, (1, 4))
        # aggregate across batch dimension
        aggregate_nodes_edges(nodes, edges, (0, 2), divisor_dim=0)

    elif cfg.aggregation == 'none':
        # aggregate across batch dimensions
        aggregate_nodes_edges(nodes, edges, (0, 3), divisor_dim=0)

    else:
        raise ValueError(f"Unknown aggregation: {cfg.aggregation}")

    return nodes, edges



def process_examples(model, embed, attns, mlps, resids, dictionaries, example_basename, examples, cfg: Config, hist_agg: HistAggregator):
    batch_size = cfg.batch_size
    num_examples = min([cfg.num_examples, len(examples)])
    n_batches = math.ceil(num_examples / batch_size)
    batches = [
        examples[batch*batch_size:(batch+1)*batch_size] for batch in range(n_batches)
    ]
    if num_examples < cfg.num_examples and not cfg.disable_tqdm: # warn the user
        print(f"Total number of examples is less than {cfg.num_examples}. Using {num_examples} examples instead.")

    if not cfg.plot_only:
        nodes, edges = compute_circuit(model, embed, attns, mlps, resids, dictionaries,
                                       example_basename, examples, cfg, hist_agg, num_examples, batches)
    else:
        with open(f'{args.circuit_dir}/{example_basename}_{cfg.as_fname()}.pt', 'rb') as infile:
            save_dict = t.load(infile)
        nodes = save_dict['nodes']
        edges = save_dict['edges']

    # feature annotations
    try:
        annotations = {}
        with open(f"annotations/{args.dict_id}_{args.dict_size}.jsonl", 'r') as annotations_data:
            for annotation_line in annotations_data:
                annotation = json.loads(annotation_line)
                annotations[annotation["Name"]] = annotation["Annotation"]
    except FileNotFoundError:
        annotations = None


    example_text = None
    if cfg.data_type == 'hf':
        example_text = ''.join(model.tokenizer.decode(examples[0]['clean_prefix'][0])) + ' -> ' + model.tokenizer.decode([examples[0]['clean_answer']])
    elif cfg.aggregation == "none":
        example_text = model.tokenizer.batch_decode(examples[0]["clean_prefix"])[0]

    if cfg.collect_hists == 0:
        plot_circuit(nodes,
                    edges,
                    annotations,
                    cfg,
                    example_text,
                    save_path=f'{args.plot_dir}/{example_basename}_{cfg.as_fname()}'
                    )

def compute_circuit(model, embed, attns, mlps, resids, dictionaries, example_basename, examples, cfg: Config, hist_agg, num_examples, batches):
    running_nodes = None
    running_edges = None

    for batch in tqdm(batches, desc="Batches", disable=cfg.disable_tqdm):
        clean_inputs = t.cat([e['clean_prefix'] for e in batch], dim=0).to(cfg.device)
        clean_answer_idxs = t.tensor([e['clean_answer'] for e in batch], dtype=t.long, device=cfg.device)
        if cfg.model == 'gpt2':
            model_out = model.lm_head
        else:
            model_out = model.embed_out

        if cfg.data_type in ['nopair', 'hf']:
            patch_inputs = None
            def metric_fn(model):
                return (
                        -1 * t.gather(
                            t.nn.functional.log_softmax(model_out.output[:,-1,:], dim=-1), dim=-1, index=clean_answer_idxs.view(-1, 1)
                        ).squeeze(-1)
                    )
        else:
            patch_inputs = t.cat([e['patch_prefix'] for e in batch], dim=0).to(cfg.device)
            patch_answer_idxs = t.tensor([e['patch_answer'] for e in batch], dtype=t.long, device=cfg.device)
            def metric_fn(model):
                return (
                        t.gather(model_out.output[:,-1,:], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) - \
                        t.gather(model_out.output[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
                    )

        nodes, edges = get_circuit(
                clean_inputs,
                patch_inputs,
                model,
                embed,
                attns,
                mlps,
                resids,
                dictionaries,
                cfg,
                hist_agg,
                metric_fn,
                metric_kwargs=dict(),
            )

        if running_nodes is None:
            running_nodes = {k : len(batch) * nodes[k].to('cpu') for k in nodes.keys() if k != 'y'}
            if not cfg.nodes_only:
                running_edges = { k : { kk : len(batch) * edges[k][kk].to('cpu') for kk in edges[k].keys() } for k in edges.keys()}
        else:
            for k in nodes.keys():
                if k != 'y':
                    running_nodes[k] += len(batch) * nodes[k].to('cpu')
            if not cfg.nodes_only:
                for k in edges.keys():
                    for v in edges[k].keys():
                        running_edges[k][v] += len(batch) * edges[k][v].to('cpu')

            # memory cleanup
        del nodes, edges
        gc.collect()

    nodes = {k : v.to(cfg.device) / num_examples for k, v in running_nodes.items()}
    if not cfg.nodes_only:
        edges = {k : {kk : 1/num_examples * v.to(cfg.device) for kk, v in running_edges[k].items()} for k in running_edges.keys()}
    else: edges = None

    save_dict = {
            "examples" : examples,
            "nodes": nodes,
            "edges": edges
        }
    if cfg.collect_hists == 0:
        with open(f'{args.circuit_dir}/{example_basename}_{cfg.as_fname()}.pt', 'wb') as outfile:
            t.save(save_dict, outfile)

    return nodes, edges


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='simple_train',
                        help="A subject-verb agreement dataset in data/, or a path to a cluster .json.")
    parser.add_argument('--num_examples', '-n', type=int, default=100,
                        help="The number of examples from the --dataset over which to average indirect effects.")
    parser.add_argument('--example_length', '-l', type=int, default=None,
                        help="The max length (if using sum aggregation) or exact length (if not aggregating) of examples.")
    parser.add_argument('--model', type=str, default='EleutherAI/pythia-70m-deduped',
                        help="The Huggingface ID of the model you wish to test.")
    parser.add_argument("--dict_path", type=str, default="dictionaries/pythia-70m-deduped/",
                        help="Path to all dictionaries for your language model.")
    parser.add_argument('--d_model', type=int, default=512,
                        help="Hidden size of the language model.")
    parser.add_argument('--dict_id', type=str, default=10,
                        help="ID of the dictionaries. Use `id` to obtain circuits on neurons/heads directly.")
    parser.add_argument('--dict_size', type=int, default=32768,
                        help="The width of the dictionary encoder.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Number of examples to process at once when running circuit discovery.")
    parser.add_argument('--aggregation', type=str, default='sum',
                        help="Aggregation across token positions. Should be one of `sum` or `none`.")
    parser.add_argument('--node_sparsity', type=float, default=0.2,
                        help="Indirect effect sparsity for keeping circuit nodes.")
    parser.add_argument('--edge_sparsity', type=float, default=0.02,
                        help="Indirect effect sparsity for keeping edges.")
    parser.add_argument('--as_threshold', default=False, action='store_true',
                        help="Whether to treat node_sparsity and edge_sparsity as thresholds rather than sparsity levels.")
    parser.add_argument('--pen_thickness', type=float, default=0.5,
                        help="Scales the width of the edges in the circuit plot.")
    parser.add_argument('--collect_hists', default=0, type=int,
                        help="Collect histograms of edge and node weights for the first collect_hists examples rather"
                             " than compute circuits. 0 to disable.")
    parser.add_argument('--prune_method', type=str, default='none',
                        choices=['none', 'source-sink', 'sink-backwards', 'first-layer-sink'],
                        help="Pruning method for the finished circuit (see circuit_plotting.build_pruned_graph)")
    parser.add_argument(
        '--edge_thickness_normalization',
        type=str,
        choices=['log', 'linear'],
        default='log',
        help="Specify the normalization type for edge thickness."
    )
    parser.add_argument(
        '--data_type',
        type=str,
        choices=['nopair', 'regular', 'hf'],
        default='regular',
        help="Specify the type of the dataset."
    )

    parser.add_argument('--plot_circuit', default=False, action='store_true',
                        help="Plot the circuit after discovering it.")
    parser.add_argument('--nodes_only', default=False, action='store_true',
                        help="Only search for causally implicated features; do not draw edges.")
    parser.add_argument('--plot_only', action="store_true",
                        help="Do not run circuit discovery; just plot an existing circuit.")
    parser.add_argument("--circuit_dir", type=str, default="circuits/",
                        help="Directory to save/load circuits.")
    parser.add_argument("--plot_dir", type=str, default="circuits/figures/",
                        help="Directory to save figures.")
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    cfg = Config()
    cfg.update(args)

    device = cfg.device

    model = LanguageModel(cfg.model, device_map=device, dispatch=True)

    if cfg.model != 'gpt2':
        embed = model.gpt_neox.embed_in
        attns = [layer.attention for layer in model.gpt_neox.layers]
        mlps = [layer.mlp for layer in model.gpt_neox.layers]
        resids = [layer for layer in model.gpt_neox.layers]
    else:
        embed = None   # embedding SAE doesn't exist for gpt2
        attns = [layer.attn for layer in model.transformer.h]
        mlps = [layer.mlp for layer in model.transformer.h]
        resids = [layer for layer in model.transformer.h]

    dictionaries = {}
    hists = {}

    if cfg.dict_id == 'id':
        dictionaries[embed] = IdentityDict(cfg.d_model)
        for i in range(len(model.gpt_neox.layers)):
            dictionaries[resids[i]] = IdentityDict(cfg.d_model)
            dictionaries[mlps[i]] = IdentityDict(cfg.d_model)
            dictionaries[attns[i]] = IdentityDict(cfg.d_model)

    elif cfg.dict_id == 'gpt2':
        cfg.resid_posn = 'post'
        cfg.parallel_attn = False
        for i in range(len(model.transformer.h)):
            dictionaries[resids[i]] = AutoEncoder.from_hf(
                "jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs",
                f"v5_32k_layer_{i}.pt/sae_weights.safetensors",
                device=device
            )
            dictionaries[mlps[i]] = AutoEncoder.from_hf(
                "jbloom/GPT2-Small-OAI-v5-128k-mlp-out-SAEs",
                f"v5_128k_layer_{i}/sae_weights.safetensors",
                device=device
            )
            dictionaries[attns[i]] = AutoEncoder.from_hf(
                "jbloom/GPT2-Small-OAI-v5-128k-attn-out-SAEs",
                f"v5_128k_layer_{i}/sae_weights.safetensors",
                device=device
            )
    else:
        cfg.parallel_attn = True
        cfg.resid_posn = 'post'
        dictionaries[embed] = AutoEncoder.from_pretrained(
            f'{args.dict_path}/embed/{args.dict_id}_{args.dict_size}/ae.pt',
            device=device
        )

        for i in range(len(model.gpt_neox.layers)):
            dictionaries[resids[i]] = AutoEncoder.from_pretrained(
                f'{args.dict_path}/resid_out_layer{i}/{args.dict_id}_{args.dict_size}/ae.pt',
                device=device
            )
            dictionaries[mlps[i]] = AutoEncoder.from_pretrained(
                f'{args.dict_path}/mlp_out_layer{i}/{args.dict_id}_{args.dict_size}/ae.pt',
                device=device
            )
            dictionaries[attns[i]] = AutoEncoder.from_pretrained(
                f'{args.dict_path}/attn_out_layer{i}/{args.dict_id}_{args.dict_size}/ae.pt',
                device=device
            )

    if args.data_type == 'nopair':
        save_basename = os.path.splitext(os.path.basename(args.dataset))[0]
        examples = load_examples_nopair(args.dataset, args.num_examples, model, length=args.example_length)
    elif args.data_type == 'regular':
        data_path = f"data/{args.dataset}.json"
        save_basename = args.dataset
        if args.aggregation == "sum":
            examples = load_examples(data_path, args.num_examples, model, pad_to_length=args.example_length)
        else:
            examples = load_examples(data_path, args.num_examples, model, length=args.example_length)
    elif args.data_type == 'hf':
        save_basename = args.dataset.replace('/', '_')
        examples = load_examples_hf(args.dataset, args.num_examples, model, length=args.example_length)

    hist_agg = HistAggregator(cfg.example_length)

    if args.data_type == 'hf':
        for i ,example in tqdm(enumerate(examples)):
            example_basename = save_basename + f"_{example[0]['document_idx']}"

            process_examples(model, embed, attns, mlps, resids, dictionaries, example_basename, example, cfg, hist_agg)
            if i >= cfg.collect_hists:
                hist_agg.save(f'{args.circuit_dir}/{save_basename}_{cfg.as_fname()}.hist.pt')
                break
    else:
        process_examples(model, embed, attns, mlps, resids, dictionaries, save_basename, examples, cfg, hist_agg)
