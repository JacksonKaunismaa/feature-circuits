import argparse
import gc
import json
import math
import os
from collections import defaultdict

import torch as t
from einops import rearrange
from tqdm import tqdm

from activation_utils import SparseAct
from attribution import patching_effect, jvp
from circuit_plotting import plot_circuit, plot_circuit_posaligned
from dictionary_learning import AutoEncoder
from loading_utils import load_examples, load_examples_nopair
from nnsight import LanguageModel


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


def get_circuit(
        clean,
        patch,
        model,
        embed,
        attns,
        mlps,
        resids,
        dictionaries,
        metric_fn,
        metric_kwargs=dict(),
        aggregation='sum', # or 'none' for not aggregating across sequence position
        nodes_only=False,
        node_threshold=0.1,
        edge_threshold=0.01,
):
    
    all_submods = ([embed] if embed is not None else []) + [submod for layer_submods in zip(mlps, attns, resids) for submod in layer_submods]
    
    # first get the patching effect of everything on y
    effects, deltas, grads, total_effect = patching_effect(
        clean,
        patch,
        model,
        all_submods,
        dictionaries,
        metric_fn,
        metric_kwargs=metric_kwargs,
        method='attrib' # get better approximations for early layers by using ig
    )

    print('effects done!')
    def unflatten(tensor): # will break if dictionaries vary in size between layers
        b, s, f = effects[resids[0]].act.shape
        unflattened = rearrange(tensor, '(b s x) -> b s x', b=b, s=s)
        # MR_grad is artifically given extra size by jvp, but then here we subselect
        # so that act includes only the features, and res includes the contracted residuals
        # we have 1996800 = (10*6*32768)=1966080 (features) + (10*6*512) (resids) + (10*6) (contracted resids) 
        #                   + 60 again for some reason (considered part of "features" according to b,s,f =act.shape)
        # when we unflatten, act ends up getting (10*6*32768) features, and res gets just the 60 (contracted)
        return SparseAct(act=unflattened[...,:f], res=unflattened[...,f:])
    
    def unflatten2(grad, feat_idx):
        # (vj_indices, vj_values, (d_downstream_contracted, d_upstream))
        b, s, f = effects[resids[0]].act.shape
        indices = grad[0][feat_idx].to(model.device).unsqueeze(0)
        values = grad[1][feat_idx].to(model.device)
        shape = (grad[2][1],)
        tensor = t.sparse_coo_tensor(indices, values, shape, is_coalesced=True).to_dense()
        unflattened = rearrange(tensor, '(b s x) -> b s x', b=b, s=s)
        return SparseAct(act=unflattened[...,:f], res=unflattened[...,f:])
    
    features_by_submod = {
        submod : (effects[submod].to_tensor().flatten().abs() > node_threshold).nonzero().flatten().tolist() for submod in all_submods
    }  # submodule -> list of indices

    n_layers = len(resids)

    nodes = {'y' : total_effect}
    if embed is not None:
        nodes['embed'] = effects[embed]
    for i in range(n_layers):
        nodes[f'attn_{i}'] = effects[attns[i]]
        nodes[f'mlp_{i}'] = effects[mlps[i]]
        nodes[f'resid_{i}'] = effects[resids[i]]

    if nodes_only:
        if aggregation == 'sum':
            for k in nodes:
                if k != 'y':
                    nodes[k] = nodes[k].sum(dim=1)
        nodes = {k : v.mean(dim=0) for k, v in nodes.items()}
        return nodes, None

    edges = defaultdict(lambda:{})
    edges[f'resid_{len(resids)-1}'] = { 'y' : effects[resids[-1]].to_tensor().flatten().to_sparse() }

    def N(upstream, downstream, return_without_right=True):
        return jvp(
            clean,
            model,
            dictionaries,
            downstream,  # downstream submod
            features_by_submod[downstream],  # downstream features
            upstream,   # upstream submod
            grads[downstream],  # left_vec
            deltas[upstream],   # right_vec
            return_without_right=return_without_right,
        )


    # now we work backward through the model to get the edges
    # max_abs = 0.0
    for layer in reversed(range(len(resids))):
        resid = resids[layer]
        mlp = mlps[layer]
        attn = attns[layer]

        MR_effect, MR_grad = N(mlp, resid)
        print("MR done")
        AR_effect, AR_grad = N(attn, resid)
        print("AR done")
        # print(abs(MR_effect - MR_grad).sum())

        edges[f'mlp_{layer}'][f'resid_{layer}'] = MR_effect
        edges[f'attn_{layer}'][f'resid_{layer}'] = AR_effect

        if layer > 0:
            prev_resid = resids[layer-1]
        else:
            if embed is not None:
                prev_resid = embed
            else:
                continue

        RM_effect = N(prev_resid, mlp, return_without_right=False)
        print("RM done")
        RA_effect = N(prev_resid, attn, return_without_right=False)
        print("RA done")

        # MR_grad = MR_grad.coalesce()
        # AR_grad = AR_grad.coalesce()
        # print("coalescede!")
        # print({feat_idx : unflatten(MR_grad[feat_idx].to_dense()).shape for feat_idx in features_by_submod[resid]})

        # features_by_submod[resid] -> 172 indices in the n+1 residual
        # 171 indices are in the range [0, 10*6*32768=1966080], one index is 1966139
        # MR_grad is a jacobian of (indices in resid) x (indices in mlp)
        # MR_grad   is [1966140, 1996800], 
        # MR_effect is [1966140, 1966140]
        # 1996800 = 10*6*(32768+512)
        # 1966140 = 10*6*32768 + 60 or 10*6*(32768 + 1)
        # when we index MR_grad with one of the feat_idxs, we select all the edges from 
        # the mlp to that particular resid node that contributed to that resid node
        # ie. its \nabla_{mlp} resid_i = left_vec
        # we then unflatten which takes us back into a SparseAct, which will have
        # MR_grad gives shape of 10 x 6 x 512 for res
        # MR_effect gives shape of 10 x 6 x 1 for res
        # both have same shape of acts of 10 x 6 x 32768
        # they have the same elements, but they get shuffled around differently, since
        # res.sum() + act.sum() is the same for both, but res.sum() and act.sum() are different
        # reason: MR_effect and MR_grad have the exact same values and indices. However, since
        # they have different shape parameters, when we unflatten them, the values get mangled a bit
        # and things go to res that shouldn't go to res. 
        # if b=1, we have s*(f+1) entries of actual data, and then MR_grad expands it out to s*(f+e),
        # so we append s*(e-1) zeros to the end of the memory block,
        # like this: A B C D E F G H I J K L M N O P 0 0 0 0 (b=1, s=4, f=3, e=2)
        # grad:   [[A, B, C], [F, G, H], [K, L, M], [P, 0, 0]]=acts & [[D, E], [I, J], [N, O], [0, 0]]=res
        # effect: [[A, B, C], [E, F, G], [I, J, K], [M, N, O]]=acts & [[D], [H], [L], [P]]=res
        # so things get rearranged badly past the first sequence position


        RMR_effect = jvp(
            clean,     # input
            model,     # model 
            dictionaries,   # dictionaries
            mlp,  # downstream submod
            features_by_submod[resid],  # downstream features
            prev_resid,   # upstream submod
            {feat_idx : unflatten2(MR_grad, feat_idx) for feat_idx in features_by_submod[resid]}, # left_vec
            deltas[prev_resid],    # right_vec
        )
        print("RMR done")
        RAR_effect = jvp(
            clean,
            model,
            dictionaries,
            attn,
            features_by_submod[resid],
            prev_resid,
            {feat_idx : unflatten2(AR_grad, feat_idx) for feat_idx in features_by_submod[resid]},
            deltas[prev_resid],
        )
        print("RAR done!")

        # max_abs += abs(RMR_effect).sum() + abs(RAR_effect).sum()
        RR_effect = N(prev_resid, resid, return_without_right=False)

        if layer > 0:    # for the future: infer this from a graph structure
            edges[f'resid_{layer-1}'][f'mlp_{layer}'] = RM_effect
            edges[f'resid_{layer-1}'][f'attn_{layer}'] = RA_effect
            edges[f'resid_{layer-1}'][f'resid_{layer}'] = RR_effect - RMR_effect - RAR_effect
        else:
            edges['embed'][f'mlp_{layer}'] = RM_effect
            edges['embed'][f'attn_{layer}'] = RA_effect
            edges['embed'][f'resid_0'] = RR_effect - RMR_effect - RAR_effect
        print('layer complete!', layer)
    # if max_abs > 0:
    #     print("Detected non-zero RMR effects!")
    # rearrange weight matrices
    for child in edges:
        # get shape for child
        bc, sc, fc = nodes[child].act.shape
        for parent in edges[child]:
            weight_matrix = edges[child][parent]
            if parent == 'y':
                weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc+1))
            else:
                bp, sp, fp = nodes[parent].act.shape
                assert bp == bc
                weight_matrix = sparse_reshape(weight_matrix, (bp, sp, fp+1, bc, sc, fc+1))
            edges[child][parent] = weight_matrix
    
    if aggregation == 'sum':
        # aggregate across sequence position
        for child in edges:
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = weight_matrix.sum(dim=1)
                else:
                    weight_matrix = weight_matrix.sum(dim=(1, 4))
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != 'y':
                nodes[node] = nodes[node].sum(dim=1)

        # aggregate across batch dimension
        for child in edges:
            bc, fc = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = weight_matrix.sum(dim=0) / bc
                else:
                    bp, fp = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = weight_matrix.sum(dim=(0,2)) / bc
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != 'y':
                nodes[node] = nodes[node].mean(dim=0)
    
    elif aggregation == 'none':

        # aggregate across batch dimensions
        for child in edges:
            # get shape for child
            bc, sc, fc = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc+1))
                    weight_matrix = weight_matrix.sum(dim=0) / bc
                else:
                    bp, sp, fp = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = sparse_reshape(weight_matrix, (bp, sp, fp+1, bc, sc, fc+1))
                    weight_matrix = weight_matrix.sum(dim=(0, 3)) / bc
                edges[child][parent] = weight_matrix
        for node in nodes:
            nodes[node] = nodes[node].mean(dim=0)

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return nodes, edges


def get_circuit_cluster(dataset,
                        model_name="EleutherAI/pythia-70m-deduped",
                        d_model=512,
                        dict_id=10,
                        dict_size=32768,
                        max_length=64,
                        max_examples=100,
                        batch_size=2,
                        node_threshold=0.1,
                        edge_threshold=0.01,
                        device="cuda:0",
                        dict_path="dictionaries/pythia-70m-deduped/",
                        dataset_name="cluster_circuit",
                        circuit_dir="circuits/",
                        plot_dir="circuits/figures/",
                        model=None,
                        dictionaries=None,):
    
    model = LanguageModel(model_name, device_map=device, dispatch=True)

    embed = model.gpt_neox.embed_in
    attns = [layer.attention for layer in model.gpt_neox.layers]
    mlps = [layer.mlp for layer in model.gpt_neox.layers]
    resids = [layer for layer in model.gpt_neox.layers]
    dictionaries = {}
    dictionaries[embed] = AutoEncoder.from_pretrained(
        os.path.join(dict_path, f'embed/{dict_id}_{dict_size}/ae.pt'),
        device=device
    )
    for i in range(len(model.gpt_neox.layers)):
        dictionaries[attns[i]] = AutoEncoder.from_pretrained(
            os.path.join(dict_path, f'attn_out_layer{i}/{dict_id}_{dict_size}/ae.pt'),
            device=device
        )
        dictionaries[mlps[i]] = AutoEncoder.from_pretrained(
            os.path.join(dict_path, f'mlp_out_layer{i}/{dict_id}_{dict_size}/ae.pt'),
            device=device
        )
        dictionaries[resids[i]] = AutoEncoder.from_pretrained(
            os.path.join(dict_path, f'resid_out_layer{i}/{dict_id}_{dict_size}/ae.pt'),
            device=device
        )

    examples = load_examples_nopair(dataset, max_examples, model, length=max_length)

    num_examples = min(len(examples), max_examples)
    n_batches = math.ceil(num_examples / batch_size)
    batches = [
        examples[batch*batch_size:(batch+1)*batch_size] for batch in range(n_batches)
    ]
    if num_examples < max_examples: # warn the user
        print(f"Total number of examples is less than {max_examples}. Using {num_examples} examples instead.")

    running_nodes = None
    running_edges = None

    for batch in tqdm(batches, desc="Batches"):
        clean_inputs = t.cat([e['clean_prefix'] for e in batch], dim=0).to(device)
        clean_answer_idxs = t.tensor([e['clean_answer'] for e in batch], dtype=t.long, device=device)

        patch_inputs = None
        def metric_fn(model):
            return (
                -1 * t.gather(
                    t.nn.functional.log_softmax(model.embed_out.output[:,-1,:], dim=-1), dim=-1, index=clean_answer_idxs.view(-1, 1)
                ).squeeze(-1)
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
            metric_fn,
            aggregation="sum",
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )

        if running_nodes is None:
            running_nodes = {k : len(batch) * nodes[k].to('cpu') for k in nodes.keys() if k != 'y'}
            running_edges = { k : { kk : len(batch) * edges[k][kk].to('cpu') for kk in edges[k].keys() } for k in edges.keys()}
        else:
            for k in nodes.keys():
                if k != 'y':
                    running_nodes[k] += len(batch) * nodes[k].to('cpu')
            for k in edges.keys():
                for v in edges[k].keys():
                    running_edges[k][v] += len(batch) * edges[k][v].to('cpu')
        
        # memory cleanup
        del nodes, edges
        gc.collect()

    nodes = {k : v.to(device) / num_examples for k, v in running_nodes.items()}
    edges = {k : {kk : 1/num_examples * v.to(device) for kk, v in running_edges[k].items()} for k in running_edges.keys()}

    save_dict = {
        "examples" : examples,
        "nodes": nodes,
        "edges": edges
    }
    save_basename = f"{dataset_name}_dict{dict_id}_node{node_threshold}_edge{edge_threshold}_n{num_examples}_aggsum"
    with open(f'{circuit_dir}/{save_basename}.pt', 'wb') as outfile:
        t.save(save_dict, outfile)

    nodes = save_dict['nodes']
    edges = save_dict['edges']

    # feature annotations
    try:
        annotations = {}
        with open(f'annotations/{dict_id}_{dict_size}.jsonl', 'r') as f:
            for line in f:
                line = json.loads(line)
                annotations[line['Name']] = line['Annotation']
    except:
        annotations = None

    plot_circuit(
        nodes, 
        edges, 
        layers=len(model.gpt_neox.layers), 
        node_threshold=node_threshold, 
        edge_threshold=edge_threshold, 
        pen_thickness=1, 
        annotations=annotations, 
        save_dir=os.path.join(plot_dir, save_basename))


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
    parser.add_argument('--node_threshold', type=float, default=0.2,
                        help="Indirect effect threshold for keeping circuit nodes.")
    parser.add_argument('--edge_threshold', type=float, default=0.02,
                        help="Indirect effect threshold for keeping edges.")
    parser.add_argument('--pen_thickness', type=float, default=1,
                        help="Scales the width of the edges in the circuit plot.")
    parser.add_argument('--nopair', default=False, action="store_true",
                        help="Use if your data does not contain contrastive (minimal) pairs.")
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


    device = args.device

    model = LanguageModel(args.model, device_map=device, dispatch=True)

    if args.model != 'gpt2':
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
    if args.dict_id == 'id':
        from dictionary_learning.dictionary import IdentityDict
        dictionaries[embed] = IdentityDict(args.d_model)
        for i in range(len(model.gpt_neox.layers)):
            dictionaries[attns[i]] = IdentityDict(args.d_model)
            dictionaries[mlps[i]] = IdentityDict(args.d_model)
            dictionaries[resids[i]] = IdentityDict(args.d_model)
    elif args.dict_id == 'gpt':
        for i in range(len(model.transformer.h)):
            dictionaries[attns[i]] = AutoEncoder.from_saelens(
                "gpt2-small-hook-z-kk",
                f"blocks.{i}.hook_z",
                device=device
            )
            dictionaries[mlps[i]] = AutoEncoder.from_saelens(
                "gpt2-small-mlp-tm",
                f"blocks.{i}.hook_mlp_out",
                device=device
            )
            dictionaries[resids[i]] = AutoEncoder.from_saelens(
                "gpt2-small-res-jb",
                f"blocks.{i}.hook_resid_pre",
                device=device
            )

    else:
        dictionaries[embed] = AutoEncoder.from_pretrained(
            f'{args.dict_path}/embed/{args.dict_id}_{args.dict_size}/ae.pt',
            device=device
        )
        for i in range(len(model.gpt_neox.layers)):
            dictionaries[attns[i]] = AutoEncoder.from_pretrained(
                f'{args.dict_path}/attn_out_layer{i}/{args.dict_id}_{args.dict_size}/ae.pt',
                device=device
            )
            dictionaries[mlps[i]] = AutoEncoder.from_pretrained(
                f'{args.dict_path}/mlp_out_layer{i}/{args.dict_id}_{args.dict_size}/ae.pt',
                device=device
            )
            dictionaries[resids[i]] = AutoEncoder.from_pretrained(
                f'{args.dict_path}/resid_out_layer{i}/{args.dict_id}_{args.dict_size}/ae.pt',
                device=device
            )
        print(dictionaries.keys())
    
    if args.nopair:
        save_basename = os.path.splitext(os.path.basename(args.dataset))[0]
        examples = load_examples_nopair(args.dataset, args.num_examples, model, length=args.example_length)
    else:
        data_path = f"data/{args.dataset}.json"
        save_basename = args.dataset
        if args.aggregation == "sum":
            examples = load_examples(data_path, args.num_examples, model, pad_to_length=args.example_length)
        else:
            examples = load_examples(data_path, args.num_examples, model, length=args.example_length)
    
    batch_size = args.batch_size
    num_examples = min([args.num_examples, len(examples)])
    n_batches = math.ceil(num_examples / batch_size)
    batches = [
        examples[batch*batch_size:(batch+1)*batch_size] for batch in range(n_batches)
    ]
    if num_examples < args.num_examples: # warn the user
        print(f"Total number of examples is less than {args.num_examples}. Using {num_examples} examples instead.")

    if not args.plot_only:
        running_nodes = None
        running_edges = None

        for batch in tqdm(batches, desc="Batches"):
                
            clean_inputs = t.cat([e['clean_prefix'] for e in batch], dim=0).to(device)
            clean_answer_idxs = t.tensor([e['clean_answer'] for e in batch], dtype=t.long, device=device)
            if args.model == 'gpt2':
                model_out = model.lm_head
            else:
                model_out = model.embed_out

            if args.nopair:
                patch_inputs = None
                def metric_fn(model):
                    return (
                        -1 * t.gather(
                            t.nn.functional.log_softmax(model_out.output[:,-1,:], dim=-1), dim=-1, index=clean_answer_idxs.view(-1, 1)
                        ).squeeze(-1)
                    )
            else:
                patch_inputs = t.cat([e['patch_prefix'] for e in batch], dim=0).to(device)
                patch_answer_idxs = t.tensor([e['patch_answer'] for e in batch], dtype=t.long, device=device)
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
                metric_fn,
                nodes_only=args.nodes_only,
                aggregation=args.aggregation,
                node_threshold=args.node_threshold,
                edge_threshold=args.edge_threshold,
            )

            if running_nodes is None:
                running_nodes = {k : len(batch) * nodes[k].to('cpu') for k in nodes.keys() if k != 'y'}
                if not args.nodes_only: running_edges = { k : { kk : len(batch) * edges[k][kk].to('cpu') for kk in edges[k].keys() } for k in edges.keys()}
            else:
                for k in nodes.keys():
                    if k != 'y':
                        running_nodes[k] += len(batch) * nodes[k].to('cpu')
                if not args.nodes_only:
                    for k in edges.keys():
                        for v in edges[k].keys():
                            running_edges[k][v] += len(batch) * edges[k][v].to('cpu')
            
            # memory cleanup
            del nodes, edges
            gc.collect()

        nodes = {k : v.to(device) / num_examples for k, v in running_nodes.items()}
        if not args.nodes_only: 
            edges = {k : {kk : 1/num_examples * v.to(device) for kk, v in running_edges[k].items()} for k in running_edges.keys()}
        else: edges = None

        save_dict = {
            "examples" : examples,
            "nodes": nodes,
            "edges": edges
        }
        with open(f'{args.circuit_dir}/{save_basename}_dict{args.dict_id}_node{args.node_threshold}_edge{args.edge_threshold}_n{num_examples}_agg{args.aggregation}.pt', 'wb') as outfile:
            t.save(save_dict, outfile)

    else:
        with open(f'{args.circuit_dir}/{save_basename}_dict{args.dict_id}_node{args.node_threshold}_edge{args.edge_threshold}_n{num_examples}_agg{args.aggregation}.pt', 'rb') as infile:
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
    except:
        annotations = None

    if args.aggregation == "none":
        example = model.tokenizer.batch_decode(examples[0]["clean_prefix"])[0]
        plot_circuit_posaligned(
            nodes, 
            edges,
            layers=len(model.gpt_neox.layers), 
            length=args.example_length,
            example_text=example,
            node_threshold=args.node_threshold, 
            edge_threshold=args.edge_threshold, 
            pen_thickness=args.pen_thickness, 
            annotations=annotations, 
            save_dir=f'{args.plot_dir}/{save_basename}_dict{args.dict_id}_node{args.node_threshold}_edge{args.edge_threshold}_n{num_examples}_agg{args.aggregation}'
        )
    else:
        plot_circuit(
            nodes, 
            edges, 
            layers=len(model.gpt_neox.layers), 
            node_threshold=args.node_threshold, 
            edge_threshold=args.edge_threshold, 
            pen_thickness=args.pen_thickness, 
            annotations=annotations, 
            save_dir=f'{args.plot_dir}/{save_basename}_dict{args.dict_id}_node{args.node_threshold}_edge{args.edge_threshold}_n{num_examples}_agg{args.aggregation}'
        )