from collections import namedtuple
import torch as t
from tqdm import tqdm
from numpy import ndindex
from typing import Dict, Union
import torch

from config import Config
from histogram_aggregator import HistAggregator
from activation_utils import SparseAct

DEBUGGING = False

if DEBUGGING:
    tracer_kwargs = {'validate' : True, 'scan' : True}
else:
    tracer_kwargs = {'validate' : False, 'scan' : False}

EffectOut = namedtuple('EffectOut', ['effects', 'deltas', 'grads', 'total_effect'])

def _pe_attrib(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        metric_kwargs=dict(),
):

    # first run through a test input to figure out which hidden states are tuples
    output_submods = {}
    with model.trace("_"):
        for submodule in submodules:
            output_submods[submodule] = submodule.output.save()

    is_tuple = {k: type(v.shape) == tuple for k, v in output_submods.items()}

    hidden_states_clean = {}
    grads = {}
    with model.trace(clean, **tracer_kwargs):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            x_hat, f = dictionary(x, output_features=True) # x_hat implicitly depends on f
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=f, res=residual).save()
            grads[submodule] = hidden_states_clean[submodule].grad.save()
            residual.grad = t.zeros_like(residual)
            x_recon = x_hat + residual
            if is_tuple[submodule]:
                submodule.output[0][:] = x_recon
            else:
                submodule.output = x_recon
            x.grad = x_recon.grad
        metric_clean = metric_fn(model, **metric_kwargs).save()
        metric_clean.sum().backward()
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}
    grads = {k : v.value for k, v in grads.items()}

    if patch is None:
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with model.trace(patch, **tracer_kwargs), t.inference_mode():
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.output
                if is_tuple[submodule]:
                    x = x[0]
                x_hat, f = dictionary(x, output_features=True)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(act=f, res=residual).save()
            metric_patch = metric_fn(model, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()
        hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    with torch.no_grad():
        for submodule in submodules:
            patch_state, clean_state, grad = hidden_states_patch[submodule], hidden_states_clean[submodule], grads[submodule]
            delta = patch_state - clean_state.detach() if patch_state is not None else -clean_state.detach()
            # print("delta", delta.shape, 'grad', grad.shape)
            effect = delta @ grad  # this is just elementwise product for activations, and something weird for err
            # print("effect for", submodule, effect.shape)  # for SAE errors
            effects[submodule] = effect
            deltas[submodule] = delta
            grads[submodule] = grad
        total_effect = total_effect if total_effect is not None else None

    return EffectOut(effects, deltas, grads, total_effect)

def _pe_ig(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        steps=10,
        metric_kwargs=dict(),
):

    # first run through a test input to figure out which hidden states are tuples
    output_submods = {}
    with model.trace("_"):
        for submodule in submodules:
            output_submods[submodule] = submodule.output.save()

    is_tuple = {k: type(v.shape) == tuple for k, v in output_submods.items()}

    hidden_states_clean = {}
    with model.trace(clean, **tracer_kwargs), t.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=f.save(), res=residual.save())
        metric_clean = metric_fn(model, **metric_kwargs).save()
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}

    if patch is None:
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with model.trace(patch, **tracer_kwargs), t.no_grad():
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.output
                if is_tuple[submodule]:
                    x = x[0]
                f = dictionary.encode(x)
                x_hat = dictionary.decode(f)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(act=f.save(), res=residual.save())
            metric_patch = metric_fn(model, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()
        hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    grads = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
        with model.trace(**tracer_kwargs) as tracer:
            metrics = []
            fs = []
            for step in range(steps):
                alpha = step / steps
                f = (1 - alpha) * clean_state + alpha * patch_state
                f.act.retain_grad()
                f.res.retain_grad()
                fs.append(f)
                with tracer.invoke(clean, scan=tracer_kwargs['scan']):
                    if is_tuple[submodule]:
                        submodule.output[0][:] = dictionary.decode(f.act) + f.res
                    else:
                        submodule.output = dictionary.decode(f.act) + f.res
                    metrics.append(metric_fn(model, **metric_kwargs))
            metric = sum([m for m in metrics])
            metric.sum().backward(retain_graph=True) # TODO : why is this necessary? Probably shouldn't be, contact jaden

        mean_grad = sum([f.act.grad for f in fs]) / steps
        mean_residual_grad = sum([f.res.grad for f in fs]) / steps
        grad = SparseAct(act=mean_grad, res=mean_residual_grad)
        delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
        effect = grad @ delta

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad

    return EffectOut(effects, deltas, grads, total_effect)


def _pe_exact(
    clean,
    patch,
    model,
    submodules,
    dictionaries,
    metric_fn,
    ):

    # first run through a test input to figure out which hidden states are tuples
    output_submods = {}
    with model.trace("_"):
        for submodule in submodules:
            output_submods[submodule] = submodule.output.save()

    is_tuple = {k: type(v.shape) == tuple for k, v in output_submods.items()}

    hidden_states_clean = {}
    with model.trace(clean, **tracer_kwargs), t.inference_mode():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=f, res=residual).save()
        metric_clean = metric_fn(model).save()
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}

    if patch is None:
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with model.trace(patch, **tracer_kwargs), t.inference_mode():
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.output
                if is_tuple[submodule]:
                    x = x[0]
                f = dictionary.encode(x)
                x_hat = dictionary.decode(f)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(act=f, res=residual).save()
            metric_patch = metric_fn(model).save()
        total_effect = metric_patch.value - metric_clean.value
        hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
        effect = SparseAct(act=t.zeros_like(clean_state.act), resc=t.zeros(*clean_state.res.shape[:-1])).to(model.device)

        # iterate over positions and features for which clean and patch differ
        idxs = t.nonzero(patch_state.act - clean_state.act)
        for idx in tqdm(idxs):
            with t.inference_mode():
                with model.trace(clean, **tracer_kwargs):
                    f = clean_state.act.clone()
                    f[tuple(idx)] = patch_state.act[tuple(idx)]
                    x_hat = dictionary.decode(f)
                    if is_tuple[submodule]:
                        submodule.output[0][:] = x_hat + clean_state.res
                    else:
                        submodule.output = x_hat + clean_state.res
                    metric = metric_fn(model).save()
                effect.act[tuple(idx)] = (metric.value - metric_clean.value).sum()

        for idx in list(ndindex(effect.resc.shape)):
            with t.inference_mode():
                with model.trace(clean, **tracer_kwargs):
                    res = clean_state.res.clone()
                    res[tuple(idx)] = patch_state.res[tuple(idx)]
                    x_hat = dictionary.decode(clean_state.act)
                    if is_tuple[submodule]:
                        submodule.output[0][:] = x_hat + res
                    else:
                        submodule.output = x_hat + res
                    metric = metric_fn(model).save()
                effect.resc[tuple(idx)] = (metric.value - metric_clean.value).sum()

        effects[submodule] = effect
        deltas[submodule] = patch_state - clean_state
    total_effect = total_effect if total_effect is not None else None

    return EffectOut(effects, deltas, None, total_effect)


def patching_effect(
        clean,
        patch,
        model,
        submodules,
        dictionaries,
        metric_fn,
        method='attrib',
        steps=10,
        metric_kwargs=dict()
):
    if method == 'attrib':
        return _pe_attrib(clean, patch, model, submodules, dictionaries, metric_fn, metric_kwargs=metric_kwargs)
    elif method == 'ig':
        return _pe_ig(clean, patch, model, submodules, dictionaries, metric_fn, steps=steps, metric_kwargs=metric_kwargs)
    elif method == 'exact':
        return _pe_exact(clean, patch, model, submodules, dictionaries, metric_fn)
    else:
        raise ValueError(f"Unknown method {method}")


def threshold_effects(effect: t.Tensor, threshold: float, as_threshold: bool, stack=True, k_sparsity: int | None =None):
    """
    Return the indices of the top-k features with the highest absolute effect, or if as_threshold is True, the indices
    of the features with absolute effect greater than threshold.

    Args:
        effect: tensor to apply threshold to
        threshold: threshold value (or sparsity value)
        as_threshold: whether to treat `threshold` as a threshold or a sparsity value
        stack: whether to stack the indices into a tensor. Only has effect if as_threshold is False
        k_sparsity: if not None, the number of features to return (otherwise, calculated from `effect` and `threshold`)

    Returns:
        if stack == False:
            indices: indices of the top-k features
            values: values of the top-k
        else:
            indices: indices of the top-k features, stacked into a tensor, and then .tolist()'d
    """
    if as_threshold:
        ind = t.nonzero(effect.flatten().abs() > threshold).flatten()
        if stack:
            return t.stack(t.unravel_index(ind, effect.shape), dim=1).tolist()
        return ind, effect.flatten()[ind]

    else:  # as_sparsity
        if k_sparsity is None:
            n_features = effect.numel()
            k_sparsity = int(threshold * n_features)
        topk = effect.abs().flatten().topk(k_sparsity)
        topk_ind = topk.indices[topk.values > 0]
        if stack:
            return t.stack(t.unravel_index(topk_ind, effect.shape), dim=1).tolist()
        return topk_ind, topk.values[topk.values > 0]


def jvp(
        input,
        model,
        dictionaries,
        downstream_submod,
        downstream_features,
        upstream_submod,
        left_vec : Union[SparseAct, Dict[int, SparseAct]],
        right_vec : SparseAct,
        cfg: Config,
        hist_agg: HistAggregator,
        intermediate_stop_grads=None,
):
    """
    Return a sparse shape [# downstream features + 1, # upstream features + 1] tensor of Jacobian-vector products.
    """

    if intermediate_stop_grads is None:
        intermediate_stop_grads = []

    if not downstream_features: # handle empty list
        return t.sparse_coo_tensor(t.zeros((6, 0), dtype=t.long), t.zeros(0)).to(model.device)

    # first run through a test input to figure out which hidden states are tuples
    output_submods = {}
    with model.trace("_"):
        for submodule in [downstream_submod, upstream_submod] + intermediate_stop_grads:
            output_submods[submodule] = submodule.output.save()

    is_tuple = {k: type(v.shape) == tuple for k, v in output_submods.items()}


    if cfg.as_threshold:
        k_sparsity = None
    else:
        n_enc = dictionaries[upstream_submod].encoder.out_features
        numel_per_batch = n_enc * input.shape[1]
        k_sparsity = int(cfg.edge_sparsity * numel_per_batch)
        print('feat', len(downstream_features), input.shape[1])
        print('\tthresh', k_sparsity)

    downstream_dict, upstream_dict = dictionaries[downstream_submod], dictionaries[upstream_submod]

    vjv_indices = {}
    vjv_values = {}

    with model.trace(input, **tracer_kwargs):
        # first specify forward pass modifications
        x = upstream_submod.output.save()
        if is_tuple[upstream_submod]:
            x = x[0]
        x_hat, f = upstream_dict(x, output_features=True)
        x_res = x - x_hat
        upstream_act = SparseAct(act=f, res=x_res).save()
        if is_tuple[upstream_submod]:
            upstream_submod.output[0][:] = x_hat + x_res
        else:
            upstream_submod.output = x_hat + x_res
        y = downstream_submod.output
        if is_tuple[downstream_submod]:
            y = y[0]
        y_hat, g = downstream_dict(y, output_features=True)
        y_res = y - y_hat
        downstream_act = SparseAct(act=g, res=y_res).save()

        to_backprops = (left_vec @ downstream_act).to_tensor()#.flatten()

        for downstream_feat in downstream_features:
            downstream_feat = tuple(downstream_feat)
            for submod in intermediate_stop_grads:
                if is_tuple[submod]:
                    submod.output[0].grad = t.zeros_like(submod.output[0])
                else:
                    submod.output.grad = t.zeros_like(submod.output)

            x_res.grad = t.zeros_like(x_res)

            vjv = (upstream_act.grad @ right_vec).to_tensor() # eq 5 is vjv
            to_backprops[downstream_feat].backward(retain_graph=True)

            # if upstream_submod._module_path == '.gpt_neox.layers.5.attention' and downstream_submod._module_path == '.gpt_neox.layers.5.mlp':
            #     to_backprops = to_backprops.save()
            #     upstream_act = upstream_act.save()
            #     upstream_grad = upstream_act.grad.save()
            #     vjv_saved = (upstream_act.grad @ right_vec).save()
            #     vjv = vjv.save()
            #     break

            if cfg.collect_hists > 0:
                hist_agg.trace_edge_hist(upstream_submod, downstream_submod, vjv)


            vjv_ind, vjv_val = threshold_effects(vjv, cfg.edge_sparsity,
                                                 cfg.as_threshold,
                                                 k_sparsity=k_sparsity,
                                                 stack=False)

            vjv_indices[downstream_feat] = vjv_ind.save()#(vjv_topk.indices * flat_index_mul)
            vjv_values[downstream_feat] = vjv_val.save()

    # construct return values

    ## get shapes
    d_downstream_contracted = ((downstream_act.value @ downstream_act.value).to_tensor()).shape
    d_upstream_contracted = ((upstream_act.value @ upstream_act.value).to_tensor()).shape

    if cfg.collect_hists > 0:
        hist_agg.aggregate_edge_hist(upstream_submod, downstream_submod)

    print('\tnnz', sum(vjv_indices[tuple(downstream_feat)].shape[0] for downstream_feat in downstream_features))
    ## make tensors
    downstream_indices = t.tensor([downstream_feat for downstream_feat in downstream_features
                                for _ in vjv_indices[tuple(downstream_feat)].value], device=model.device).T
    upstream_indices = t.cat([t.stack(t.unravel_index(vjv_indices[tuple(downstream_feat)].value, d_upstream_contracted), dim=1)
                              for downstream_feat in downstream_features], dim=0).T
    vjv_indices = t.cat([downstream_indices, upstream_indices], dim=0).to(model.device)
    vjv_values = t.cat([vjv_values[tuple(downstream_feat)].value for downstream_feat in downstream_features], dim=0)

    return t.sparse_coo_tensor(vjv_indices, vjv_values, (*d_downstream_contracted, *d_upstream_contracted))