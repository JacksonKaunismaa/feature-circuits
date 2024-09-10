from collections import namedtuple
import torch as t
from tqdm import tqdm
from numpy import ndindex
from typing import Dict, Union
from activation_utils import SparseAct
import torch

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
    

def jvp(
        input,
        model,
        dictionaries,
        downstream_submod,
        downstream_features,
        upstream_submod,
        left_vec : Union[SparseAct, Dict[int, SparseAct]],
        right_vec : SparseAct,
        return_without_right = False,
        edge_sparsity=0.0005,
        name=None,
        shapes=None
):
    """
    Return a sparse shape [# downstream features + 1, # upstream features + 1] tensor of Jacobian-vector products.
    """

    if not downstream_features: # handle empty list
        if not return_without_right:
            return t.sparse_coo_tensor(t.zeros((2, 0), dtype=t.long), t.zeros(0)).to(model.device)
        else:
            return t.sparse_coo_tensor(t.zeros((2, 0), dtype=t.long), t.zeros(0)).to(model.device), t.sparse_coo_tensor(t.zeros((2, 0), dtype=t.long), t.zeros(0)).to(model.device)

    # first run through a test input to figure out which hidden states are tuples
    output_submods = {}
    with model.trace("_"):
        for submodule in [downstream_submod, upstream_submod]:
            output_submods[submodule] = submodule.output.save()
    
    is_tuple = {k: type(v.shape) == tuple for k, v in output_submods.items()}
        
        
    # the +1 is to deal with the fact that we concatenate the resc and the act
    n_enc = dictionaries[upstream_submod].encoder.out_features
    numel_per_batch = n_enc * input.shape[1]
    # numel_per_batch = input.shape[1] * numel_per_batch_seq
    # seq_len = input.shape[1]
    k_threshold = int(edge_sparsity * numel_per_batch)
    print('feat', len(downstream_features), input.shape[1])
    print('\tthresh', k_threshold)
    # batch_ind = torch.arange(input.shape[0]).view(-1, 1, 1) * numel_per_batch
    # seq_ind = torch.arange(seq_len).view(1, -1, 1) * numel_per_batch_seq
    # flat_index_add = batch_ind + seq_ind

    downstream_dict, upstream_dict = dictionaries[downstream_submod], dictionaries[upstream_submod]

    vjv_indices = {}
    vjv_values = {}
    if return_without_right:
        vj_indices = {}
        vj_values = {}

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

        if isinstance(left_vec, SparseAct):
            # left_vec is downstream grads (\nabla_d m)
            # to backprop is (\nabl_d m) @ d (in eq 5)
            downstream_grads_times_acts = (left_vec @ downstream_act).to_tensor()#.flatten()
            def to_backprop(feat): 
                return downstream_grads_times_acts[feat] # should be nabla_d metric @ d
        elif isinstance(left_vec, dict):
            def to_backprop(feat):
                downstream_grads_via_feat = left_vec[feat]
                # print(feat, 'act', left_vec[feat].act.nonzero().sum().item(), left_vec[feat].act.sum().item(),
                #       left_vec[feat].res.nonzero().sum().item(), left_vec[feat].res.sum().item())
                # this is (\nabla_d m * \nabla_{m_bar} d * (m_bar_patch - m_bar_clean)) @ m_bar  (in eq 6)
                downstream_grads_via_feat_times_acts = (downstream_grads_via_feat @ downstream_act)
                # sum over downstream features
                return downstream_grads_via_feat_times_acts.to_tensor().sum()

        for downstream_feat in downstream_features:
            downstream_feat = tuple(downstream_feat)
            # or in eq 6: \nabla_u {(\nabla_d m * \nabla_{m_bar} d * (m_bar_patch - m_bar_clean)) * m_bar} 
            #             * (u_patch - u_clean)
            vjv = (upstream_act.grad @ right_vec).to_tensor()#.flatten()  # eq 5 is vjv
            if return_without_right:
                vj = upstream_act.grad.to_tensor()#.flatten()
            x_res.grad = t.zeros_like(x_res)
            # when doing eq 6, this indexes into an m_bar grad? downstream_feat corresponds to d

            to_backprop(downstream_feat).backward(retain_graph=True)
            

            # vjv_ind = vjv.nonzero().squeeze(-1).save()
            # vjv_indices[downstream_feat] = vjv_ind
            # vjv_values[downstream_feat] = vjv[vjv_ind[:,0], vjv_ind[:,1], vjv_ind[:,2]].save()
            # vjv_save = vjv.save()
            vjv_topk = vjv.flatten().topk(k_threshold)
            
            nz_topk = vjv_topk.values.nonzero().squeeze(-1)  # filter out zeros
            vjv_topk_ind = vjv_topk.indices[nz_topk]
            vjv_topk_val = vjv_topk.values[nz_topk]
            
            vjv_indices[downstream_feat] = vjv_topk_ind.save()#(vjv_topk.indices * flat_index_mul)
            vjv_values[downstream_feat] = vjv_topk_val.save()
            
            if return_without_right:
                vj_ind = vj.nonzero().squeeze(-1).save()
                vj_indices[downstream_feat] = vj_ind
                vj_values[downstream_feat] = vj[vj_ind[:,0], vj_ind[:,1], vj_ind[:,2]].save()

    # construct return values   
    # print(vjv_indices[downstream_feat])


    ## get shapes
    d_downstream_contracted = ((downstream_act.value @ downstream_act.value).to_tensor()).shape
    d_upstream_contracted = ((upstream_act.value @ upstream_act.value).to_tensor()).shape
    if return_without_right:
        d_upstream = (upstream_act.value.to_tensor()).shape
        
    if shapes is not None:
        d_downstream_contracted, d_upstream_contracted = shapes[:3], shapes[3:]
    
    print('\tnnz', sum([vjv_indices[tuple(downstream_feat)].shape[0] for downstream_feat in downstream_features]))
    ## make tensors
    downstream_indices = t.tensor([downstream_feat for downstream_feat in downstream_features 
                                for _ in vjv_indices[tuple(downstream_feat)].value], device=model.device).T
    # print(downstream_indices.shape)
    upstream_indices = t.cat([t.stack(t.unravel_index(vjv_indices[tuple(downstream_feat)].value, d_upstream_contracted), dim=1)
                              for downstream_feat in downstream_features], dim=0).T
    # print(downstream_indices.shape, upstream_indices.shape)
    vjv_indices = t.cat([downstream_indices, upstream_indices], dim=0).to(model.device)
    vjv_values = t.cat([vjv_values[tuple(downstream_feat)].value for downstream_feat in downstream_features], dim=0)
    # print(vjv_values.shape, upstream_act.value.shape, d_downstream_contracted, d_upstream_contracted, d_upstream,
    #       upstream_act.value.res.shape, upstream_act.value.act.shape)
    if not return_without_right:
        return t.sparse_coo_tensor(vjv_indices, vjv_values, (*d_downstream_contracted, *d_upstream_contracted))
        
    
    # vj_indices = t.tensor(
    #     [[downstream_feat for downstream_feat in downstream_features for _ in vj_indices[downstream_feat].value],
    #     t.cat([vj_indices[downstream_feat].value for downstream_feat in downstream_features], dim=0)],
    # device=model.device)
    # 1+1
    # vj_values = t.cat([vj_values[downstream_feat].value for downstream_feat in downstream_features], dim=0)

    return (
        t.sparse_coo_tensor(vjv_indices, vjv_values, (*d_downstream_contracted, *d_upstream_contracted)),
        # t.sparse_coo_tensor(vj_indices, vj_values, (d_downstream_contracted, d_upstream))
        # (vjv_indices, vjv_values, (d_downstream_contracted, d_upstream_contracted)),
        (vj_indices, vj_values, (d_downstream_contracted, d_upstream))
    )