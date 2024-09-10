from functools import partial
from graphviz import Digraph
from collections import defaultdict
import re
import os
from tqdm import trange
import torch as t
import numpy as np
import networkx as nx

def get_name(component, layer, idx, err_idx, add_layer=True):
    if add_layer:
        component = f"{component}_{layer}"
    match idx:
        case (seq, feat):
            if feat == err_idx: feat = 'ε'
            if layer == -1: return f'{seq}, embed/{feat}'
            return f'{seq}, {component}/{feat}'
        case (feat,):
            if feat == err_idx: feat = 'ε'
            if layer == -1: return f'embed/{feat}'
            return f'{component}/{feat}'
        case _: raise ValueError(f"Invalid idx: {idx}")
        
        
def normalize_weight(weight, edge_scale, pen_thickness, normalization='linear'):
    if normalization == 'linear':
        n_weight = weight / (edge_scale)
    elif normalization == 'log':
        n_weight = abs(np.log(abs(weight) + 1e-6) / np.log(edge_scale + 1e-6))
    return str(abs(n_weight) * pen_thickness)

def dfs(edges, start, end):
    visited = set()
    edge_set = set()
    
    def dfs_helper(node):
        if node == end:
            return
        visited.add(node)
        for neighbor in edges[node]:
            edge_set.add((node, neighbor))
            if neighbor not in visited:
                dfs_helper(neighbor)
                
    dfs_helper(start)
    return edge_set


def _to_hex(number, scale):
    number = number / scale
    
    # Define how the intensity changes based on the number
    # - Negative numbers increase red component to max
    # - Positive numbers increase blue component to max
    # - 0 results in white
    if number < 0:
        # Increase towards red, full intensity at -1.0
        red = 255
        green = blue = int((1 + number) * 255)  # Increase other components less as it gets more negative
    elif number > 0:
        # Increase towards blue, full intensity at 1.0
        blue = 255
        red = green = int((1 - number) * 255)  # Increase other components less as it gets more positive
    else:
        # Exact 0, resulting in white
        red = green = blue = 255 
    
    # decide whether text is black or white depending on darkness of color
    text_hex = "#000000" if (red*0.299 + green*0.587 + blue*0.114) > 170 else "#ffffff"

    # Convert to hex, ensuring each component is 2 digits
    hex_code = f'#{red:02X}{green:02X}{blue:02X}'
    
    return hex_code, text_hex


def _get_label(name, annotations=None):
    if annotations is None:
        return name
    else:
        match name.split(', '):
            case seq, feat:
                if feat in annotations:
                    component = feat.split('/')[0]
                    component = feat.split('_')[0]
                    return f'{seq}, {annotations[feat]} ({component})'
                return name
            case [feat]:
                if feat in annotations:
                    component = feat.split('/')[0]
                    component = feat.split('_')[0]
                    return f'{annotations[feat]} ({component})'


def plot_circuit(nodes, edges, layers=6, node_threshold=0.1, edge_threshold=0.01, pen_thickness=1, annotations=None, save_dir='circuit',
                 ylabel=None, seq_len=1.0, normalization='log', prune=True, post_resids=False):

    # get min and max node effects
    mins = {n: v.to_tensor().min() for n, v in nodes.items() if n != 'y'}
    maxs = {n: v.to_tensor().max() for n, v in nodes.items() if n != 'y'}
    min_effect = min(mins.values())
    max_effect = max(maxs.values())
    scale = max(abs(min_effect), abs(max_effect))    
    to_hex = partial(_to_hex, scale=scale)
    get_label = partial(_get_label, annotations=annotations)

    # rename embed to resid_-1
    start_layer = 0
    nodes_by_submod = {}
    if 'embed' in nodes:
        nodes_by_submod['resid_-1'] = {tuple(idx.tolist()) : nodes['embed'].to_tensor()[tuple(idx)].item() for idx in (nodes['embed'].to_tensor().abs() > node_threshold).nonzero()}
        edges['resid_-1'] = edges['embed']
        start_layer = -1
    
    for layer in range(layers):
        for component in ['attn', 'mlp', 'resid']:
            submod_nodes = nodes[f'{component}_{layer}'].to_tensor()
            n_features = submod_nodes.numel()
            k_threshold = int(node_threshold * n_features * seq_len)
            topk = submod_nodes.abs().flatten().topk(k_threshold)
            topk_ind = topk.indices[topk.values > 0]
            topk_ind = t.stack(t.unravel_index(topk_ind, submod_nodes.shape), dim=1).tolist()
            
            nodes_by_submod[f'{component}_{layer}'] = {
                tuple(idx) : submod_nodes[tuple(idx)].item() for idx in topk_ind
            }
    
    pruned_G = build_pruned_graph(nodes, edges, layers, start_layer, nodes_by_submod, post_resids) if prune else None
    G = build_formatted_graph(nodes, edges, layers, pen_thickness, ylabel, normalization, to_hex, 
                              get_label, start_layer, nodes_by_submod, post_resids, pruned_G=pruned_G)

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))
    G.render(save_dir, format='png', cleanup=False)
    
    
def build_pruned_graph(nodes, edges, layers, start_layer, nodes_by_submod, post_resids, first_layer_type='attn'):
    G = nx.DiGraph()
    
    for layer in trange(start_layer, layers):
        if post_resids and layer == start_layer:
            start_node = f'{first_layer_type}_{start_layer}'
        else:
            start_node = f'resid_{layer-1}'
        end_node = f'resid_{layer}'
        
        # select all edges that are on the path from resid-1 to resid
        if layer == start_layer and not post_resids: continue
        layer_edges = dfs(edges, start_node, end_node)
        
        for (up, down) in layer_edges:
            err_idx = nodes[up].shape[0]
            for upstream_idx in nodes_by_submod[up].keys():
                for downstream_idx in nodes_by_submod[down].keys():
                    weight = edges[up][down][tuple(downstream_idx)][tuple(upstream_idx)].item()
                    if abs(weight) > 0:
                        uname = get_name(up, layer, upstream_idx, err_idx, add_layer=False)
                        dname = get_name(down, layer, downstream_idx, err_idx, add_layer=False)
                        G.add_edge(uname, dname)

    err_idx = nodes[f'resid_{layers-1}'].shape[0]
    terminal_node = 'y'
    for idx in nodes_by_submod[f'resid_{layers-1}'].keys():
        name = get_name('resid', layers-1, idx, err_idx)
        G.add_edge(name, terminal_node)
        
    # restrict to nodes that have a path to terminal
    reversed_graph = G.reverse(copy=False)
    nodes_reaching_terminal = nx.descendants(reversed_graph, terminal_node)
    nodes_reaching_terminal.add(terminal_node)
    
    # restrict to nodes that are reachable from a starter node
    relevant_nodes = set()
    starter_nodes = {}
    for component in ['attn', 'mlp', 'resid']:  # allow any first layer node type to be a starter
        starter_nodes = nodes_by_submod[f'{component}_{start_layer}'].keys()
        for starter in starter_nodes:
            err_idx = nodes[f'{component}_{start_layer}'].shape[0]
            s = get_name(component, start_layer, starter, err_idx)
            if s in nodes_reaching_terminal:  # if the starter actually goes to terminal
                reachable_nodes = nx.descendants(G, s)  # all nodes reachable from this starter node
                relevant_nodes.update(reachable_nodes.intersection(nodes_reaching_terminal))  # if go to terminal and reachable by this starter
                relevant_nodes.add(s)
        
    # subgraph with only nodes that are on a path from resid_{start} to y
    pruned_graph = G.subgraph(relevant_nodes).copy()    
    return pruned_graph 

def add_node_prune_filter(name, G, pruned_G, **kwargs):
    if pruned_G is not None and name not in pruned_G.nodes:
        return
    G.node(name, **kwargs)
    
def add_edge_prune_filter(uname, dname, G, pruned_G, **kwargs):
    if pruned_G is not None and (uname, dname) not in pruned_G.edges:
        return
    G.edge(uname, dname, **kwargs)

def build_formatted_graph(nodes, edges, layers, pen_thickness, ylabel, normalization, to_hex, get_label, start_layer, nodes_by_submod, 
                          post_resids, first_layer_type='attn', pruned_G=None):
    G = Digraph(name='Feature circuit')
    G.graph_attr.update(rankdir='BT', newrank='true')
    G.node_attr.update(shape="box", style="rounded")
    
    for layer in trange(start_layer, layers):
        for component in ['attn', 'mlp', 'resid']:
            if layer == start_layer and component != 'resid': continue
            err_idx = nodes[f'{component}_{layer}'].shape[0]
            with G.subgraph(name=f'layer {layer} {component}') as subgraph:
                subgraph.attr(rank='same')
                max_seq_pos = None
                for idx, effect in nodes_by_submod[f'{component}_{layer}'].items():
                    name = get_name(component, layer, idx, err_idx)
                    fillhex, texthex = to_hex(effect)
                    if name[-1:].endswith('ε'):
                        add_node_prune_filter(name, subgraph, pruned_G, shape='triangle', width="1.6", height="0.8", fixedsize="true",
                                      fillcolor=fillhex, style='filled', fontcolor=texthex)
                    else:
                        add_node_prune_filter(name, subgraph, pruned_G, label=get_label(name), fillcolor=fillhex, fontcolor=texthex,
                                      style='filled')
                    # if sequence position is present, separate nodes by sequence position
                    match idx:
                        case (seq, _):
                            subgraph.node(f'{component}_{layer}_#{seq}_pre', style='invis'), subgraph.node(f'{component}_{layer}_#{seq}_post', style='invis')
                            subgraph.edge(f'{component}_{layer}_#{seq}_pre', name, style='invis'), subgraph.edge(name, f'{component}_{layer}_#{seq}_post', style='invis')

                            if max_seq_pos is None or seq > max_seq_pos:
                                max_seq_pos = seq

                if max_seq_pos is None: continue
                # make sure the auxiliary ordering nodes are in right order
                for seq in reversed(range(max_seq_pos+1)):
                    if f'{component}_{layer}_#{seq}_pre' in ''.join(subgraph.body):
                        for seq_prev in range(seq):
                            if f'{component}_{layer}_#{seq_prev}_post' in ''.join(subgraph.body):
                                subgraph.edge(f'{component}_{layer}_#{seq_prev}_post', f'{component}_{layer}_#{seq}_pre', style='invis')

        if post_resids and layer == start_layer:
            start_node = f'{first_layer_type}_{start_layer}'
        else:
            start_node = f'resid_{layer-1}'
        end_node = f'resid_{layer}'
        
        # # select all edges that are on the path from resid-1 to resid
        if layer == start_layer and not post_resids: continue
        layer_edges = dfs(edges, start_node, end_node)
        
        for (up, down) in layer_edges:
            err_idx = nodes[up].shape[0]
            jacobian = edges[up][down]
            edge_scale = float(max(abs(jacobian.values().min()), abs(jacobian.values().max())))
            for upstream_idx in nodes_by_submod[up].keys():
                for downstream_idx in nodes_by_submod[down].keys():
                    weight = jacobian[tuple(downstream_idx)][tuple(upstream_idx)].item()
                    if abs(weight) > 0:
                        uname = get_name(up, layer, upstream_idx, err_idx, add_layer=False)
                        dname = get_name(down, layer, downstream_idx, err_idx, add_layer=False)

                        add_edge_prune_filter(uname, dname, G, pruned_G, 
                                            penwidth=normalize_weight(weight, edge_scale, pen_thickness, normalization),
                                            color = 'red' if weight < 0 else 'blue')

    # the cherry on top
    add_node_prune_filter('y', G, pruned_G, shape='diamond', xlabel=ylabel)
    err_idx = nodes[f'resid_{layers-1}'].shape[0]
    jacobian = edges[f'resid_{layers-1}']['y']
    edge_scale = float(max(abs(jacobian.values().min()), abs(jacobian.values().max())))
    
    for idx in nodes_by_submod[f'resid_{layers-1}'].keys():
        weight = jacobian[tuple(idx)].item()
        name = get_name('resid', layers-1, idx, err_idx)
        
        add_edge_prune_filter(name, 'y', G, pruned_G,
                              penwidth=normalize_weight(weight, edge_scale, pen_thickness, normalization),
                              color = 'red' if weight < 0 else 'blue')
        
    return G


def plot_circuit_posaligned(nodes, edges, layers=6, length=6, example_text="The managers that the parent likes",
                            node_threshold=0.1, edge_threshold=0.01, pen_thickness=3, annotations=None, save_dir='circuit'):

    # get min and max node effects
    min_effect = min([v.to_tensor().min() for n, v in nodes.items() if n != 'y'])
    max_effect = max([v.to_tensor().max() for n, v in nodes.items() if n != 'y'])
    scale = max(abs(min_effect), abs(max_effect))
    to_hex = partial(_to_hex, scale=scale)
    get_label = partial(_get_label, annotations=annotations)

    words = example_text.split()

    G = Digraph(name='Feature circuit')
    G.graph_attr.update(rankdir='BT', newrank='true')
    G.node_attr.update(shape="box", style="rounded")

    start_layer = 0
    if 'embed' in nodes:
        nodes_by_submod = {
            'resid_-1' : {tuple(idx.tolist()) : nodes['embed'].to_tensor()[tuple(idx)].item() for idx in (nodes['embed'].to_tensor().abs() > node_threshold).nonzero()}
        }
        edges['resid_-1'] = edges['embed']
        start_layer = -1
    nodes_by_seqpos = defaultdict(list)
    nodes_by_layer = defaultdict(list)
    edgeset = set()

    for layer in range(layers):
        for component in ['attn', 'mlp', 'resid']:
            submod_nodes = nodes[f'{component}_{layer}'].to_tensor()
            nodes_by_submod[f'{component}_{layer}'] = {
                tuple(idx.tolist()) : submod_nodes[tuple(idx)].item() for idx in (submod_nodes.abs() > node_threshold).nonzero()
            }

    # add words to bottom of graph
    with G.subgraph(name=f'words') as subgraph:
        subgraph.attr(rank='same')
        prev_word = None
        for idx in range(length):
            word = words[idx]
            subgraph.node(word, shape='none', group=str(idx), fillcolor='transparent',
                          fontsize="30pt")
            if prev_word is not None:
                subgraph.edge(prev_word, word, style='invis', minlen="2")
            prev_word = word

    for layer in range(start_layer, layers):
        for component in ['attn', 'mlp', 'resid']:
            if layer == start_layer and component != 'resid': continue
            with G.subgraph(name=f'layer {layer} {component}') as subgraph:
                subgraph.attr(rank='same')
                max_seq_pos = None
                for idx, effect in nodes_by_submod[f'{component}_{layer}'].items():
                    name = get_name(component, layer, idx)
                    seq_pos, basename = name.split(", ")
                    fillhex, texthex = to_hex(effect)
                    if name[-1:] == 'ε':
                        subgraph.node(name, shape='triangle', group=seq_pos, width="1.6", height="0.8", fixedsize="true",
                                      fillcolor=fillhex, style='filled', fontcolor=texthex)
                    else:
                        subgraph.node(name, label=get_label(name), group=seq_pos, fillcolor=fillhex, fontcolor=texthex,
                                      style='filled')
                    
                    if len(nodes_by_seqpos[seq_pos]) == 0:
                        G.edge(words[int(seq_pos)], name, style='dotted', arrowhead='none', penwidth="1.5")
                        edgeset.add((words[int(seq_pos)], name))

                    nodes_by_seqpos[seq_pos].append(name)
                    nodes_by_layer[layer].append(name)

                    # if sequence position is present, separate nodes by sequence position
                    match idx:
                        case (seq, _):
                            subgraph.node(f'{component}_{layer}_#{seq}_pre', style='invis'), subgraph.node(f'{component}_{layer}_#{seq}_post', style='invis')
                            subgraph.edge(f'{component}_{layer}_#{seq}_pre', name, style='invis'), subgraph.edge(name, f'{component}_{layer}_#{seq}_post', style='invis')
                            if max_seq_pos is None or seq > max_seq_pos:
                                max_seq_pos = seq

                if max_seq_pos is None: continue
                # make sure the auxiliary ordering nodes are in right order
                for seq in reversed(range(max_seq_pos+1)):
                    if f'{component}_{layer}_#{seq}_pre' in ''.join(subgraph.body):
                        for seq_prev in range(seq):
                            if f'{component}_{layer}_#{seq_prev}_post' in ''.join(subgraph.body):
                                subgraph.edge(f'{component}_{layer}_#{seq_prev}_post', f'{component}_{layer}_#{seq}_pre', style='invis')

        
        for component in ['attn', 'mlp']:
            if layer == -1: continue
            for upstream_idx in nodes_by_submod[f'{component}_{layer}'].keys():
                for downstream_idx in nodes_by_submod[f'resid_{layer}'].keys():
                    weight = edges[f'{component}_{layer}'][f'resid_{layer}'][tuple(downstream_idx)][tuple(upstream_idx)].item()
                    if abs(weight) > edge_threshold:
                        uname = get_name(component, layer, upstream_idx)
                        dname = get_name('resid', layer, downstream_idx)
                        G.edge(
                            uname, dname,
                            penwidth=str(abs(weight) * pen_thickness),
                            color = 'red' if weight < 0 else 'blue'
                        )
                        edgeset.add((uname, dname))
        
        # add edges to previous layer resid
        for component in ['attn', 'mlp', 'resid']:
            if layer == -1: continue
            for upstream_idx in nodes_by_submod[f'resid_{layer-1}'].keys():
                for downstream_idx in nodes_by_submod[f'{component}_{layer}'].keys():
                    weight = edges[f'resid_{layer-1}'][f'{component}_{layer}'][tuple(downstream_idx)][tuple(upstream_idx)].item()
                    if abs(weight) > edge_threshold:
                        uname = get_name('resid', layer-1, upstream_idx)
                        dname = get_name(component, layer, downstream_idx)
                        G.edge(
                            uname, dname,
                            penwidth=str(abs(weight) * pen_thickness),
                            color = 'red' if weight < 0 else 'blue'
                        )
                        edgeset.add((uname, dname))


    # the cherry on top
    G.node('y', shape='diamond')
    for idx in nodes_by_submod[f'resid_{layers-1}'].keys():
        weight = edges[f'resid_{layers-1}']['y'][tuple(idx)].item()
        if abs(weight) > edge_threshold:
            name = get_name('resid', layers-1, idx)
            G.edge(
                name, 'y',
                penwidth=str(abs(weight) * pen_thickness),
                color = 'red' if weight < 0 else 'blue'
            )
            edgeset.add((uname, dname))

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))
    G.render(save_dir, format='png', cleanup=True)