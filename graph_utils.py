def dfs(edges, start, end):
    visited = set()
    edge_set = []

    def dfs_helper(node):
        if node == end:
            return
        visited.add(node)
        for neighbor in edges[node]:
            edge_set.append((node, neighbor))
            if neighbor not in visited:
                dfs_helper(neighbor)

    dfs_helper(start)
    return edge_set

def iterate_edges(edges, layer, first_component):
    if layer == 0:
        start_node = first_component
    else:
        start_node = f'resid_{layer-1}'
    end_node = f'resid_{layer}'

    # select all edges that are on the path from resid-1 to resid
    layer_edges = dfs(edges, start_node, end_node)

    # add each to the graph, checking to make sure that the weight is non-zero
    yield from layer_edges