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