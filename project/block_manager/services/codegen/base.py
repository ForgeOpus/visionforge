"""
Base utilities for code generation
Shared functions for both PyTorch and TensorFlow code generation
"""

from collections import deque
from typing import List, Dict, Any


def topological_sort(nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
    """
    Sort nodes in topological order based on edges using Kahn's algorithm.

    Args:
        nodes: List of node definitions
        edges: List of edge definitions

    Returns:
        List of nodes in topological order
    """
    node_map = {node['id']: node for node in nodes}

    # Build adjacency list and in-degree count
    graph = {node['id']: [] for node in nodes}
    in_degree = {node['id']: 0 for node in nodes}

    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        if source in graph and target in graph:
            graph[source].append(target)
            in_degree[target] += 1

    # Kahn's algorithm
    queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
    sorted_ids = []

    while queue:
        node_id = queue.popleft()
        sorted_ids.append(node_id)

        for neighbor in graph[node_id]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Cycle detection: if not all nodes were sorted, there's a cycle
    if len(sorted_ids) != len(nodes):
        # Find nodes that are still in the cycle (have non-zero in-degree)
        cycle_nodes = [node_id for node_id, degree in in_degree.items() if degree > 0]
        raise ValueError(
            f"Graph contains a cycle. Neural networks must be acyclic (feedforward). "
            f"Nodes involved in cycle: {', '.join(cycle_nodes[:5])}"
            + (" and more..." if len(cycle_nodes) > 5 else "")
        )

    # Return nodes in sorted order
    return [node_map[node_id] for node_id in sorted_ids if node_id in node_map]


def get_input_variable(incoming: List[str], var_map: Dict[str, str]) -> str:
    """
    Determine input variable name based on incoming connections.

    Args:
        incoming: List of incoming node IDs
        var_map: Map of node ID to variable name

    Returns:
        Variable name or list of variable names for multiple inputs
    """
    if not incoming:
        return 'x'
    elif len(incoming) == 1:
        return var_map.get(incoming[0], 'x')
    else:
        # Multiple inputs (for concat, add, etc.)
        input_vars = [var_map.get(src, 'x') for src in incoming]
        return f"[{', '.join(input_vars)}]"


def get_node_type(node: Dict[str, Any]) -> str:
    """Extract node type from node definition"""
    return node.get('data', {}).get('blockType', 'unknown')


def get_node_config(node: Dict[str, Any]) -> Dict[str, Any]:
    """Extract configuration from node definition"""
    return node.get('data', {}).get('config', {})
