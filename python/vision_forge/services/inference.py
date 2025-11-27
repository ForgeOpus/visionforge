"""
Dimension inference service
Automatically computes tensor shapes through the architecture graph
"""
from typing import List, Dict, Any, Optional, Tuple
from collections import deque


class ShapeInferenceEngine:
    """Infers tensor shapes through the architecture"""
    
    def __init__(self, nodes: List[Dict], edges: List[Dict]):
        self.nodes = nodes
        self.edges = edges
        self.node_map = {node['id']: node for node in nodes}
        self.inferred_shapes = {}
    
    def infer_shapes(self) -> Dict[str, Dict[str, Any]]:
        """
        Infer shapes for all blocks in the architecture
        Returns: Dictionary mapping node_id to {'inputShape': ..., 'outputShape': ...}
        """
        self.inferred_shapes = {}
        
        # Build graph structure
        graph = self._build_graph()
        
        # Topological sort to process nodes in dependency order
        sorted_nodes = self._topological_sort(graph)
        
        if not sorted_nodes:
            return {}
        
        # Process each node in topological order
        for node_id in sorted_nodes:
            node = self.node_map.get(node_id)
            if not node:
                continue
            
            self._infer_node_shapes(node, graph)
        
        return self.inferred_shapes
    
    def _build_graph(self) -> Dict[str, List[str]]:
        """Build adjacency list for the graph"""
        graph = {node['id']: [] for node in self.nodes}
        
        for edge in self.edges:
            source = edge.get('source')
            target = edge.get('target')
            if source in graph:
                graph[source].append(target)
        
        return graph
    
    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """
        Perform topological sort using Kahn's algorithm
        Returns nodes in dependency order
        """
        # Calculate in-degree for each node
        in_degree = {node_id: 0 for node_id in graph}
        for node_id, neighbors in graph.items():
            for neighbor in neighbors:
                in_degree[neighbor] += 1
        
        # Queue of nodes with no dependencies
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        sorted_nodes = []
        
        while queue:
            node_id = queue.popleft()
            sorted_nodes.append(node_id)
            
            # Reduce in-degree for neighbors
            for neighbor in graph.get(node_id, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If not all nodes are sorted, there's a cycle
        if len(sorted_nodes) != len(graph):
            return []
        
        return sorted_nodes
    
    def _infer_node_shapes(self, node: Dict, graph: Dict[str, List[str]]):
        """Infer input and output shapes for a single node"""
        node_id = node['id']
        block_type = self._get_block_type(node)
        config = node.get('data', {}).get('config', {})
        
        # Get input shape from previous node(s)
        input_shape = self._get_input_shape(node_id)
        
        # Compute output shape based on block type
        output_shape = self._compute_output_shape(block_type, input_shape, config)
        
        # Store inferred shapes
        self.inferred_shapes[node_id] = {
            'inputShape': input_shape,
            'outputShape': output_shape,
        }
    
    def _get_block_type(self, node):
        """Helper to get block type from node, handling different formats"""
        # Check if type is at top level (from frontend)
        if 'type' in node:
            return node['type']
        # Check if blockType is in data
        return node.get('data', {}).get('blockType', '')
    
    def _get_input_shape(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get input shape for a node from its predecessors"""
        # Find edges that target this node
        incoming_edges = [e for e in self.edges if e.get('target') == node_id]
        
        if not incoming_edges:
            # No incoming edges - this might be an input node
            node = self.node_map.get(node_id)
            if node and self._get_block_type(node) == 'input':
                # Get shape from input block configuration
                input_shape = node.get('data', {}).get('inputShape')
                if not input_shape:
                    # Try config
                    config = node.get('data', {}).get('config', {})
                    input_shape = config.get('inputShape')
                return input_shape
            return None
        
        # Get shape from source node
        source_id = incoming_edges[0].get('source')
        if source_id in self.inferred_shapes:
            return self.inferred_shapes[source_id]['outputShape']
        
        return None
    
    def _compute_output_shape(self, block_type: str, input_shape: Optional[Dict], 
                              config: Dict) -> Optional[Dict[str, Any]]:
        """
        Compute output shape based on block type and configuration
        """
        if not input_shape:
            if block_type == 'input':
                # Input block defines its own shape
                return config.get('inputShape') or input_shape
            return None
        
        dims = input_shape.get('dims', [])
        
        # Handle different block types
        if block_type == 'input':
            return input_shape
        
        elif block_type == 'linear':
            # Linear: [batch, in_features] -> [batch, out_features]
            out_features = config.get('out_features')
            if out_features and len(dims) >= 2:
                return {'dims': [dims[0], out_features]}
            return input_shape
        
        elif block_type == 'conv2d':
            # Conv2D: [batch, in_channels, height, width] -> [batch, out_channels, new_h, new_w]
            if len(dims) == 4:
                batch, in_ch, h, w = dims
                out_channels = config.get('out_channels', in_ch)
                kernel_size = config.get('kernel_size', 3)
                stride = config.get('stride', 1)
                padding = config.get('padding', 0)
                
                # Calculate output dimensions
                new_h = ((h + 2 * padding - kernel_size) // stride) + 1
                new_w = ((w + 2 * padding - kernel_size) // stride) + 1
                
                return {'dims': [batch, out_channels, new_h, new_w]}
            return input_shape
        
        elif block_type == 'flatten':
            # Flatten: [batch, ...] -> [batch, product_of_rest]
            if len(dims) > 1:
                batch = dims[0]
                flattened = 1
                for d in dims[1:]:
                    if isinstance(d, int):
                        flattened *= d
                return {'dims': [batch, flattened]}
            return input_shape
        
        elif block_type == 'maxpool' or block_type == 'avgpool':
            # Pooling: reduces spatial dimensions
            if len(dims) == 4:
                batch, channels, h, w = dims
                kernel_size = config.get('kernel_size', 2)
                stride = config.get('stride', kernel_size)
                padding = config.get('padding', 0)
                
                new_h = ((h + 2 * padding - kernel_size) // stride) + 1
                new_w = ((w + 2 * padding - kernel_size) // stride) + 1
                
                return {'dims': [batch, channels, new_h, new_w]}
            return input_shape
        
        elif block_type in ['relu', 'sigmoid', 'tanh', 'softmax', 'dropout', 'batchnorm']:
            # Activation and normalization layers preserve shape
            return input_shape
        
        elif block_type == 'concat':
            # Concatenate: combine multiple inputs along a dimension
            # This requires multiple inputs - for now, preserve input shape
            return input_shape
        
        elif block_type == 'add':
            # Element-wise addition preserves shape
            return input_shape
        
        else:
            # Unknown block type - preserve input shape
            return input_shape


def infer_dimensions(nodes: List[Dict], edges: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """
    Infer dimensions for all nodes in the architecture
    
    Args:
        nodes: List of node dictionaries from frontend
        edges: List of edge dictionaries from frontend
    
    Returns:
        Dictionary mapping node_id to inferred shapes
    """
    engine = ShapeInferenceEngine(nodes, edges)
    return engine.infer_shapes()
