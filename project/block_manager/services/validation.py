"""
Validation service for model architectures
Handles shape checking, connection validation, and architecture integrity
"""
from typing import List, Dict, Any, Tuple, Optional


class ValidationError:
    """Represents a validation error"""
    def __init__(self, message: str, node_id: str = None, edge_id: str = None, 
                 error_type: str = 'error', suggestion: str = None):
        self.message = message
        self.node_id = node_id
        self.edge_id = edge_id
        self.type = error_type
        self.suggestion = suggestion
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'message': self.message,
            'type': self.type,
        }
        if self.node_id:
            result['nodeId'] = self.node_id
        if self.edge_id:
            result['edgeId'] = self.edge_id
        if self.suggestion:
            result['suggestion'] = self.suggestion
        return result


class ArchitectureValidator:
    """Validates model architecture for correctness"""
    
    def __init__(self, nodes: List[Dict], edges: List[Dict]):
        self.nodes = nodes
        self.edges = edges
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.node_map = {node['id']: node for node in nodes}
    
    def validate(self) -> Tuple[bool, List[Dict], List[Dict]]:
        """
        Run all validation checks
        Returns: (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Run validation checks
        self._validate_input_blocks()
        self._validate_connections()
        self._validate_orphaned_blocks()
        self._validate_block_configurations()
        self._check_circular_dependencies()
        
        is_valid = len(self.errors) == 0
        
        return (
            is_valid,
            [e.to_dict() for e in self.errors],
            [w.to_dict() for w in self.warnings]
        )
    
    def _validate_input_blocks(self):
        """Check that there's at least one input block"""
        input_blocks = [
            node for node in self.nodes 
            if self._get_block_type(node) == 'input'
        ]
        
        if not input_blocks:
            self.errors.append(ValidationError(
                message='Architecture must have at least one input block',
                error_type='error',
                suggestion='Add an Input block from the palette'
            ))
        elif len(input_blocks) > 1:
            self.warnings.append(ValidationError(
                message=f'Architecture has {len(input_blocks)} input blocks. Consider if multiple inputs are necessary.',
                error_type='warning'
            ))
    
    def _get_block_type(self, node):
        """Helper to get block type from node, handling different formats"""
        # Check if type is at top level (from frontend)
        if 'type' in node:
            return node['type']
        # Check if blockType is in data
        return node.get('data', {}).get('blockType', '')
    
    def _validate_connections(self):
        """Validate all connections between blocks"""
        # Build connection map
        incoming_edges = {}
        outgoing_edges = {}
        
        for edge in self.edges:
            target_id = edge.get('target')
            source_id = edge.get('source')
            
            if target_id not in incoming_edges:
                incoming_edges[target_id] = []
            incoming_edges[target_id].append(edge)
            
            if source_id not in outgoing_edges:
                outgoing_edges[source_id] = []
            outgoing_edges[source_id].append(edge)
        
        # Check for blocks with multiple inputs (except merge blocks)
        for node_id, edges_list in incoming_edges.items():
            if len(edges_list) > 1:
                node = self.node_map.get(node_id)
                if node:
                    block_type = self._get_block_type(node)
                    # Allow multiple inputs for merge blocks
                    if block_type not in ['concat', 'add']:
                        self.errors.append(ValidationError(
                            message=f'Block has multiple input connections but is not a merge block',
                            node_id=node_id,
                            error_type='error',
                            suggestion='Use a Concatenate or Add block to merge multiple inputs'
                        ))
    
    def _validate_orphaned_blocks(self):
        """Check for blocks that aren't connected to the graph"""
        if not self.edges:
            # If there are no edges but there are nodes, all non-input nodes are orphaned
            for node in self.nodes:
                if self._get_block_type(node) != 'input':
                    self.warnings.append(ValidationError(
                        message='Block is not connected to the graph',
                        node_id=node['id'],
                        error_type='warning',
                        suggestion='Connect this block or remove it from the canvas'
                    ))
            return
        
        connected_nodes = set()
        for edge in self.edges:
            connected_nodes.add(edge.get('source'))
            connected_nodes.add(edge.get('target'))
        
        for node in self.nodes:
            node_id = node['id']
            block_type = self._get_block_type(node)
            
            # Input blocks don't need incoming connections
            if node_id not in connected_nodes and block_type != 'input':
                self.warnings.append(ValidationError(
                    message='Block is not connected to the graph',
                    node_id=node_id,
                    error_type='warning',
                    suggestion='Connect this block or remove it from the canvas'
                ))
    
    def _validate_block_configurations(self):
        """Check that required block parameters are configured"""
        for node in self.nodes:
            node_id = node['id']
            block_type = self._get_block_type(node)
            config = node.get('data', {}).get('config', {})
            
            # Check required parameters based on block type
            if block_type == 'linear':
                if 'out_features' not in config or not config.get('out_features'):
                    self.errors.append(ValidationError(
                        message='Linear layer requires out_features parameter',
                        node_id=node_id,
                        error_type='error',
                        suggestion='Configure the number of output features in the configuration panel'
                    ))
            
            elif block_type == 'conv2d':
                if 'out_channels' not in config or not config.get('out_channels'):
                    self.errors.append(ValidationError(
                        message='Conv2D layer requires out_channels parameter',
                        node_id=node_id,
                        error_type='error',
                        suggestion='Configure the number of output channels in the configuration panel'
                    ))
                if 'kernel_size' not in config or not config.get('kernel_size'):
                    self.errors.append(ValidationError(
                        message='Conv2D layer requires kernel_size parameter',
                        node_id=node_id,
                        error_type='error',
                        suggestion='Configure the kernel size in the configuration panel'
                    ))
            
            elif block_type == 'input':
                input_shape = node.get('data', {}).get('inputShape')
                if not input_shape or not input_shape.get('dims'):
                    self.errors.append(ValidationError(
                        message='Input block requires input shape configuration',
                        node_id=node_id,
                        error_type='error',
                        suggestion='Configure the input dimensions in the configuration panel'
                    ))
    
    def _check_circular_dependencies(self):
        """Check for circular dependencies in the graph"""
        # Build adjacency list
        graph = {}
        for node in self.nodes:
            graph[node['id']] = []
        
        for edge in self.edges:
            source = edge.get('source')
            target = edge.get('target')
            if source in graph:
                graph[source].append(target)
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for neighbor in graph.get(node_id, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in graph:
            if node_id not in visited:
                if has_cycle(node_id):
                    self.errors.append(ValidationError(
                        message='Architecture contains circular dependencies',
                        error_type='error',
                        suggestion='Remove connections that create cycles - neural networks must be directed acyclic graphs'
                    ))
                    break


def validate_architecture(nodes: List[Dict], edges: List[Dict]) -> Dict[str, Any]:
    """
    Validate an architecture and return validation results
    
    Args:
        nodes: List of node dictionaries from frontend
        edges: List of edge dictionaries from frontend
    
    Returns:
        Dictionary with validation results
    """
    validator = ArchitectureValidator(nodes, edges)
    is_valid, errors, warnings = validator.validate()
    
    return {
        'isValid': is_valid,
        'errors': errors,
        'warnings': warnings,
    }
