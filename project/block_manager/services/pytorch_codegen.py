"""
PyTorch Code Generation Service
Generates PyTorch nn.Module code from architecture graphs with professional class-based structure
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import logging
import json
import time

# Configure logging
logger = logging.getLogger(__name__)


# ============================================
# Custom Exception Classes
# ============================================

class GroupDefinitionNotFoundError(Exception):
    """Raised when a group block references a non-existent definition."""
    
    def __init__(self, node_id: str, definition_id: str):
        self.node_id = node_id
        self.definition_id = definition_id
        super().__init__(
            f"Group block {node_id} references undefined definition {definition_id}"
        )


class ShapeMismatchError(Exception):
    """Raised when internal layers have incompatible shapes."""
    
    def __init__(self, block_name: str, layer_name: str, expected: Dict, actual: Dict):
        self.block_name = block_name
        self.layer_name = layer_name
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Shape mismatch in block '{block_name}' at layer '{layer_name}': "
            f"expected {expected}, got {actual}"
        )


class CyclicDependencyError(Exception):
    """Raised when internal structure contains cycles."""

    def __init__(self, block_name: str, cycle_nodes: List[str]):
        self.block_name = block_name
        self.cycle_nodes = cycle_nodes
        super().__init__(
            f"Cyclic dependency detected in block '{block_name}': {' -> '.join(cycle_nodes)}"
        )


class UnsupportedNodeTypeError(Exception):
    """Raised when encountering an unsupported node type during code generation."""

    def __init__(self, node_id: str, node_type: str, framework: str):
        self.node_id = node_id
        self.node_type = node_type
        self.framework = framework
        super().__init__(
            f"Unsupported node type '{node_type}' for {framework} in node {node_id}. "
            f"Please use a supported layer type or implement this layer manually."
        )


class ShapeInferenceError(Exception):
    """Raised when shape inference fails for a node."""

    def __init__(self, node_id: str, node_type: str, reason: str, suggestion: str = None):
        self.node_id = node_id
        self.node_type = node_type
        self.reason = reason
        self.suggestion = suggestion
        msg = f"Shape inference failed for node {node_id} ({node_type}): {reason}"
        if suggestion:
            msg += f"\nSuggestion: {suggestion}"
        super().__init__(msg)


class MissingShapeDataError(Exception):
    """Raised when required shape data is missing from upstream nodes."""

    def __init__(self, node_id: str, upstream_node_id: str, missing_keys: List[str]):
        self.node_id = node_id
        self.upstream_node_id = upstream_node_id
        self.missing_keys = missing_keys
        super().__init__(
            f"Node {node_id} requires shape data from upstream node {upstream_node_id}, "
            f"but the following keys are missing: {', '.join(missing_keys)}. "
            f"Check that the upstream node produces valid output shapes."
        )


# ============================================
# Shape Data Validation Utility
# ============================================

def safe_get_shape_data(
    shape_map: Dict[str, Dict[str, Any]],
    node_id: str,
    upstream_node_id: str,
    required_keys: List[str],
    default_values: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Safely retrieve shape data from upstream node with validation.

    This function ensures that shape data access is safe by:
    1. Checking that the upstream node exists in the shape map
    2. Validating that shape data is not None and is a dictionary
    3. Verifying all required keys are present
    4. Providing clear error messages when data is missing

    Args:
        shape_map: Map of node IDs to shape information dictionaries
        node_id: Current node ID (for error messages and tracing)
        upstream_node_id: ID of upstream node to retrieve shape from
        required_keys: List of required shape keys (e.g., ['out_channels', 'out_height'])
        default_values: Optional default values to use if data is missing

    Returns:
        Dictionary containing the requested shape data

    Raises:
        MissingShapeDataError: If required data is missing and no defaults provided
        ShapeInferenceError: If upstream shape data is invalid (None or not a dict)

    Example:
        >>> shape_data = safe_get_shape_data(
        ...     shape_map,
        ...     'conv2',
        ...     'conv1',
        ...     ['out_channels', 'out_height', 'out_width'],
        ...     default_values={'out_channels': 64, 'out_height': 32, 'out_width': 32}
        ... )
        >>> print(shape_data['out_channels'])
        64
    """
    result = {}

    # Check if upstream node exists in shape map
    if upstream_node_id not in shape_map:
        if default_values:
            return default_values.copy()
        raise MissingShapeDataError(
            node_id=node_id,
            upstream_node_id=upstream_node_id,
            missing_keys=required_keys
        )

    upstream_shape = shape_map[upstream_node_id]

    # Validate upstream shape is not None and is a dict
    if upstream_shape is None or not isinstance(upstream_shape, dict):
        if default_values:
            return default_values.copy()
        raise ShapeInferenceError(
            node_id=node_id,
            node_type="unknown",
            reason=f"Upstream node {upstream_node_id} has invalid shape data (None or not a dict)",
            suggestion="Check that the upstream node is properly configured and connected"
        )

    # Extract required keys with validation
    missing_keys = []
    for key in required_keys:
        if key in upstream_shape:
            result[key] = upstream_shape[key]
        elif default_values and key in default_values:
            result[key] = default_values[key]
        else:
            missing_keys.append(key)

    if missing_keys:
        if default_values:
            for key in missing_keys:
                if key in default_values:
                    result[key] = default_values[key]
            return result
        raise MissingShapeDataError(
            node_id=node_id,
            upstream_node_id=upstream_node_id,
            missing_keys=missing_keys
        )

    return result


class GroupBlockShapeComputer:
    """
    Computes output shapes for group blocks by traversing internal structure.
    
    This class handles shape inference for group blocks by:
    1. Retrieving the internal structure of a group block
    2. Topologically sorting internal nodes
    3. Propagating shapes through the internal graph
    4. Mapping internal output nodes to external output ports
    
    Performance optimizations:
    - Shape caching to avoid redundant computations
    - Lazy topological sorting (cached per definition)
    - Cache invalidation on definition changes
    """
    
    def __init__(self, group_definitions: Dict[str, Dict[str, Any]], cache_size: int = 1000, profiler: Optional['ShapeInferenceProfiler'] = None):
        """
        Initialize with group definitions.
        
        Args:
            group_definitions: Map of definition ID to definition dict
            cache_size: Maximum number of cached shape computations (default: 1000)
            profiler: Optional profiler for performance analysis
        """
        self.group_definitions = group_definitions
        self.shape_cache = {}  # Cache computed shapes: {(group_def_id, input_shape_tuple): output_shape}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Cache for topological sorts per definition
        self.topo_sort_cache = {}  # {group_def_id: sorted_nodes}
        
        # Track definition versions for cache invalidation
        self.definition_versions = {}  # {group_def_id: version_hash}
        self._initialize_definition_versions()
        
        # Optional profiler
        self.profiler = profiler
    
    def compute_output_shape(
        self,
        group_def_id: str,
        input_shape: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], List[Exception]]:
        """
        Compute output shape for a group block given input shape.
        
        Args:
            group_def_id: ID of the group definition
            input_shape: Input shape dict with keys like 'out_channels', 'out_height', etc.
            
        Returns:
            Tuple of (output shape dict with computed dimensions or None, list of errors)
        """
        start_time = time.time() if self.profiler and self.profiler.enabled else None
        
        errors = []
        
        # Check cache first
        cache_key = (group_def_id, self._shape_to_tuple(input_shape))
        if cache_key in self.shape_cache:
            self.cache_hits += 1
            logger.debug(f"Cache hit for group {group_def_id} (hit rate: {self.cache_hits}/{self.cache_hits + self.cache_misses})")
            if start_time:
                self.profiler.record_timing('compute_output_shape_cached', time.time() - start_time)
            return self.shape_cache[cache_key], []
        
        self.cache_misses += 1
        
        # Get group definition
        if group_def_id not in self.group_definitions:
            error = GroupDefinitionNotFoundError('unknown', group_def_id)
            logger.error(str(error))
            errors.append(error)
            return None, errors
        
        group_def = self.group_definitions[group_def_id]
        group_name = group_def.get('name', group_def_id)
        internal_structure = group_def.get('internal_structure', {})
        
        if not internal_structure:
            logger.warning(f"Group {group_name} has no internal structure")
            return input_shape.copy(), []
        
        internal_nodes = internal_structure.get('nodes', [])
        internal_edges = internal_structure.get('edges', [])
        port_mappings = internal_structure.get('portMappings', [])
        
        # Handle edge case: no internal nodes
        if not internal_nodes:
            logger.warning(f"Group {group_name} has no internal nodes")
            return input_shape.copy(), []
        
        # Compute internal shapes
        try:
            internal_shape_map, shape_errors = self.compute_internal_shapes(
                internal_nodes,
                internal_edges,
                port_mappings,
                input_shape,
                group_name
            )
            
            # Collect any errors from internal shape computation
            errors.extend(shape_errors)
            
            if not internal_shape_map:
                logger.error(f"Failed to compute internal shapes for group {group_name}")
                return None, errors
            
            # Find output port mappings
            output_ports = [pm for pm in port_mappings if pm.get('type') == 'output']
            
            if not output_ports:
                logger.warning(f"Group {group_name} has no output ports")
                return input_shape.copy(), errors
            
            # Handle multiple output ports - return dict with shapes for each port
            if len(output_ports) > 1:
                logger.debug(f"Group {group_name} has {len(output_ports)} output ports")
                output_shapes = {}
                for idx, port in enumerate(output_ports):
                    internal_node_id = port.get('internalNodeId')
                    port_label = port.get('externalPortLabel', f'output_{idx}')
                    if internal_node_id in internal_shape_map:
                        output_shapes[port_label] = internal_shape_map[internal_node_id]
                    else:
                        error_msg = f"Output port '{port_label}' maps to unknown node {internal_node_id}"
                        logger.error(error_msg)
                        errors.append(Exception(error_msg))
                
                # For now, return the first output port's shape for backward compatibility
                # In the future, we should return all output shapes
                if output_shapes:
                    first_output_shape = list(output_shapes.values())[0]
                    # Cache the result only if no errors
                    if not errors:
                        self.shape_cache[cache_key] = first_output_shape
                        self._evict_cache_if_needed()
                    return first_output_shape, errors
                else:
                    return None, errors
            
            # For single output, return the shape of the mapped internal node
            internal_node_id = output_ports[0].get('internalNodeId')
            if internal_node_id in internal_shape_map:
                output_shape = internal_shape_map[internal_node_id]
                # Cache the result only if no errors
                if not errors:
                    self.shape_cache[cache_key] = output_shape
                    self._evict_cache_if_needed()
                return output_shape, errors
            else:
                error_msg = f"Output port maps to unknown node {internal_node_id}"
                logger.error(error_msg)
                errors.append(Exception(error_msg))
                return None, errors
            
        except CyclicDependencyError as e:
            logger.error(f"Cyclic dependency in group {group_name}: {e}")
            errors.append(e)
            if start_time:
                self.profiler.record_timing('compute_output_shape_error', time.time() - start_time)
            return None, errors
        except Exception as e:
            logger.error(f"Error computing output shape for group {group_name}: {e}")
            errors.append(e)
            if start_time:
                self.profiler.record_timing('compute_output_shape_error', time.time() - start_time)
            return None, errors
        finally:
            # Record timing for successful computation
            if start_time and not errors:
                self.profiler.record_timing('compute_output_shape_success', time.time() - start_time)
    
    def compute_internal_shapes(
        self,
        internal_nodes: List[Dict],
        internal_edges: List[Dict],
        port_mappings: List[Dict],
        external_input_shape: Dict[str, Any],
        group_name: str = "unknown"
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Exception]]:
        """
        Compute shapes for all internal nodes.
        
        Args:
            internal_nodes: List of nodes inside the block
            internal_edges: List of edges inside the block
            port_mappings: Port mapping configuration
            external_input_shape: Shape coming into the block
            group_name: Name of the group block for error reporting
            
        Returns:
            Tuple of (map of node_id to shape info, list of errors)
        """
        import time
        start_time = time.time() if self.profiler and self.profiler.enabled else None
        
        errors = []
        
        # Edge case: no internal edges - validate that all nodes are input/output nodes
        if not internal_edges:
            logger.warning(f"Group {group_name} has no internal edges")
            # Check if we have only input/output nodes
            non_io_nodes = [n for n in internal_nodes if get_node_type(n) not in ('input', 'dataloader', 'output')]
            if non_io_nodes:
                error_msg = f"Group {group_name} has {len(non_io_nodes)} non-input/output nodes but no edges connecting them"
                logger.error(error_msg)
                errors.append(Exception(error_msg))
                # Still try to process nodes with default shapes
        
        # Topologically sort internal nodes (with caching)
        try:
            sorted_nodes = self._get_cached_topological_sort(
                group_name, internal_nodes, internal_edges
            )
        except Exception as e:
            # Check if this is a cyclic dependency
            if "cycle" in str(e).lower():
                cycle_error = CyclicDependencyError(group_name, [])
                logger.error(str(cycle_error))
                errors.append(cycle_error)
            else:
                logger.error(f"Failed to topologically sort internal nodes: {e}")
                errors.append(e)
            return {}, errors
        
        # Build edge map for finding inputs
        edge_map = {}
        for edge in internal_edges:
            target = edge.get('target')
            source = edge.get('source')
            if target not in edge_map:
                edge_map[target] = []
            edge_map[target].append(source)
        
        # Initialize shape map
        internal_shape_map = {}
        
        # Handle multiple input ports - map each to external input shape
        input_ports = [pm for pm in port_mappings if pm.get('type') == 'input']
        
        if len(input_ports) > 1:
            logger.debug(f"Group {group_name} has {len(input_ports)} input ports")
            # For multiple inputs, we need to handle them separately
            # For now, we'll use the same external_input_shape for all inputs
            # In the future, we should support different shapes for different inputs
            for idx, input_port in enumerate(input_ports):
                internal_node_id = input_port.get('internalNodeId')
                port_label = input_port.get('externalPortLabel', f'input_{idx}')
                if internal_node_id:
                    # Use the same shape for all inputs for now
                    internal_shape_map[internal_node_id] = external_input_shape.copy()
                    logger.debug(f"Mapped input port '{port_label}' to node {internal_node_id}")
        else:
            # Single input port
            for input_port in input_ports:
                internal_node_id = input_port.get('internalNodeId')
                if internal_node_id:
                    internal_shape_map[internal_node_id] = external_input_shape.copy()
        
        # Detect disconnected subgraphs - nodes with no path from input ports
        # Build a set of reachable nodes from input ports
        reachable_nodes = set()
        if input_ports:
            # BFS from input nodes
            from collections import deque
            queue = deque()
            for input_port in input_ports:
                internal_node_id = input_port.get('internalNodeId')
                if internal_node_id:
                    queue.append(internal_node_id)
                    reachable_nodes.add(internal_node_id)
            
            # Build forward edge map (source -> targets)
            forward_edge_map = {}
            for edge in internal_edges:
                source = edge.get('source')
                target = edge.get('target')
                if source not in forward_edge_map:
                    forward_edge_map[source] = []
                forward_edge_map[source].append(target)
            
            # BFS to find all reachable nodes
            while queue:
                current = queue.popleft()
                for neighbor in forward_edge_map.get(current, []):
                    if neighbor not in reachable_nodes:
                        reachable_nodes.add(neighbor)
                        queue.append(neighbor)
        
        # Check for disconnected nodes
        all_node_ids = {node['id'] for node in internal_nodes}
        disconnected_nodes = all_node_ids - reachable_nodes
        # Filter out input/output nodes from disconnected check
        disconnected_non_io = [nid for nid in disconnected_nodes 
                               if get_node_type(next((n for n in internal_nodes if n['id'] == nid), {})) 
                               not in ('input', 'dataloader', 'output')]
        
        if disconnected_non_io:
            logger.warning(f"Group {group_name} has {len(disconnected_non_io)} disconnected nodes: {disconnected_non_io[:3]}")
            # This is a warning, not an error - we'll still process what we can
        
        # Process each internal node in topological order
        for node in sorted_nodes:
            node_id = node['id']
            node_type = get_node_type(node)
            config = node.get('data', {}).get('config', {})
            node_label = node.get('data', {}).get('label', node_type)
            
            # Skip if already computed (input nodes)
            if node_id in internal_shape_map:
                continue
            
            # Get incoming edges
            incoming = edge_map.get(node_id, [])
            
            # Initialize shape info for this node
            shape_info = {}
            
            # Handle different node types
            if node_type == 'input':
                # Input nodes should already be in the map
                if node_id not in internal_shape_map:
                    internal_shape_map[node_id] = external_input_shape.copy()
                continue
            
            # Handle nodes with multiple inputs (concat, add, etc.)
            if node_type in ('concat', 'add') and len(incoming) > 1:
                logger.debug(f"Processing {node_type} node {node_label} with {len(incoming)} inputs")
                
                # Validate that all inputs have compatible shapes
                input_shapes = []
                for src_id in incoming:
                    if src_id in internal_shape_map:
                        input_shapes.append(internal_shape_map[src_id])
                    else:
                        logger.warning(f"Input {src_id} for {node_type} node {node_label} has no computed shape")
                
                if not input_shapes:
                    # No valid inputs, use default
                    shape_info = {'out_channels': 64, 'out_height': 7, 'out_width': 7}
                elif node_type == 'concat':
                    # For concat, sum the channels
                    total_channels = sum(s.get('out_channels', 0) for s in input_shapes)
                    # Use spatial dimensions from first input
                    shape_info['out_channels'] = total_channels
                    if 'out_height' in input_shapes[0]:
                        shape_info['out_height'] = input_shapes[0]['out_height']
                    if 'out_width' in input_shapes[0]:
                        shape_info['out_width'] = input_shapes[0]['out_width']
                elif node_type == 'add':
                    # For add, channels must match - use first input's shape
                    shape_info = input_shapes[0].copy()
                    # Validate that all inputs have same channels
                    for idx, s in enumerate(input_shapes[1:], 1):
                        if s.get('out_channels') != shape_info.get('out_channels'):
                            error = ShapeMismatchError(
                                group_name,
                                node_label,
                                {'out_channels': shape_info.get('out_channels')},
                                {'out_channels': s.get('out_channels')}
                            )
                            logger.error(str(error))
                            errors.append(error)
                
                internal_shape_map[node_id] = shape_info
                continue
            
            elif node_type == 'conv2d':
                # Get input channels from previous layer
                if incoming and incoming[0] in internal_shape_map:
                    prev_shape = internal_shape_map[incoming[0]]
                    if 'out_channels' not in prev_shape:
                        # Shape mismatch: expected channels but got features
                        error = ShapeMismatchError(
                            group_name,
                            node_label,
                            {'out_channels': 'required'},
                            prev_shape
                        )
                        logger.error(str(error))
                        errors.append(error)
                        shape_info['in_channels'] = 3  # Use default
                    else:
                        shape_info['in_channels'] = prev_shape.get('out_channels', 3)
                else:
                    shape_info['in_channels'] = 3
                
                # Output channels from config
                shape_info['out_channels'] = config.get('out_channels', 64)
                
                # Calculate output spatial dimensions
                if incoming and incoming[0] in internal_shape_map:
                    prev_shape = internal_shape_map[incoming[0]]
                    kernel_size = config.get('kernel_size', 3)
                    stride = config.get('stride', 1)
                    padding = config.get('padding', 0)
                    
                    if 'out_height' in prev_shape and 'out_width' in prev_shape:
                        shape_info['out_height'] = (prev_shape['out_height'] + 2*padding - kernel_size) // stride + 1
                        shape_info['out_width'] = (prev_shape['out_width'] + 2*padding - kernel_size) // stride + 1
            
            elif node_type == 'maxpool':
                # Preserve channels, reduce spatial dimensions
                if incoming and incoming[0] in internal_shape_map:
                    prev_shape = internal_shape_map[incoming[0]]
                    if 'out_channels' not in prev_shape:
                        # Shape mismatch: expected channels
                        error = ShapeMismatchError(
                            group_name,
                            node_label,
                            {'out_channels': 'required'},
                            prev_shape
                        )
                        logger.error(str(error))
                        errors.append(error)
                        shape_info['in_channels'] = 64  # Use default
                        shape_info['out_channels'] = 64
                    else:
                        shape_info['in_channels'] = prev_shape.get('out_channels', 64)
                        shape_info['out_channels'] = shape_info['in_channels']
                    
                    kernel_size = config.get('kernel_size', 2)
                    stride = config.get('stride', 2)
                    padding = config.get('padding', 0)
                    
                    if 'out_height' in prev_shape and 'out_width' in prev_shape:
                        shape_info['out_height'] = (prev_shape['out_height'] + 2*padding - kernel_size) // stride + 1
                        shape_info['out_width'] = (prev_shape['out_width'] + 2*padding - kernel_size) // stride + 1
            
            elif node_type == 'flatten':
                # Convert spatial dimensions to features
                if incoming and incoming[0] in internal_shape_map:
                    prev_shape = internal_shape_map[incoming[0]]
                    channels = prev_shape.get('out_channels', 64)
                    height = prev_shape.get('out_height', 7)
                    width = prev_shape.get('out_width', 7)
                    shape_info['out_features'] = channels * height * width
            
            elif node_type == 'linear':
                # Get input features from previous layer
                if incoming and incoming[0] in internal_shape_map:
                    prev_shape = internal_shape_map[incoming[0]]
                    # Accept both 'out_features' (PyTorch) and 'out_units' (TensorFlow)
                    if 'out_features' not in prev_shape and 'out_units' not in prev_shape:
                        # Shape mismatch: expected features but got channels
                        error = ShapeMismatchError(
                            group_name,
                            node_label,
                            {'out_features': 'required'},
                            prev_shape
                        )
                        logger.error(str(error))
                        errors.append(error)
                        shape_info['in_features'] = 512  # Use default
                    else:
                        # Use out_features if available, otherwise out_units
                        shape_info['in_features'] = prev_shape.get('out_features', prev_shape.get('out_units', 512))
                else:
                    shape_info['in_features'] = 512
                
                # Output features from config
                shape_info['out_features'] = config.get('out_features', 128)
            
            elif node_type == 'batchnorm' or node_type == 'batchnorm2d':
                # Preserve dimensions, just need num_features
                if incoming and incoming[0] in internal_shape_map:
                    prev_shape = internal_shape_map[incoming[0]]
                    if 'out_channels' not in prev_shape:
                        # Shape mismatch: expected channels but got features
                        error = ShapeMismatchError(
                            group_name,
                            node_label,
                            {'out_channels': 'required'},
                            prev_shape
                        )
                        logger.error(str(error))
                        errors.append(error)
                        shape_info['num_features'] = 64  # Use default
                        shape_info['out_channels'] = 64
                    else:
                        shape_info['num_features'] = prev_shape.get('out_channels', 64)
                        shape_info['out_channels'] = shape_info['num_features']
                        if 'out_height' in prev_shape:
                            shape_info['out_height'] = prev_shape['out_height']
                        if 'out_width' in prev_shape:
                            shape_info['out_width'] = prev_shape['out_width']
            
            elif node_type == 'group':
                # Handle nested group blocks recursively
                nested_group_def_id = node.get('data', {}).get('groupDefinitionId')
                
                if not nested_group_def_id:
                    logger.warning(f"Nested group block {node_label} has no definition ID")
                    # Use input shape if available
                    if incoming and incoming[0] in internal_shape_map:
                        shape_info = internal_shape_map[incoming[0]].copy()
                    else:
                        shape_info = {'out_channels': 64, 'out_height': 7, 'out_width': 7}
                elif not incoming:
                    logger.warning(f"Nested group block {node_label} has no incoming edges")
                    # Use default shape
                    shape_info = {'out_channels': 64, 'out_height': 7, 'out_width': 7}
                elif incoming[0] not in internal_shape_map:
                    logger.warning(f"Nested group block {node_label} has incoming edge from node with no computed shape")
                    # Use default shape
                    shape_info = {'out_channels': 64, 'out_height': 7, 'out_width': 7}
                else:
                    # Recursively compute nested group block shape
                    nested_input_shape = internal_shape_map[incoming[0]]
                    logger.debug(f"Recursively computing shape for nested group {node_label} (def: {nested_group_def_id})")
                    nested_output_shape, nested_errors = self.compute_output_shape(nested_group_def_id, nested_input_shape)
                    
                    # Collect errors from nested computation
                    errors.extend(nested_errors)
                    
                    if nested_output_shape:
                        shape_info = nested_output_shape
                        logger.debug(f"Nested group {node_label} output shape: {nested_output_shape}")
                    else:
                        # Fallback: copy input shape
                        shape_info = nested_input_shape.copy()
                        logger.warning(f"Failed to compute shape for nested group {node_label}, using input shape")
            
            else:
                # For other layers, try to preserve shape from input
                if incoming and incoming[0] in internal_shape_map:
                    prev_shape = internal_shape_map[incoming[0]]
                    shape_info.update(prev_shape)
            
            internal_shape_map[node_id] = shape_info
        
        # Record timing
        if start_time:
            self.profiler.record_timing('compute_internal_shapes', time.time() - start_time)
        
        return internal_shape_map, errors
    
    def _get_cached_topological_sort(
        self,
        group_name: str,
        internal_nodes: List[Dict],
        internal_edges: List[Dict]
    ) -> List[Dict]:
        """
        Get topologically sorted nodes with caching.
        
        Args:
            group_name: Name of the group (for cache key)
            internal_nodes: List of internal nodes
            internal_edges: List of internal edges
            
        Returns:
            List of topologically sorted nodes
        """
        # Use group_name as cache key (assumes nodes/edges don't change for same group)
        if group_name in self.topo_sort_cache:
            logger.debug(f"Using cached topological sort for {group_name}")
            return self.topo_sort_cache[group_name]
        
        # Compute topological sort
        sorted_nodes = topological_sort(internal_nodes, internal_edges)
        
        # Cache the result
        self.topo_sort_cache[group_name] = sorted_nodes
        logger.debug(f"Cached topological sort for {group_name} ({len(sorted_nodes)} nodes)")
        
        return sorted_nodes
    
    def _shape_to_tuple(self, shape: Dict[str, Any]) -> tuple:
        """
        Convert shape dict to tuple for use as cache key.
        
        Args:
            shape: Shape dictionary
            
        Returns:
            Tuple representation of shape
        """
        # Create a sorted tuple of key-value pairs
        return tuple(sorted(shape.items()))
    
    def _initialize_definition_versions(self):
        """Initialize version hashes for all definitions."""
        for def_id, definition in self.group_definitions.items():
            self.definition_versions[def_id] = self._compute_definition_hash(definition)
    
    def _compute_definition_hash(self, definition: Dict[str, Any]) -> int:
        """
        Compute a hash of the definition for cache invalidation.
        
        Args:
            definition: Group block definition
            
        Returns:
            Hash value representing the definition structure
        """
        import json
        # Hash the internal structure to detect changes
        internal_structure = definition.get('internal_structure', {})
        # Convert to JSON string for consistent hashing
        structure_str = json.dumps(internal_structure, sort_keys=True)
        return hash(structure_str)
    
    def invalidate_cache_for_definition(self, group_def_id: str):
        """
        Invalidate all cached data for a specific definition.
        
        Args:
            group_def_id: ID of the group definition that changed
        """
        # Remove shape cache entries for this definition
        keys_to_remove = [key for key in self.shape_cache.keys() if key[0] == group_def_id]
        for key in keys_to_remove:
            del self.shape_cache[key]
        
        # Remove topological sort cache
        if group_def_id in self.topo_sort_cache:
            del self.topo_sort_cache[group_def_id]
        
        # Update version hash
        if group_def_id in self.group_definitions:
            self.definition_versions[group_def_id] = self._compute_definition_hash(
                self.group_definitions[group_def_id]
            )
        
        logger.debug(f"Cache invalidated for definition {group_def_id}")
    
    def update_definition(self, group_def_id: str, new_definition: Dict[str, Any]):
        """
        Update a group definition and invalidate related caches.
        
        Args:
            group_def_id: ID of the group definition
            new_definition: New definition data
        """
        # Check if definition actually changed
        old_hash = self.definition_versions.get(group_def_id)
        new_hash = self._compute_definition_hash(new_definition)
        
        if old_hash != new_hash:
            # Definition changed, invalidate caches
            self.group_definitions[group_def_id] = new_definition
            self.invalidate_cache_for_definition(group_def_id)
            logger.info(f"Definition {group_def_id} updated and cache invalidated")
        else:
            # No structural change, just update the definition
            self.group_definitions[group_def_id] = new_definition
            logger.debug(f"Definition {group_def_id} updated (no structural change)")
    
    def _evict_cache_if_needed(self):
        """Evict oldest cache entries if cache size limit is exceeded."""
        if len(self.shape_cache) > self.cache_size:
            # Simple LRU: remove oldest 10% of entries
            num_to_remove = max(1, len(self.shape_cache) // 10)
            keys_to_remove = list(self.shape_cache.keys())[:num_to_remove]
            for key in keys_to_remove:
                del self.shape_cache[key]
            logger.debug(f"Evicted {num_to_remove} cache entries (cache size: {len(self.shape_cache)})")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.shape_cache),
            'cache_limit': self.cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'topo_sort_cache_size': len(self.topo_sort_cache)
        }
    
    def clear_cache(self):
        """Clear all caches and reset statistics."""
        self.shape_cache.clear()
        self.topo_sort_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.debug("All caches cleared")


class ShapeInferenceProfiler:
    """
    Profiler for shape inference performance analysis.
    
    Tracks timing and statistics for shape inference operations
    to identify performance bottlenecks in large architectures.
    """
    
    def __init__(self):
        """Initialize the profiler."""
        self.timings = {}  # {operation_name: [durations]}
        self.enabled = False
    
    def enable(self):
        """Enable profiling."""
        self.enabled = True
        logger.info("Shape inference profiling enabled")
    
    def disable(self):
        """Disable profiling."""
        self.enabled = False
        logger.info("Shape inference profiling disabled")
    
    def record_timing(self, operation: str, duration: float):
        """
        Record timing for an operation.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
        """
        if not self.enabled:
            return
        
        if operation not in self.timings:
            self.timings[operation] = []
        self.timings[operation].append(duration)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get profiling statistics.
        
        Returns:
            Dictionary with statistics for each operation
        """
        stats = {}
        for operation, durations in self.timings.items():
            if durations:
                stats[operation] = {
                    'count': len(durations),
                    'total': sum(durations),
                    'mean': sum(durations) / len(durations),
                    'min': min(durations),
                    'max': max(durations)
                }
        return stats
    
    def print_report(self):
        """Print a formatted profiling report."""
        if not self.timings:
            print("No profiling data collected")
            return
        
        print("\n" + "=" * 80)
        print("Shape Inference Performance Report")
        print("=" * 80)
        
        stats = self.get_stats()
        for operation, data in sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True):
            print(f"\n{operation}:")
            print(f"  Count: {data['count']}")
            print(f"  Total: {data['total']:.4f}s")
            print(f"  Mean:  {data['mean']:.4f}s")
            print(f"  Min:   {data['min']:.4f}s")
            print(f"  Max:   {data['max']:.4f}s")
        
        print("\n" + "=" * 80)
    
    def reset(self):
        """Reset all profiling data."""
        self.timings.clear()


class PyTorchBlockGenerator:
    """
    Generator for PyTorch nn.Module code for group blocks.
    
    Converts GroupBlockDefinition into reusable nn.Module subclasses
    with proper initialization and forward pass logic.
    """
    
    def __init__(
        self,
        group_definitions: List[Dict[str, Any]],
        shape_computer: Optional[GroupBlockShapeComputer] = None
    ):
        """
        Initialize the block generator.
        
        Args:
            group_definitions: List of GroupBlockDefinition dictionaries
            shape_computer: Optional shape computer for internal shape inference
        """
        self.group_definitions = {defn['id']: defn for defn in group_definitions}
        self.generated_classes = {}  # Cache generated class code
        self.shape_computer = shape_computer or GroupBlockShapeComputer(self.group_definitions)
        
    def generate_all_block_classes(self) -> str:
        """
        Generate all block class definitions.
        
        Returns:
            String containing all block class definitions
        """
        if not self.group_definitions:
            return ""
            
        code_parts = []
        code_parts.append("# ============================================")
        code_parts.append("# Custom Block Definitions")
        code_parts.append("# ============================================\n")
        
        for defn_id, definition in self.group_definitions.items():
            block_class = self.generate_block_class(definition)
            code_parts.append(block_class)
            code_parts.append("\n")
            
        return "\n".join(code_parts)
    
    def generate_block_class(
        self,
        definition: Dict[str, Any],
        example_input_shape: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate nn.Module subclass for a single block definition.
        
        Args:
            definition: GroupBlockDefinition dictionary
            example_input_shape: Optional example input shape for computing internal shapes
            
        Returns:
            String containing the complete block class definition
        """
        block_name = definition['name']
        class_name = self._to_class_name(block_name)
        description = definition.get('description', '')
        
        # Get internal structure
        internal_structure = definition.get('internal_structure', {})
        internal_nodes = internal_structure.get('nodes', [])
        internal_edges = internal_structure.get('edges', [])
        port_mappings = internal_structure.get('portMappings', [])
        
        # Sort internal nodes topologically
        sorted_nodes = topological_sort(internal_nodes, internal_edges)
        
        # Compute internal shapes if example provided
        internal_shape_map = {}
        if example_input_shape:
            internal_shape_map, _ = self.shape_computer.compute_internal_shapes(
                internal_nodes,
                internal_edges,
                port_mappings,
                example_input_shape,
                block_name
            )
        else:
            # Fallback to old behavior without shape computer
            internal_shape_map, _ = infer_shapes(sorted_nodes, internal_edges)
        
        # Generate __init__ method
        init_method = self._generate_init_method(sorted_nodes, internal_shape_map, port_mappings)
        
        # Generate forward method
        forward_method = self._generate_forward_method(
            sorted_nodes, internal_edges, internal_shape_map, port_mappings
        )
        
        # Build class docstring
        docstring = self._generate_block_docstring(
            block_name, description, port_mappings, sorted_nodes
        )
        
        # Assemble the complete class
        class_code = f'''class {class_name}(nn.Module):
    """{docstring}"""

{init_method}

{forward_method}'''
        
        # Cache the generated class
        self.generated_classes[definition['id']] = class_name
        
        return class_code
    
    def _generate_init_method(
        self,
        nodes: List[Dict[str, Any]],
        shape_map: Dict[str, Dict[str, Any]],
        port_mappings: List[Dict[str, Any]]
    ) -> str:
        """Generate __init__ method with layer instantiation."""
        lines = []

        # Detect which shape parameters are needed by scanning nodes
        needs_in_channels = False
        needs_in_features = False
        needs_num_features = False

        for node in nodes:
            node_type = get_node_type(node)
            if node_type in ('input', 'dataloader', 'output'):
                continue
            if node_type == 'conv2d':
                needs_in_channels = True
            elif node_type == 'linear':
                needs_in_features = True
            elif node_type in ('batchnorm', 'batchnorm2d'):
                needs_num_features = True

        # Generate __init__ signature with detected parameters
        params = []
        if needs_in_channels:
            params.append("in_channels=None")
        if needs_in_features:
            params.append("in_features=None")
        if needs_num_features:
            params.append("num_features=None")

        if params:
            lines.append(f"    def __init__(self, {', '.join(params)}):")
        else:
            lines.append("    def __init__(self):")

        lines.append('        """Initialize all internal layers."""')
        lines.append(f"        super().__init__()")
        lines.append("")

        # Track which nodes need to be instantiated and which is first of each type
        layer_count = {}
        first_layer_of_type = {}

        for idx, node in enumerate(nodes):
            node_id = node['id']
            node_type = get_node_type(node)
            config = node.get('data', {}).get('config', {})
            shape_info = shape_map.get(node_id, {})

            # Skip input/output nodes
            if node_type in ('input', 'dataloader', 'output'):
                continue

            # Track if this is the first layer of its type
            is_first = node_type not in first_layer_of_type
            if is_first:
                first_layer_of_type[node_type] = node_id

            # Generate layer instantiation
            layer_name = self._get_internal_layer_name(node_type, node_id, layer_count)
            layer_class_name = self._get_layer_class_name_for_node(node_type, config)

            # Generate instantiation with proper arguments
            instantiation = self._generate_layer_instantiation_line(
                layer_name, layer_class_name, node_type, shape_info, config, is_first
            )

            if instantiation:
                lines.append(f"        {instantiation}")

        return "\n".join(lines)
    
    def _generate_forward_method(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        shape_map: Dict[str, Dict[str, Any]],
        port_mappings: List[Dict[str, Any]]
    ) -> str:
        """Generate forward method with internal connection logic."""
        lines = []
        
        # Determine input parameters from port mappings
        input_ports = [pm for pm in port_mappings if pm['type'] == 'input']
        output_ports = [pm for pm in port_mappings if pm['type'] == 'output']
        
        # Generate method signature
        if len(input_ports) == 1:
            lines.append("    def forward(self, x: torch.Tensor) -> torch.Tensor:")
        else:
            param_names = [f"input_{i}" for i in range(len(input_ports))]
            params = ", ".join([f"{name}: torch.Tensor" for name in param_names])
            lines.append(f"    def forward(self, {params}) -> torch.Tensor:")
        
        lines.append('        """')
        lines.append('        Forward pass through the block.')
        lines.append('')
        lines.append('        Args:')
        if len(input_ports) == 1:
            lines.append('            x: Input tensor')
        else:
            for i, port in enumerate(input_ports):
                label = port.get('externalPortLabel', f'input_{i}')
                lines.append(f'            input_{i}: {label}')
        lines.append('')
        lines.append('        Returns:')
        if len(output_ports) == 1:
            lines.append('            Output tensor')
        else:
            lines.append('            Tuple of output tensors')
        lines.append('        """')
        
        # Build edge map for finding inputs
        edge_map = {}
        for edge in edges:
            target = edge.get('target')
            source = edge.get('source')
            if target not in edge_map:
                edge_map[target] = []
            edge_map[target].append(source)
        
        # Map internal node IDs to variable names
        var_map = {}
        layer_count = {}
        
        # Map input ports to initial variables
        for i, port in enumerate(input_ports):
            internal_node_id = port['internalNodeId']
            if len(input_ports) == 1:
                var_map[internal_node_id] = 'x'
            else:
                var_map[internal_node_id] = f'input_{i}'
        
        # Generate forward pass for each internal node
        for node in nodes:
            node_id = node['id']
            node_type = get_node_type(node)
            config = node.get('data', {}).get('config', {})
            
            # Skip input/output nodes
            if node_type in ('input', 'dataloader', 'output'):
                # Input nodes are already mapped
                if node_id not in var_map:
                    var_map[node_id] = 'x'
                continue
            
            # Get layer name
            layer_name = self._get_internal_layer_name(node_type, node_id, layer_count)
            
            # Get input variable(s)
            incoming = edge_map.get(node_id, [])
            if not incoming:
                # No incoming edges, might be an input node we missed
                input_var = 'x'
            elif len(incoming) == 1:
                input_var = var_map.get(incoming[0], 'x')
            else:
                # Multiple inputs (for concat, add, etc.)
                input_vars = [var_map.get(src, 'x') for src in incoming]
                input_var = f"[{', '.join(input_vars)}]"
            
            # Generate output variable name (sanitize node_id to avoid hyphens)
            output_var = f"x_{node_id[:8].replace('-', '_')}"
            var_map[node_id] = output_var
            
            # Generate forward line
            if node_type in ('concat', 'add'):
                lines.append(f"        {output_var} = self.{layer_name}({input_var})")
            else:
                lines.append(f"        {output_var} = self.{layer_name}({input_var})")
        
        # Map output ports to return values
        if len(output_ports) == 1:
            output_node_id = output_ports[0]['internalNodeId']
            output_var = var_map.get(output_node_id, 'x')
            lines.append(f"        return {output_var}")
        else:
            output_vars = []
            for port in output_ports:
                output_node_id = port['internalNodeId']
                output_vars.append(var_map.get(output_node_id, 'x'))
            lines.append(f"        return ({', '.join(output_vars)})")
        
        return "\n".join(lines)
    
    def _generate_block_docstring(
        self,
        block_name: str,
        description: str,
        port_mappings: List[Dict[str, Any]],
        nodes: List[Dict[str, Any]]
    ) -> str:
        """Generate comprehensive docstring for block class."""
        lines = []
        lines.append(f"Custom Block: {block_name}")
        lines.append("")
        
        if description:
            lines.append(description)
            lines.append("")
        
        lines.append("This block encapsulates a reusable subgraph of layers.")
        lines.append("")
        
        # Document ports
        input_ports = [pm for pm in port_mappings if pm['type'] == 'input']
        output_ports = [pm for pm in port_mappings if pm['type'] == 'output']
        
        if input_ports:
            lines.append("Input Ports:")
            for port in input_ports:
                label = port.get('externalPortLabel', 'input')
                lines.append(f"    - {label}")
        
        if output_ports:
            lines.append("")
            lines.append("Output Ports:")
            for port in output_ports:
                label = port.get('externalPortLabel', 'output')
                lines.append(f"    - {label}")
        
        lines.append("")
        lines.append(f"Internal Layers: {len([n for n in nodes if get_node_type(n) not in ('input', 'dataloader', 'output')])}")
        
        return "\n    ".join(lines)
    
    def _generate_layer_instantiation_line(
        self,
        layer_name: str,
        layer_class_name: str,
        node_type: str,
        shape_info: Dict[str, Any],
        config: Dict[str, Any],
        is_first: bool = False
    ) -> str:
        """Generate layer instantiation line with proper arguments using shape_map values."""
        # Determine if layer needs shape arguments
        if node_type == 'conv2d':
            if is_first:
                # Use parameter for first conv2d layer
                return f"self.{layer_name} = {layer_class_name}(in_channels=in_channels)"
            else:
                # Use computed value from shape map (no hardcoded defaults)
                in_channels = shape_info.get('in_channels')
                if in_channels is not None:
                    return f"self.{layer_name} = {layer_class_name}(in_channels={in_channels})"
                else:
                    # If shape inference failed, use parameter
                    logger.warning(f"No in_channels in shape_map for {layer_name}, using parameter")
                    return f"self.{layer_name} = {layer_class_name}(in_channels=in_channels)"
        elif node_type == 'linear':
            if is_first:
                # Use parameter for first linear layer
                return f"self.{layer_name} = {layer_class_name}(in_features=in_features)"
            else:
                # Use computed value from shape map (no hardcoded defaults)
                in_features = shape_info.get('in_features')
                if in_features is not None:
                    return f"self.{layer_name} = {layer_class_name}(in_features={in_features})"
                else:
                    # If shape inference failed, use parameter
                    logger.warning(f"No in_features in shape_map for {layer_name}, using parameter")
                    return f"self.{layer_name} = {layer_class_name}(in_features=in_features)"
        elif node_type in ('batchnorm', 'batchnorm2d'):
            if is_first:
                # Use parameter for first batchnorm layer
                return f"self.{layer_name} = {layer_class_name}(num_features=num_features)"
            else:
                # Use computed value from shape map (no hardcoded defaults)
                num_features = shape_info.get('num_features')
                if num_features is not None:
                    return f"self.{layer_name} = {layer_class_name}(num_features={num_features})"
                else:
                    # If shape inference failed, use parameter
                    logger.warning(f"No num_features in shape_map for {layer_name}, using parameter")
                    return f"self.{layer_name} = {layer_class_name}(num_features=num_features)"
        else:
            return f"self.{layer_name} = {layer_class_name}()"
    
    def _get_internal_layer_name(
        self,
        node_type: str,
        node_id: str,
        layer_count: Dict[str, int]
    ) -> str:
        """Generate unique layer variable name for internal node."""
        # Use node_id suffix for uniqueness (sanitize to avoid hyphens)
        suffix = node_id[:8].replace('-', '_')
        base_name = node_type.replace('_', '')

        # Track count for this type
        if node_type not in layer_count:
            layer_count[node_type] = 0
        layer_count[node_type] += 1

        return f"{base_name}_{suffix}"
    
    def _get_layer_class_name_for_node(
        self,
        node_type: str,
        config: Dict[str, Any]
    ) -> str:
        """
        Get the layer class name that will be used in the main model.

        Uses the shared _build_layer_class_name helper to ensure consistency
        with get_layer_class_name and enable proper deduplication.
        """
        return _build_layer_class_name(node_type, config)
    
    def _to_class_name(self, name: str) -> str:
        """Convert block name to valid Python class name."""
        import re
        # Remove special characters and convert to PascalCase
        name = re.sub(r'[^a-zA-Z0-9]', ' ', name)
        name = ''.join(word.capitalize() for word in name.split())
        if not name:
            return 'CustomBlock'
        if name[0].isdigit():
            name = 'Block' + name
        return name + 'Block'
    
    def get_block_class_name(self, definition_id: str) -> Optional[str]:
        """
        Get the generated class name for a block definition.
        
        Args:
            definition_id: ID of the GroupBlockDefinition
            
        Returns:
            Class name if generated, None otherwise
        """
        return self.generated_classes.get(definition_id)


def generate_pytorch_code(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    project_name: str = "GeneratedModel",
    group_definitions: Optional[List[Dict[str, Any]]] = None
) -> Tuple[Dict[str, str], List[Exception]]:
    """
    Generate complete PyTorch code including model, training, and data loading.
    Each layer gets its own reusable class, all combined in a main model class.

    Args:
        nodes: List of node dictionaries from architecture
        edges: List of edge dictionaries defining connections
        project_name: Name for the generated model class
        group_definitions: Optional list of GroupBlockDefinition dictionaries

    Returns:
        Tuple of (dictionary with keys: 'model', 'train', 'dataset', 'config', list of errors)
    """
    # Topologically sort nodes
    sorted_nodes = topological_sort(nodes, edges)

    # Initialize block generator if we have group definitions
    block_generator = None
    group_def_dict = None
    shape_computer = None
    if group_definitions:
        # Convert list to dict for shape inference
        group_def_dict = {defn['id']: defn for defn in group_definitions}
        # Create shape computer for reuse
        shape_computer = GroupBlockShapeComputer(group_def_dict)
        # Create block generator with shape computer
        block_generator = PyTorchBlockGenerator(group_definitions, shape_computer)

    # Infer shapes through the graph (now with group definitions)
    shape_map, shape_errors = infer_shapes(sorted_nodes, edges, group_def_dict)

    # Validate computed shapes for critical issues
    validation_errors = validate_shape_map(sorted_nodes, shape_map)
    if validation_errors:
        logger.warning(f"Shape validation found {len(validation_errors)} potential issues")
        shape_errors.extend(validation_errors)

    # Generate different components
    model_code = generate_model_file(sorted_nodes, edges, project_name, shape_map, block_generator, shape_errors)
    train_code = generate_training_script(project_name)
    dataset_code = generate_dataset_class(nodes)
    config_code = generate_config_file(nodes)

    return {
        'model': model_code,
        'train': train_code,
        'dataset': dataset_code,
        'config': config_code
    }, shape_errors


def generate_single_layer_class(
    node: Dict[str, Any],
    node_index: int = 0,
    shape_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate professional class-based code for a single layer.
    Used for individual node preview in the visual editor.

    Args:
        node: Node dictionary with type, data, config
        node_index: Index for layer naming (default: 0)
        shape_info: Optional shape information dict. If None, extracted from node.

    Returns:
        String containing the complete layer class definition
    """
    # Extract node information
    node_type = get_node_type(node)
    config = node.get('data', {}).get('config', {})

    # Extract or infer shape information
    if shape_info is None:
        shape_info = extract_shape_info_from_node(node)

    # Skip nodes that don't generate layers
    if node_type in ('input', 'dataloader', 'output'):
        return f'''# {node_type.upper()} Node
# This is handled automatically during model execution
# Input shape: {shape_info.get('out_channels', '?')} channels or {shape_info.get('out_features', '?')} features'''

    # Generate the layer class using existing function
    layer_class = generate_layer_class(node, node_index, config, node_type, shape_info)

    if layer_class:
        return layer_class
    else:
        return f'''# Unsupported layer type: {node_type}
# Please use the full export to generate complete model code'''


def extract_shape_info_from_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract shape information from a single node's metadata.

    Args:
        node: Node dictionary

    Returns:
        Dictionary with shape information (in_channels, out_channels, in_features, out_features, etc.)
    """
    shape_info = {}
    node_type = get_node_type(node)
    config = node.get('data', {}).get('config', {})

    # Try to get shape from node metadata
    input_shape = node.get('data', {}).get('inputShape', {})
    output_shape = node.get('data', {}).get('outputShape', {})

    # Extract from inputShape/outputShape if available
    if input_shape and isinstance(input_shape, dict):
        dims = input_shape.get('dims', [])
        if len(dims) >= 4:  # NCHW format
            shape_info['in_channels'] = dims[1]
            shape_info['in_height'] = dims[2]
            shape_info['in_width'] = dims[3]
        elif len(dims) >= 2:
            shape_info['in_features'] = dims[1]

    if output_shape and isinstance(output_shape, dict):
        dims = output_shape.get('dims', [])
        if len(dims) >= 4:  # NCHW format
            shape_info['out_channels'] = dims[1]
            shape_info['out_height'] = dims[2]
            shape_info['out_width'] = dims[3]
        elif len(dims) >= 2:
            shape_info['out_features'] = dims[1]

    # Infer from config if not in metadata
    if node_type == 'conv2d':
        if 'in_channels' not in shape_info:
            shape_info['in_channels'] = 3  # Default
        if 'out_channels' not in shape_info:
            shape_info['out_channels'] = config.get('out_channels', 64)
        # Try to estimate output dimensions if not provided
        if 'out_height' not in shape_info:
            shape_info['out_height'] = '?'
        if 'out_width' not in shape_info:
            shape_info['out_width'] = '?'

    elif node_type == 'linear':
        if 'in_features' not in shape_info:
            shape_info['in_features'] = 512  # Default
        if 'out_features' not in shape_info:
            shape_info['out_features'] = config.get('out_features', 128)

    elif node_type == 'batchnorm':
        if 'num_features' not in shape_info:
            shape_info['num_features'] = shape_info.get('out_channels', shape_info.get('in_channels', 64))

    elif node_type == 'flatten':
        if 'out_features' not in shape_info:
            # Estimate based on typical conv output
            channels = shape_info.get('in_channels', 512)
            height = shape_info.get('in_height', 7)
            width = shape_info.get('in_width', 7)
            if isinstance(height, int) and isinstance(width, int):
                shape_info['out_features'] = channels * height * width
            else:
                shape_info['out_features'] = '?'

    return shape_info


def topological_sort(nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
    """Sort nodes in topological order based on edges using Kahn's algorithm"""
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

    # Return nodes in sorted order
    return [node_map[node_id] for node_id in sorted_ids if node_id in node_map]


def extract_output_shape_from_metadata(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract output shape from node's frontend-provided metadata.

    The frontend computes output shapes accurately during the visual design phase
    and stores them in node.data.outputShape. This function extracts those
    pre-computed shapes, which are considered authoritative.

    Args:
        node: Node dictionary with potential data.outputShape metadata

    Returns:
        Dictionary with shape keys (out_channels, out_features, etc.) or None if
        metadata is incomplete/missing
    """
    output_shape = node.get('data', {}).get('outputShape', {})
    if not output_shape or not isinstance(output_shape, dict):
        return None

    dims = output_shape.get('dims', [])
    if not dims:
        return None

    shape_info = {}

    # PyTorch uses NCHW format: [batch, channels, height, width]
    if len(dims) == 4:
        shape_info['out_channels'] = dims[1]
        shape_info['out_height'] = dims[2]
        shape_info['out_width'] = dims[3]
    elif len(dims) == 2:  # [batch, features] - for Linear/Flatten output
        shape_info['out_features'] = dims[1]
    else:
        # Unusual shape format - log for debugging but don't fail
        logger.debug(f"Unusual output shape dims: {dims}")
        return None

    return shape_info


def infer_shapes(
    nodes: List[Dict],
    edges: List[Dict],
    group_definitions: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Dict[str, Any]], List[Exception]]:
    """
    Infer input/output shapes for each layer in the graph.
    Enhanced to handle group blocks properly.

    Args:
        nodes: List of node dictionaries
        edges: List of edge dictionaries
        group_definitions: Optional map of group definition IDs to definitions

    Returns:
        Tuple of (dictionary mapping node_id to shape info, list of errors encountered)
    """
    shape_map = {}
    errors = []

    # Initialize shape computer for group blocks
    shape_computer = None
    if group_definitions:
        shape_computer = GroupBlockShapeComputer(group_definitions)

    # Build edge map for finding inputs
    edge_map = {}
    for edge in edges:
        target = edge.get('target')
        source = edge.get('source')
        if target not in edge_map:
            edge_map[target] = []
        edge_map[target].append(source)

    # Topologically sort nodes to ensure we process layers in dependency order
    # This is CRITICAL: we must compute upstream layer shapes before downstream layers
    sorted_nodes = topological_sort(nodes, edges)

    # Process nodes in topological order
    for node in sorted_nodes:
        node_id = node['id']
        node_type = get_node_type(node)
        config = node.get('data', {}).get('config', {})

        # Get incoming edges
        incoming = edge_map.get(node_id, [])

        # ========== PHASE 1: Extract output metadata (if available) ==========
        # Frontend provides accurate output shapes in metadata
        metadata_shape = extract_output_shape_from_metadata(node)
        shape_info = metadata_shape if metadata_shape else {}

        # ========== PHASE 2: Compute input dimensions from upstream nodes ==========
        # Input dimensions ALWAYS come from upstream, regardless of metadata
        # This is critical for layers like Conv2d, Linear, BatchNorm

        if node_type == 'input':
            # Input nodes have no upstream - parse from config if metadata doesn't exist
            if not metadata_shape:
                shape_str = config.get('shape', '[1, 3, 224, 224]')
                try:
                    # Try to parse shape
                    shape = json.loads(shape_str)
                    if len(shape) >= 4:
                        shape_info['out_channels'] = shape[1]  # NCHW format
                        shape_info['out_height'] = shape[2]
                        shape_info['out_width'] = shape[3]
                    elif len(shape) >= 2:
                        shape_info['out_features'] = shape[1]
                except (json.JSONDecodeError, ValueError, KeyError, IndexError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse input shape for node {node_id}: {e}. "
                        f"Using default shape [1, 3, 224, 224] (NCHW)"
                    )
                    errors.append(ShapeInferenceError(
                        node_id=node_id,
                        node_type=node_type,
                        reason=f"Failed to parse shape configuration: {str(e)}",
                        suggestion="Check that the input shape is a valid JSON array like [1, 3, 224, 224]"
                    ))
                    shape_info['out_channels'] = 3
                    shape_info['out_height'] = 224
                    shape_info['out_width'] = 224

        elif node_type == 'conv2d':
            # Get input channels from upstream layer (ALWAYS required)
            if incoming and incoming[0] in shape_map:
                try:
                    upstream_shape = safe_get_shape_data(
                        shape_map=shape_map,
                        node_id=node_id,
                        upstream_node_id=incoming[0],
                        required_keys=['out_channels'],
                        default_values={'out_channels': 3}
                    )
                    shape_info['in_channels'] = upstream_shape['out_channels']
                except (MissingShapeDataError, ShapeInferenceError) as e:
                    logger.warning(f"Shape inference warning for node {node_id}: {e}. Using default.")
                    errors.append(e)
                    shape_info['in_channels'] = 3
            else:
                shape_info['in_channels'] = 3

            # Output channels: use metadata if available, otherwise config
            if 'out_channels' not in shape_info:
                shape_info['out_channels'] = config.get('out_channels', 64)

            # Spatial dimensions: use metadata if available, otherwise calculate
            if 'out_height' not in shape_info or 'out_width' not in shape_info:
                if incoming and incoming[0] in shape_map:
                    try:
                        prev_shape = safe_get_shape_data(
                            shape_map=shape_map,
                            node_id=node_id,
                            upstream_node_id=incoming[0],
                            required_keys=['out_height', 'out_width'],
                            default_values=None
                        )
                        kernel_size = config.get('kernel_size', 3)
                        stride = config.get('stride', 1)
                        padding = config.get('padding', 0)

                        shape_info['out_height'] = (prev_shape['out_height'] + 2*padding - kernel_size) // stride + 1
                        shape_info['out_width'] = (prev_shape['out_width'] + 2*padding - kernel_size) // stride + 1
                    except (MissingShapeDataError, ShapeInferenceError) as e:
                        logger.warning(f"Could not compute spatial dimensions for conv2d {node_id}: {e}")
                        errors.append(e)

        elif node_type == 'maxpool':
            # MaxPool preserves channels from upstream
            if incoming and incoming[0] in shape_map:
                try:
                    prev_shape = safe_get_shape_data(
                        shape_map=shape_map,
                        node_id=node_id,
                        upstream_node_id=incoming[0],
                        required_keys=['out_channels'],
                        default_values={'out_channels': 64}
                    )
                    shape_info['out_channels'] = prev_shape['out_channels']
                except (MissingShapeDataError, ShapeInferenceError) as e:
                    logger.warning(f"Shape inference warning for maxpool {node_id}: {e}")
                    errors.append(e)
                    shape_info['out_channels'] = 64
            else:
                shape_info['out_channels'] = 64

            # Spatial dimensions: use metadata if available, otherwise calculate
            if 'out_height' not in shape_info or 'out_width' not in shape_info:
                if incoming and incoming[0] in shape_map:
                    try:
                        prev_shape = safe_get_shape_data(
                            shape_map=shape_map,
                            node_id=node_id,
                            upstream_node_id=incoming[0],
                            required_keys=['out_height', 'out_width'],
                            default_values={'out_height': 7, 'out_width': 7}
                        )
                        kernel_size = config.get('kernel_size', 2)
                        stride = config.get('stride', 2)
                        padding = config.get('padding', 0)

                        shape_info['out_height'] = (prev_shape['out_height'] + 2*padding - kernel_size) // stride + 1
                        shape_info['out_width'] = (prev_shape['out_width'] + 2*padding - kernel_size) // stride + 1
                    except (MissingShapeDataError, ShapeInferenceError) as e:
                        logger.warning(f"Could not compute spatial dimensions for maxpool {node_id}: {e}")
                        errors.append(e)

        elif node_type == 'flatten':
            # Flatten converts spatial dimensions to features
            # Use metadata if available, otherwise calculate from upstream
            if 'out_features' not in shape_info:
                if incoming and incoming[0] in shape_map:
                    try:
                        prev_shape = safe_get_shape_data(
                            shape_map=shape_map,
                            node_id=node_id,
                            upstream_node_id=incoming[0],
                            required_keys=['out_channels', 'out_height', 'out_width'],
                            default_values={'out_channels': 64, 'out_height': 7, 'out_width': 7}
                        )
                        channels = prev_shape['out_channels']
                        height = prev_shape['out_height']
                        width = prev_shape['out_width']
                        shape_info['out_features'] = channels * height * width
                    except (MissingShapeDataError, ShapeInferenceError) as e:
                        logger.warning(f"Shape inference warning for flatten {node_id}: {e}")
                        errors.append(e)
                        shape_info['out_features'] = 3136  # 64 * 7 * 7
                else:
                    shape_info['out_features'] = 3136  # Default

        elif node_type == 'linear':
            # Get input features from upstream layer (ALWAYS required)
            if incoming and incoming[0] in shape_map:
                try:
                    upstream_shape = safe_get_shape_data(
                        shape_map=shape_map,
                        node_id=node_id,
                        upstream_node_id=incoming[0],
                        required_keys=['out_features'],
                        default_values={'out_features': 512}
                    )
                    shape_info['in_features'] = upstream_shape['out_features']
                except (MissingShapeDataError, ShapeInferenceError) as e:
                    logger.warning(f"Shape inference warning for linear {node_id}: {e}")
                    errors.append(e)
                    shape_info['in_features'] = 512
            else:
                shape_info['in_features'] = 512

            # Output features: use metadata if available, otherwise config
            if 'out_features' not in shape_info:
                shape_info['out_features'] = config.get('out_features', 128)

        elif node_type == 'batchnorm':
            # BatchNorm preserves all dimensions from upstream
            if incoming and incoming[0] in shape_map:
                try:
                    prev_shape = safe_get_shape_data(
                        shape_map=shape_map,
                        node_id=node_id,
                        upstream_node_id=incoming[0],
                        required_keys=['out_channels'],
                        default_values={'out_channels': 64}
                    )
                    shape_info['num_features'] = prev_shape['out_channels']
                    shape_info['out_channels'] = shape_info['num_features']
                    # Copy spatial dimensions if they exist and not in metadata
                    if 'out_height' not in shape_info and 'out_height' in prev_shape:
                        shape_info['out_height'] = prev_shape['out_height']
                    if 'out_width' not in shape_info and 'out_width' in prev_shape:
                        shape_info['out_width'] = prev_shape['out_width']
                except (MissingShapeDataError, ShapeInferenceError) as e:
                    logger.warning(f"Shape inference warning for batchnorm {node_id}: {e}")
                    errors.append(e)
                    shape_info['num_features'] = 64
                    shape_info['out_channels'] = 64
            else:
                shape_info['num_features'] = 64
                shape_info['out_channels'] = 64

        elif node_type == 'group':
            # Group blocks: Use metadata if available, otherwise compute from internal structure
            if not metadata_shape:
                # No metadata - compute output shape using GroupBlockShapeComputer
                if shape_computer:
                    group_def_id = node.get('data', {}).get('groupDefinitionId')

                    if group_def_id and incoming and incoming[0] in shape_map:
                        # Get input shape from upstream node
                        input_shape = shape_map[incoming[0]]

                        # Compute output shape using internal structure
                        output_shape, shape_errors = shape_computer.compute_output_shape(
                            group_def_id,
                            input_shape
                        )

                        # Collect errors from shape computation
                        errors.extend(shape_errors)

                        if output_shape:
                            shape_info = output_shape
                            logger.debug(f"Computed shape for group block {node_id}: {output_shape}")
                        else:
                            # Fallback: copy input shape
                            shape_info = input_shape.copy()
                            logger.warning(f"Failed to compute shape for group block {node_id}, using input shape")
                    elif group_def_id and not (incoming and incoming[0] in shape_map):
                        # Group definition exists but no valid input
                        shape_info = {'out_channels': 3, 'out_height': 224, 'out_width': 224}
                        logger.warning(f"Group block {node_id} has no valid input, using default shape")
                    elif not group_def_id and incoming and incoming[0] in shape_map:
                        # No definition found, copy input shape
                        shape_info = shape_map[incoming[0]].copy()
                        logger.warning(f"No group definition ID found for node {node_id}, using input shape")
                    else:
                        # No definition and no input, use default
                        shape_info = {'out_channels': 3, 'out_height': 224, 'out_width': 224}
                        logger.warning(f"Group block {node_id} has no definition ID and no input, using default shape")
                else:
                    # No shape computer available, fallback to old behavior
                    if incoming and incoming[0] in shape_map:
                        prev_shape = shape_map[incoming[0]]
                        shape_info.update(prev_shape)
                    else:
                        shape_info['out_channels'] = 3
                        shape_info['out_height'] = 224
                        shape_info['out_width'] = 224
                    logger.warning(f"No shape computer available for group block {node_id}, using fallback behavior")

        else:
            # For other layers: Use metadata if available, otherwise preserve upstream shape
            if not metadata_shape and incoming and incoming[0] in shape_map:
                prev_shape = shape_map[incoming[0]]
                shape_info.update(prev_shape)
    
        shape_map[node_id] = shape_info

    return shape_map, errors


def validate_shape_map(
    nodes: List[Dict],
    shape_map: Dict[str, Dict[str, Any]]
) -> List[Exception]:
    """
    Validate computed shape map for common critical issues.

    This catches problems that would cause runtime errors in generated code:
    - Missing shape information
    - Invalid dimensions (zero or negative)
    - Type-specific requirements not met

    Args:
        nodes: List of all nodes
        shape_map: Computed shape mapping

    Returns:
        List of validation errors (as exceptions for consistency with shape_errors)
    """
    errors = []

    for node in nodes:
        node_id = node['id']
        node_type = get_node_type(node)

        # Skip non-layer nodes
        if node_type in ('input', 'output', 'dataloader', 'group'):
            continue

        shape_info = shape_map.get(node_id)

        # Critical: Shape info must exist
        if not shape_info:
            errors.append(ShapeInferenceError(
                node_id=node_id,
                node_type=node_type,
                reason="No shape information computed for node",
                suggestion="Check that node has valid upstream connections and metadata"
            ))
            continue

        # Type-specific validation
        if node_type == 'linear':
            # Linear MUST have in_features
            if 'in_features' not in shape_info:
                errors.append(ShapeInferenceError(
                    node_id=node_id,
                    node_type=node_type,
                    reason="Missing required in_features for Linear layer",
                    suggestion="Check upstream Flatten or Linear layer output shape"
                ))
            # in_features must be positive
            elif shape_info.get('in_features', 0) <= 0:
                errors.append(ShapeInferenceError(
                    node_id=node_id,
                    node_type=node_type,
                    reason=f"Invalid in_features={shape_info.get('in_features')} (must be > 0)",
                    suggestion="Check upstream layer produces valid output shape"
                ))

        elif node_type == 'conv2d':
            # Conv2d MUST have in_channels
            if 'in_channels' not in shape_info:
                errors.append(ShapeInferenceError(
                    node_id=node_id,
                    node_type=node_type,
                    reason="Missing required in_channels for Conv2d layer",
                    suggestion="Check upstream Conv2d or Input layer provides channels"
                ))

        elif node_type == 'flatten':
            # Flatten MUST produce out_features
            if 'out_features' not in shape_info:
                errors.append(ShapeInferenceError(
                    node_id=node_id,
                    node_type=node_type,
                    reason="Flatten layer must produce out_features",
                    suggestion="Check upstream layer has spatial dimensions (NCHW format)"
                ))
            elif shape_info.get('out_features', 0) <= 0:
                errors.append(ShapeInferenceError(
                    node_id=node_id,
                    node_type=node_type,
                    reason=f"Invalid out_features={shape_info.get('out_features')} (must be > 0)",
                    suggestion="Check upstream layer output dimensions are valid"
                ))

    return errors


def collect_all_nodes_with_internals(
    main_nodes: List[Dict],
    block_generator: Optional[PyTorchBlockGenerator] = None
) -> List[Tuple[Dict, int, str]]:
    """
    Collect all nodes including internal nodes from group blocks.
    Returns list of tuples: (node, index, source_context)
    source_context is either 'main' or 'group_{group_def_id}'

    This ensures we generate layer classes for ALL nodes, not just main model nodes.
    """
    all_nodes = []
    node_index = 0

    # Add main model nodes
    for node in main_nodes:
        all_nodes.append((node, node_index, 'main'))
        node_index += 1

    # Add internal nodes from group definitions
    if block_generator:
        for group_def_id, group_def in block_generator.group_definitions.items():
            internal_structure = group_def.get('internal_structure', {})
            internal_nodes = internal_structure.get('nodes', [])

            for internal_node in internal_nodes:
                node_type = get_node_type(internal_node)
                # Skip input/output nodes
                if node_type not in ('input', 'dataloader', 'output'):
                    all_nodes.append((internal_node, node_index, f'group_{group_def_id}'))
                    node_index += 1

    return all_nodes


def get_layer_signature(node: Dict, config: Dict[str, Any], node_type: str) -> str:
    """
    Generate a unique signature for a layer based on its type and config.
    Used for deduplication - layers with same signature can share the same class.
    """
    if node_type == 'conv2d':
        return f"conv2d_{config.get('out_channels', 64)}_{config.get('kernel_size', 3)}_{config.get('stride', 1)}_{config.get('padding', 0)}_{config.get('dilation', 1)}"
    elif node_type == 'linear':
        return f"linear_{config.get('out_features', 128)}_{config.get('bias', True)}"
    elif node_type == 'maxpool':
        return f"maxpool_{config.get('kernel_size', 2)}_{config.get('stride', 2)}_{config.get('padding', 0)}"
    elif node_type == 'dropout':
        return f"dropout_{config.get('p', 0.5)}"
    elif node_type == 'batchnorm':
        return f"batchnorm_{config.get('eps', 1e-5)}_{config.get('momentum', 0.1)}_{config.get('affine', True)}"
    elif node_type == 'softmax':
        return f"softmax_{config.get('dim', 1)}"
    elif node_type == 'attention':
        return f"attention_{config.get('embed_dim', 512)}_{config.get('num_heads', 8)}_{config.get('dropout', 0.0)}"
    elif node_type == 'custom':
        return f"custom_{config.get('name', 'CustomLayer')}"
    else:
        # For layers without config (relu, flatten, etc.)
        return node_type


def _build_layer_class_name(node_type: str, config: Dict[str, Any]) -> str:
    """
    Single source of truth for layer class naming.
    Ensures consistency between group blocks and main model.

    This function generates class names based on node type and configuration,
    enabling natural deduplication where layers with identical configs share
    the same class definition.

    Args:
        node_type: The type of the node (e.g., 'conv2d', 'linear', 'relu')
        config: The configuration dictionary for the node

    Returns:
        A descriptive class name string
    """
    type_name = node_type.replace('_', '').title()

    if node_type == 'conv2d':
        channels = config.get('out_channels', 64)
        kernel = config.get('kernel_size', 3)
        return f"{type_name}Layer_{channels}ch_{kernel}x{kernel}"
    elif node_type == 'linear':
        features = config.get('out_features', 128)
        return f"{type_name}Layer_{features}units"
    elif node_type == 'maxpool':
        kernel = config.get('kernel_size', 2)
        return f"{type_name}Layer_{kernel}x{kernel}"
    elif node_type == 'dropout':
        p = config.get('p', 0.5)
        return f"{type_name}Layer_p{p:.2f}".replace('.', 'p')
    elif node_type == 'batchnorm':
        eps = config.get('eps', 1e-5)
        momentum = config.get('momentum', 0.1)
        return f"{type_name}Layer_eps{eps}_mom{momentum}".replace('.', 'p').replace('-', 'm')
    elif node_type == 'softmax':
        dim = config.get('dim', 1)
        return f"{type_name}Layer_dim{dim}"
    elif node_type == 'attention':
        embed_dim = config.get('embed_dim', 512)
        num_heads = config.get('num_heads', 8)
        return f"{type_name}Layer_{embed_dim}d_{num_heads}h"
    elif node_type == 'custom':
        name = config.get('name', 'CustomLayer')
        safe_name = name.replace(' ', '_').replace('-', '_')
        return f"CustomLayer_{safe_name}"
    elif node_type == 'add':
        return f"{type_name}Layer"
    elif node_type == 'concat':
        dim = config.get('dim', 1)
        return f"{type_name}Layer_dim{dim}"
    else:
        # For parameter-free layers (relu, flatten, etc.)
        return f"{type_name}Layer"


def generate_model_file(
    nodes: List[Dict],
    edges: List[Dict],
    project_name: str,
    shape_map: Dict[str, Dict[str, Any]],
    block_generator: Optional[PyTorchBlockGenerator] = None,
    shape_errors: Optional[List[Exception]] = None
) -> str:
    """
    Generate complete model.py file with layer classes and main model class.
    
    Args:
        nodes: List of node dictionaries
        edges: List of edge dictionaries
        project_name: Name for the generated model class
        shape_map: Dictionary mapping node_id to shape info
        block_generator: Optional block generator for group blocks
        shape_errors: Optional list of errors encountered during shape inference
        
    Returns:
        String containing the complete model.py file
    """
    if shape_errors is None:
        shape_errors = []

    class_name = to_class_name(project_name)

    # Generate block class definitions FIRST (if any) - this populates the cache
    block_classes_code = ""
    if block_generator:
        block_classes_code = block_generator.generate_all_block_classes()

    # COLLECT ALL NODES (main + internal from groups) and generate layer classes
    all_nodes_to_generate = collect_all_nodes_with_internals(nodes, block_generator)

    # DEDUPLICATE by signature and generate layer classes
    seen_signatures = set()
    layer_classes = []

    for node, idx, source_context in all_nodes_to_generate:
        node_type = get_node_type(node)
        config = node.get('data', {}).get('config', {})
        node_id = node['id']

        # Get shape info (use shape_map for main nodes, extract for internal)
        if source_context == 'main':
            shape_info = shape_map.get(node_id, {})
        else:
            shape_info = extract_shape_info_from_node(node)

        # Generate signature for deduplication
        signature = get_layer_signature(node, config, node_type)

        # Only generate if we haven't seen this signature before
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            layer_class_code = generate_layer_class(node, idx, config, node_type, shape_info)
            if layer_class_code:
                layer_classes.append(layer_class_code)

    # Now generate layer instantiations and forward pass for MAIN MODEL ONLY
    layer_instantiations = []
    forward_pass_lines = []

    # Build edge map for forward pass
    edge_map = {}
    for edge in edges:
        target = edge.get('target')
        source = edge.get('source')
        if target not in edge_map:
            edge_map[target] = []
        edge_map[target].append(source)

    var_map = {}  # Map node_id to variable name

    for idx, node in enumerate(nodes):
        node_id = node['id']
        node_type = get_node_type(node)
        config = node.get('data', {}).get('config', {})
        shape_info = shape_map.get(node_id, {})

        if node_type in ('input', 'dataloader', 'output'):
            # Skip input/output nodes
            var_map[node_id] = 'x' if not var_map else 'x'
            continue

        # Handle group blocks differently
        if node_type == 'group':
            # Get the group definition ID
            group_def_id = node.get('data', {}).get('groupDefinitionId')

            if block_generator and group_def_id:
                # Use the block class name from the generator
                block_class_name = block_generator.get_block_class_name(group_def_id)

                if block_class_name:
                    layer_name = f"block_{node_id.replace('-', '_')}"

                    # Get upstream node's output shape from shape_map
                    incoming = edge_map.get(node_id, [])
                    params = []
                    
                    if incoming and incoming[0] in shape_map:
                        # Get upstream node's output shape
                        upstream_shape = shape_map[incoming[0]]
                        
                        # Extract in_channels or in_features from upstream shape
                        # Pass in_channels if the upstream outputs channels (convolutional layers)
                        if 'out_channels' in upstream_shape:
                            in_channels = upstream_shape['out_channels']
                            params.append(f"in_channels={in_channels}")
                            logger.debug(f"Block {node_id}: passing in_channels={in_channels} from upstream node {incoming[0]}")
                        
                        # Pass in_features if the upstream outputs features (linear layers)
                        elif 'out_features' in upstream_shape:
                            in_features = upstream_shape['out_features']
                            params.append(f"in_features={in_features}")
                            logger.debug(f"Block {node_id}: passing in_features={in_features} from upstream node {incoming[0]}")
                        
                        # Pass num_features if the upstream outputs num_features (batch norm)
                        elif 'num_features' in upstream_shape:
                            num_features = upstream_shape['num_features']
                            params.append(f"num_features={num_features}")
                            logger.debug(f"Block {node_id}: passing num_features={num_features} from upstream node {incoming[0]}")
                        else:
                            # Upstream shape exists but doesn't have expected keys
                            logger.warning(f"Block {node_id}: upstream shape {upstream_shape} doesn't contain expected keys")
                    else:
                        # Handle case where no upstream exists (use input node shape)
                        # Look for input nodes in the graph
                        input_nodes = [n for n in nodes if get_node_type(n) == 'input']
                        if input_nodes and input_nodes[0]['id'] in shape_map:
                            input_shape = shape_map[input_nodes[0]['id']]
                            
                            # Use input node's output shape
                            if 'out_channels' in input_shape:
                                in_channels = input_shape['out_channels']
                                params.append(f"in_channels={in_channels}")
                                logger.debug(f"Block {node_id}: no upstream, using input shape in_channels={in_channels}")
                            elif 'out_features' in input_shape:
                                in_features = input_shape['out_features']
                                params.append(f"in_features={in_features}")
                                logger.debug(f"Block {node_id}: no upstream, using input shape in_features={in_features}")
                            else:
                                logger.warning(f"Block {node_id}: input shape {input_shape} doesn't contain expected keys")
                        else:
                            # No upstream and no input node, use defaults
                            logger.warning(f"Block {node_id}: no upstream connection and no input node found")

                    # Generate instantiation with computed parameters
                    # Each instance gets independent shape computation based on its position in the graph
                    if params:
                        layer_instantiations.append(f"self.{layer_name} = {block_class_name}({', '.join(params)})  # Instance at position {idx}")
                    else:
                        layer_instantiations.append(f"self.{layer_name} = {block_class_name}()  # Instance at position {idx}")

                    # Generate forward pass line
                    input_var = get_input_variable(incoming, var_map)
                    output_var = 'x'
                    forward_pass_lines.append(f"{output_var} = self.{layer_name}({input_var})")
                    var_map[node_id] = output_var
                else:
                    # Block class not found, skip
                    logger.warning(f"Block class not found for group definition {group_def_id}")
                    var_map[node_id] = 'x'
            else:
                # No block generator or definition ID, skip
                logger.warning(f"No block generator or definition ID for node {node_id}")
                var_map[node_id] = 'x'
            continue

        # For regular nodes, we already generated the layer class above (no need to generate again)

        # Generate layer instantiation for __init__
        layer_name = get_layer_variable_name(node_type, idx, config)
        layer_class_name = get_layer_class_name(node_type, idx, config)
        layer_init = generate_layer_instantiation(layer_class_name, layer_name, shape_info, node_type)
        if layer_init:
            layer_instantiations.append(layer_init)

        # Generate forward pass line
        incoming = edge_map.get(node_id, [])
        input_var = get_input_variable(incoming, var_map)
        output_var = 'x'

        forward_line = generate_forward_line(node_type, layer_name, input_var, output_var, shape_info)
        if forward_line:
            forward_pass_lines.append(forward_line)

        var_map[node_id] = output_var

    # Assemble the complete file
    code = f'''"""
Generated PyTorch Model
Architecture: {class_name}
Generated by VisionForge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


'''

    # Add block class definitions (already generated at the start)
    if block_classes_code:
        code += block_classes_code + '\n\n'

    # Add all layer class definitions
    for layer_class in layer_classes:
        code += layer_class + '\n\n'

    # Add main model class
    code += f'''
class {class_name}(nn.Module):
    """
    Main model class combining all layers.

    This model was automatically generated from a visual architecture.
    Each layer is implemented as a separate class for clarity and reusability.
    """

    def __init__(self):
        """Initialize all layers in the model."""
        super({class_name}, self).__init__()

'''

    # Add layer instantiations
    for init in layer_instantiations:
        code += f'        {init}\n'

    code += '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor after passing through all layers
        """
'''

    # Add forward pass lines
    for line in forward_pass_lines:
        code += f'        {line}\n'

    code += '''
        return x


def create_model() -> nn.Module:
    """
    Create and return an instance of the model.

    Returns:
        Initialized model ready for training or inference
    """
    model = {class_name}()
    return model


if __name__ == '__main__':
    # Create model and print summary
    model = create_model()
    print(f"Model: {class_name}")
    print(f"Total parameters: {{sum(p.numel() for p in model.parameters()):,}}")
    print(f"Trainable parameters: {{sum(p.numel() for p in model.parameters() if p.requires_grad):,}}")

    # Test forward pass with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"\\nInput shape: {{dummy_input.shape}}")
    print(f"Output shape: {{output.shape}}")
'''.format(class_name=class_name)

    return code


def generate_layer_class(
    node: Dict,
    idx: int,
    config: Dict[str, Any],
    node_type: str,
    shape_info: Dict[str, Any]
) -> Optional[str]:
    """Generate a complete layer class definition with documentation"""

    # Special node types that don't generate individual layer classes:
    # - input/output/dataloader: Architectural markers for graph structure
    # - group: Reusable components generated separately by BlockGenerator
    if node_type in ('input', 'output', 'dataloader', 'group'):
        return None

    class_name = get_layer_class_name(node_type, idx, config)

    if node_type == 'conv2d':
        in_channels = shape_info.get('in_channels', 3)
        out_channels = config.get('out_channels', 64)
        kernel_size = config.get('kernel_size', 3)
        stride = config.get('stride', 1)
        padding = config.get('padding', 0)
        dilation = config.get('dilation', 1)

        # Calculate output shape
        out_h = shape_info.get('out_height', '?')
        out_w = shape_info.get('out_width', '?')

        return f'''class {class_name}(nn.Module):
    """
    2D Convolutional Layer

    Applies a 2D convolution over an input signal composed of several input channels.

    Parameters:
        - Input channels: {in_channels}
        - Output channels: {out_channels}
        - Kernel size: {kernel_size}x{kernel_size}
        - Stride: {stride}
        - Padding: {padding}
        - Dilation: {dilation}

    Shape:
        - Input: [batch_size, {in_channels}, H, W]
        - Output: [batch_size, {out_channels}, {out_h}, {out_w}]
    """

    def __init__(self, in_channels: int = {in_channels}):
        """Initialize the convolutional layer."""
        super({class_name}, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            {out_channels},
            kernel_size={kernel_size},
            stride={stride},
            padding={padding},
            dilation={dilation}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional layer.

        Args:
            x: Input tensor of shape [batch, {in_channels}, H, W]

        Returns:
            Output tensor of shape [batch, {out_channels}, {out_h}, {out_w}]
        """
        # Apply convolution
        x = self.conv(x)
        return x'''

    elif node_type == 'linear':
        in_features = shape_info.get('in_features', 512)
        out_features = config.get('out_features', 128)
        bias = config.get('bias', True)

        return f'''class {class_name}(nn.Module):
    """
    Fully Connected (Linear) Layer

    Applies a linear transformation to the incoming data: y = xA^T + b

    Parameters:
        - Input features: {in_features}
        - Output features: {out_features}
        - Bias: {bias}

    Shape:
        - Input: [batch_size, {in_features}]
        - Output: [batch_size, {out_features}]
    """

    def __init__(self, in_features: int = {in_features}):
        """Initialize the linear layer."""
        super({class_name}, self).__init__()
        self.linear = nn.Linear(in_features, {out_features}, bias={bias})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linear layer.

        Args:
            x: Input tensor of shape [batch, {in_features}]

        Returns:
            Output tensor of shape [batch, {out_features}]
        """
        # Apply linear transformation
        x = self.linear(x)
        return x'''

    elif node_type == 'maxpool':
        kernel_size = config.get('kernel_size', 2)
        stride = config.get('stride', 2)
        padding = config.get('padding', 0)

        return f'''class {class_name}(nn.Module):
    """
    2D Max Pooling Layer

    Applies a 2D max pooling over an input signal.
    Reduces spatial dimensions while preserving channel count.

    Parameters:
        - Kernel size: {kernel_size}x{kernel_size}
        - Stride: {stride}
        - Padding: {padding}

    Shape:
        - Input: [batch_size, C, H, W]
        - Output: [batch_size, C, H/{stride}, W/{stride}]
    """

    def __init__(self):
        """Initialize the max pooling layer."""
        super({class_name}, self).__init__()
        self.pool = nn.MaxPool2d(
            kernel_size={kernel_size},
            stride={stride},
            padding={padding}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the pooling layer.

        Args:
            x: Input tensor of shape [batch, C, H, W]

        Returns:
            Output tensor with reduced spatial dimensions
        """
        # Apply max pooling
        x = self.pool(x)
        return x'''

    elif node_type == 'flatten':
        out_features = shape_info.get('out_features', '?')

        return f'''class {class_name}(nn.Module):
    """
    Flatten Layer

    Flattens a contiguous range of dimensions into a tensor.
    Commonly used to transition from convolutional layers to fully connected layers.

    Shape:
        - Input: [batch_size, C, H, W]
        - Output: [batch_size, C*H*W] = [batch_size, {out_features}]
    """

    def __init__(self):
        """Initialize the flatten layer."""
        super({class_name}, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the flatten layer.

        Args:
            x: Input tensor of shape [batch, C, H, W]

        Returns:
            Output tensor of shape [batch, C*H*W]
        """
        # Flatten spatial and channel dimensions
        x = self.flatten(x)
        return x'''

    elif node_type == 'relu':
        return f'''class {class_name}(nn.Module):
    """
    ReLU Activation Layer

    Applies the rectified linear unit function element-wise: ReLU(x) = max(0, x)
    Introduces non-linearity to the model.

    Shape:
        - Input: [batch_size, *] (any shape)
        - Output: [batch_size, *] (same shape as input)
    """

    def __init__(self):
        """Initialize the ReLU activation."""
        super({class_name}, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the activation.

        Args:
            x: Input tensor

        Returns:
            Output tensor with ReLU applied element-wise
        """
        # Apply ReLU activation
        x = self.relu(x)
        return x'''

    elif node_type == 'dropout':
        p = config.get('p', 0.5)

        return f'''class {class_name}(nn.Module):
    """
    Dropout Regularization Layer

    Randomly zeroes some elements of the input tensor with probability p during training.
    Helps prevent overfitting.

    Parameters:
        - Dropout probability: {p}

    Shape:
        - Input: [batch_size, *] (any shape)
        - Output: [batch_size, *] (same shape as input)
    """

    def __init__(self):
        """Initialize the dropout layer."""
        super({class_name}, self).__init__()
        self.dropout = nn.Dropout(p={p})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dropout layer.

        Args:
            x: Input tensor

        Returns:
            Output tensor with dropout applied during training
        """
        # Apply dropout (only active during training)
        x = self.dropout(x)
        return x'''

    elif node_type == 'batchnorm':
        num_features = shape_info.get('num_features', 64)
        eps = config.get('eps', 1e-5)
        momentum = config.get('momentum', 0.1)
        affine = config.get('affine', True)

        return f'''class {class_name}(nn.Module):
    """
    Batch Normalization Layer

    Normalizes the input over a mini-batch for each feature channel.
    Helps stabilize and accelerate training.

    Parameters:
        - Number of features: {num_features}
        - Epsilon: {eps}
        - Momentum: {momentum}
        - Learnable parameters: {affine}

    Shape:
        - Input: [batch_size, {num_features}, H, W]
        - Output: [batch_size, {num_features}, H, W]
    """

    def __init__(self, num_features: int = {num_features}):
        """Initialize the batch normalization layer."""
        super({class_name}, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features,
            eps={eps},
            momentum={momentum},
            affine={affine}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the batch normalization layer.

        Args:
            x: Input tensor of shape [batch, {num_features}, H, W]

        Returns:
            Normalized output tensor of same shape
        """
        # Apply batch normalization
        x = self.bn(x)
        return x'''

    elif node_type == 'softmax':
        dim = config.get('dim', 1)

        return f'''class {class_name}(nn.Module):
    """
    Softmax Activation Layer

    Applies the softmax function to normalize outputs into a probability distribution.
    Commonly used in the final layer for classification tasks.

    Parameters:
        - Dimension: {dim}

    Shape:
        - Input: [batch_size, num_classes]
        - Output: [batch_size, num_classes] (sums to 1.0 along dimension {dim})
    """

    def __init__(self):
        """Initialize the softmax layer."""
        super({class_name}, self).__init__()
        self.softmax = nn.Softmax(dim={dim})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the softmax layer.

        Args:
            x: Input tensor

        Returns:
            Probability distribution over dimension {dim}
        """
        # Apply softmax activation
        x = self.softmax(x)
        return x'''

    elif node_type == 'attention':
        embed_dim = config.get('embed_dim', 512)
        num_heads = config.get('num_heads', 8)
        dropout = config.get('dropout', 0.0)

        return f'''class {class_name}(nn.Module):
    """
    Multi-Head Self-Attention Layer

    Applies multi-head self-attention mechanism to the input.
    Allows the model to jointly attend to information from different representation subspaces.

    Parameters:
        - Embedding dimension: {embed_dim}
        - Number of heads: {num_heads}
        - Dropout: {dropout}

    Shape:
        - Input: [batch_size, seq_len, {embed_dim}]
        - Output: [batch_size, seq_len, {embed_dim}]
    """

    def __init__(self):
        """Initialize the multi-head attention layer."""
        super({class_name}, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim={embed_dim},
            num_heads={num_heads},
            dropout={dropout},
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention layer.

        Args:
            x: Input tensor of shape [batch, seq_len, {embed_dim}]

        Returns:
            Output tensor after applying multi-head attention
        """
        # Apply self-attention (query, key, value are all the same)
        x, _ = self.attention(x, x, x)
        return x'''

    elif node_type == 'custom':
        name = config.get('name', 'CustomLayer')
        description = config.get('description', 'User-defined custom layer')

        # Generate proper class name from user's layer name
        safe_name = name.replace(' ', '_').replace('-', '_')
        custom_class_name = f"CustomLayer_{safe_name}"

        return f'''class {custom_class_name}(nn.Module):
    """
    Custom User-Defined Layer: {name}

    {description}

    TODO: Implement your custom layer logic below.
    This class provides the basic structure following PyTorch conventions.
    Add your initialization and forward pass logic.

    Shape:
        - Input: [batch, *] (Define your input shape)
        - Output: [batch, *] (Define your output shape)
    """

    def __init__(self):
        """Initialize the custom layer."""
        super({custom_class_name}, self).__init__()

        # TODO: Define your layer parameters here
        # Examples:
        # self.linear = nn.Linear(in_features, out_features)
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # self.activation = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.5)

        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the custom layer.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # TODO: Implement your forward pass logic here
        # Examples:
        # x = self.linear(x)
        # x = self.activation(x)
        # x = self.dropout(x)

        # Placeholder: returns input unchanged
        # Replace this with your custom logic
        return x'''

    elif node_type == 'add':
        return f'''class {class_name}(nn.Module):
    """
    Element-wise Addition Layer

    Performs element-wise addition of multiple input tensors.
    This is commonly used in residual connections and skip connections.

    Note: All input tensors must have the same shape.

    Shape:
        - Input: List of tensors with shape [batch, *]
        - Output: Tensor with same shape as inputs
    """

    def __init__(self):
        """Initialize the addition layer (no learnable parameters)."""
        super({class_name}, self).__init__()

    def forward(self, inputs: list) -> torch.Tensor:
        """
        Forward pass through the addition layer.

        Args:
            inputs: List of input tensors with identical shapes

        Returns:
            Element-wise sum of all input tensors
        """
        result = inputs[0]
        for tensor in inputs[1:]:
            result = result + tensor
        return result'''

    elif node_type == 'concat':
        dim = config.get('dim', 1)

        return f'''class {class_name}(nn.Module):
    """
    Concatenation Layer

    Concatenates multiple tensors along a specified dimension.
    Commonly used to merge feature maps from different paths in the network.

    Parameters:
        - Concatenation dimension: {dim}

    Shape:
        - Input: List of tensors with compatible shapes
        - Output: Concatenated tensor along dimension {dim}
    """

    def __init__(self):
        """Initialize the concatenation layer (no learnable parameters)."""
        super({class_name}, self).__init__()
        self.dim = {dim}

    def forward(self, inputs: list) -> torch.Tensor:
        """
        Forward pass through the concatenation layer.

        Args:
            inputs: List of input tensors to concatenate

        Returns:
            Concatenated tensor along dimension {dim}
        """
        return torch.cat(inputs, dim=self.dim)'''

    # If we reach here, the node type is not supported
    raise UnsupportedNodeTypeError(
        node_id=node.get('id', 'unknown'),
        node_type=node_type,
        framework='PyTorch'
    )


def generate_layer_instantiation(
    class_name: str,
    layer_name: str,
    shape_info: Dict[str, Any],
    node_type: str = None
) -> str:
    """
    Generate layer instantiation line for __init__ method.

    Only certain layer types need shape parameters:
    - Conv2d: needs in_channels
    - Linear: needs in_features
    - BatchNorm: needs num_features

    Other layers (Dropout, ReLU, Flatten, etc.) are instantiated with no parameters
    or only their specific configuration parameters (handled in layer class __init__).
    """
    # Only add shape parameters for layers that actually need them
    if node_type == 'conv2d' and 'in_channels' in shape_info:
        in_ch = shape_info['in_channels']
        return f"self.{layer_name} = {class_name}(in_channels={in_ch})  # Input: {in_ch} channels"
    elif node_type == 'linear' and 'in_features' in shape_info:
        in_feat = shape_info['in_features']
        return f"self.{layer_name} = {class_name}(in_features={in_feat})  # Input: {in_feat} features"
    elif node_type in ('batchnorm', 'batchnorm2d') and 'num_features' in shape_info:
        num_feat = shape_info['num_features']
        return f"self.{layer_name} = {class_name}(num_features={num_feat})  # {num_feat} features"
    else:
        # For all other layers (Dropout, ReLU, Flatten, MaxPool, etc.):
        # Instantiate with no parameters - their config is baked into the class definition
        return f"self.{layer_name} = {class_name}()"


def generate_forward_line(
    node_type: str,
    layer_name: str,
    input_var: str,
    output_var: str,
    shape_info: Dict[str, Any]
) -> str:
    """Generate forward pass line with shape comments"""
    # Build shape comment
    shape_comment = ""
    if 'out_channels' in shape_info:
        c = shape_info['out_channels']
        h = shape_info.get('out_height', '?')
        w = shape_info.get('out_width', '?')
        shape_comment = f"  # Shape: [batch, {c}, {h}, {w}]"
    elif 'out_features' in shape_info:
        f = shape_info['out_features']
        shape_comment = f"  # Shape: [batch, {f}]"

    # Handle special cases
    if node_type in ('concat', 'add'):
        return f"{output_var} = self.{layer_name}({input_var}){shape_comment}"
    else:
        return f"{output_var} = self.{layer_name}({input_var}){shape_comment}"


def get_layer_class_name(node_type: str, idx: int, config: Dict[str, Any]) -> str:
    """
    Generate descriptive class name for layer.

    Note: idx parameter is kept for backward compatibility but is no longer used.
    Class names are now based solely on node type and config to ensure consistency
    with group block naming and enable proper deduplication.
    """
    return _build_layer_class_name(node_type, config)


def get_layer_variable_name(node_type: str, idx: int, config: Dict[str, Any]) -> str:
    """Generate descriptive variable name for layer instance"""
    # Create readable names based on layer type
    if node_type == 'conv2d':
        channels = config.get('out_channels', 64)
        return f"conv_{channels}ch"
    elif node_type == 'linear':
        features = config.get('out_features', 128)
        return f"fc_{features}"
    elif node_type == 'maxpool':
        return f"maxpool_{idx}"
    elif node_type == 'flatten':
        return f"flatten"
    elif node_type == 'relu':
        return f"relu_{idx}"
    elif node_type == 'dropout':
        return f"dropout_{idx}"
    elif node_type == 'batchnorm':
        return f"batchnorm_{idx}"
    elif node_type == 'softmax':
        return f"softmax"
    elif node_type == 'attention':
        return f"attention_{idx}"
    else:
        return f"layer_{idx}"


def get_input_variable(incoming: List[str], var_map: Dict[str, str]) -> str:
    """Determine input variable name based on incoming connections"""
    if not incoming:
        return 'x'
    elif len(incoming) == 1:
        return var_map.get(incoming[0], 'x')
    else:
        # Multiple inputs (for concat, add, etc.)
        input_vars = [var_map.get(src, 'x') for src in incoming]
        return f"[{', '.join(input_vars)}]"


def generate_training_script(project_name: str) -> str:
    """Generate comprehensive training script with best practices"""
    class_name = to_class_name(project_name)

    return f'''"""
Training Script for {class_name}
Generated by VisionForge
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional

from model import create_model
from dataset import CustomDataset


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to train on (CPU or GPU)

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Print progress
        if batch_idx % 10 == 0:
            print(f'  Batch {{batch_idx}}/{{len(train_loader)}}, '
                  f'Loss: {{loss.item():.4f}}, '
                  f'Acc: {{100.*correct/total:.2f}}%')

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model.

    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def train_model(
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    device: Optional[str] = None
) -> Dict[str, list]:
    """
    Main training function.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: L2 regularization factor
        device: Device to train on ('cuda' or 'cpu', None for auto-detect)

    Returns:
        Dictionary containing training history
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f'Using device: {{device}}')

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)

    # Create model
    model = create_model()
    model = model.to(device)
    print(f'\\nModel created: {{model.__class__.__name__}}')
    print(f'Total parameters: {{sum(p.numel() for p in model.parameters()):,}}')

    # TODO: Replace with your actual dataset
    # For now, using dummy data for demonstration
    # Replace this section with:
    # train_dataset = CustomDataset('path/to/train', ...)
    # val_dataset = CustomDataset('path/to/val', ...)

    print('\\nCreating dummy datasets (replace with actual data)...')
    train_data = torch.randn(1000, 3, 224, 224)
    train_labels = torch.randint(0, 10, (1000,))
    val_data = torch.randn(200, 3, 224, 224)
    val_labels = torch.randint(0, 10, (200,))

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Increase for faster data loading
        pin_memory=(device.type == 'cuda')
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Training history
    history = {{
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }}

    best_val_loss = float('inf')
    best_epoch = 0

    print(f'\\nStarting training for {{num_epochs}} epochs...\\n')

    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {{epoch+1}}/{{num_epochs}}')
        print('-' * 60)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print epoch summary
        print(f'\\nEpoch {{epoch+1}} Summary:')
        print(f'  Train Loss: {{train_loss:.4f}}, Train Acc: {{train_acc:.2f}}%')
        print(f'  Val Loss: {{val_loss:.4f}}, Val Acc: {{val_acc:.2f}}%')
        print(f'  Learning Rate: {{optimizer.param_groups[0]["lr"]:.6f}}')
        print()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save({{
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }}, 'best_model.pth')
            print(f'✓ Best model saved (Val Loss: {{val_loss:.4f}})')

    print('\\n' + '=' * 60)
    print(f'Training completed!')
    print(f'Best model: Epoch {{best_epoch}} (Val Loss: {{best_val_loss:.4f}})')
    print('=' * 60)

    # Save final model
    torch.save(model.state_dict(), '{project_name}_final.pth')
    print(f'\\nFinal model saved to {project_name}_final.pth')

    return history


if __name__ == '__main__':
    # Train the model
    history = train_model(
        num_epochs=10,
        batch_size=32,
        learning_rate=0.001,
        weight_decay=1e-4
    )

    print('\\nTraining complete!')
'''


def generate_dataset_class(nodes: List[Dict]) -> str:
    """Generate dataset class for data loading"""

    return '''"""
Custom Dataset Class
Generated by VisionForge
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image


class CustomDataset(Dataset):
    """
    Custom dataset for loading and preprocessing data.

    This is a template - replace with your actual data loading logic.

    Args:
        data_path: Path to the dataset directory
        transform: Optional transform to be applied to samples
        split: Dataset split ('train', 'val', or 'test')
    """

    def __init__(
        self,
        data_path: str,
        transform: Optional[callable] = None,
        split: str = 'train'
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to data directory
            transform: Optional data augmentation/preprocessing
            split: Which split to load ('train', 'val', 'test')
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.split = split

        # TODO: Replace with your actual data loading
        # Example: Load file paths and labels
        # self.samples = self._load_samples()

        # For demonstration, create dummy data
        self.num_samples = 1000 if split == 'train' else 200
        print(f'Loaded {{self.num_samples}} samples for {{split}} split')

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and return a sample from the dataset.

        Args:
            idx: Index of the sample to load

        Returns:
            Tuple of (image tensor, label)
        """
        # TODO: Replace with actual data loading
        # Example:
        # image_path = self.samples[idx]['path']
        # label = self.samples[idx]['label']
        # image = Image.open(image_path).convert('RGB')
        #
        # if self.transform:
        #     image = self.transform(image)
        #
        # return image, label

        # Dummy data (NCHW format: channels, height, width)
        image = torch.randn(3, 224, 224)
        label = idx % 10  # 10 classes

        return image, label

    def _load_samples(self):
        """
        Load sample paths and labels from disk.

        Returns:
            List of sample dictionaries with 'path' and 'label' keys
        """
        # TODO: Implement actual data loading logic
        # Example for image classification:
        #
        # samples = []
        # for class_idx, class_name in enumerate(self.classes):
        #     class_dir = self.data_path / self.split / class_name
        #     for img_path in class_dir.glob('*.jpg'):
        #         samples.append({
        #             'path': img_path,
        #             'label': class_idx
        #         })
        # return samples

        pass


# Example transforms for data augmentation
def get_train_transforms():
    """Get training data transforms with augmentation."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms():
    """Get validation data transforms (no augmentation)."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# Example usage
if __name__ == '__main__':
    # Create dataset instances
    train_dataset = CustomDataset('data/', split='train')
    val_dataset = CustomDataset('data/', split='val')

    print(f'Train dataset size: {{len(train_dataset)}}')
    print(f'Val dataset size: {{len(val_dataset)}}')

    # Get a sample
    image, label = train_dataset[0]
    print(f'\\nSample image shape: {{image.shape}}')  # Should be [3, 224, 224] (NCHW)
    print(f'Sample label: {{label}}')
'''


def generate_config_file(nodes: List[Dict]) -> str:
    """Generate configuration file with hyperparameters"""

    # Find input shape from nodes
    input_shape = "[1, 3, 224, 224]"
    for node in nodes:
        if get_node_type(node) in ('input', 'dataloader'):
            shape = node.get('data', {}).get('outputShape', {}).get('dims')
            if shape:
                input_shape = str(shape)
                break

    return f'''"""
Configuration File
Generated by VisionForge
Contains all hyperparameters and settings for training
"""

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
WEIGHT_DECAY = 1e-4

# Model Configuration (NCHW format: batch, channels, height, width)
INPUT_SHAPE = {input_shape}
NUM_CLASSES = 10  # TODO: Set to your number of classes

# Optimizer Settings
OPTIMIZER = 'adam'  # Options: 'adam', 'sgd', 'adamw'
MOMENTUM = 0.9  # For SGD
BETAS = (0.9, 0.999)  # For Adam/AdamW

# Learning Rate Scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = 'reduce_on_plateau'  # Options: 'reduce_on_plateau', 'step', 'cosine'
LR_PATIENCE = 3  # For ReduceLROnPlateau
LR_FACTOR = 0.5  # For ReduceLROnPlateau
STEP_SIZE = 5  # For StepLR
GAMMA = 0.5  # For StepLR

# Early Stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5

# Data Augmentation (for training)
USE_AUGMENTATION = True
RANDOM_CROP = True
RANDOM_HORIZONTAL_FLIP = True
COLOR_JITTER = True
ROTATION_RANGE = 15

# Device Configuration
DEVICE = 'cuda'  # Options: 'cuda', 'cpu', or None for auto-detect
USE_CUDA = True  # Use GPU if available

# Mixed Precision Training (for faster training on modern GPUs)
USE_AMP = False  # Automatic Mixed Precision

# Checkpointing
SAVE_BEST_ONLY = True
CHECKPOINT_DIR = './checkpoints'
SAVE_FREQUENCY = 1  # Save every N epochs

# Logging
LOG_INTERVAL = 10  # Print every N batches
USE_TENSORBOARD = False
TENSORBOARD_DIR = './runs'

# Data Loading
NUM_WORKERS = 4  # Number of data loading workers
PIN_MEMORY = True  # Pin memory for faster GPU transfer

# Paths
DATA_DIR = './data'
TRAIN_DIR = DATA_DIR + '/train'
VAL_DIR = DATA_DIR + '/val'
TEST_DIR = DATA_DIR + '/test'

# Model specific
DROPOUT_RATE = 0.5
BATCH_NORM_MOMENTUM = 0.1
BATCH_NORM_EPS = 1e-5
'''


def get_node_type(node: Dict) -> str:
    """Extract node type from node dictionary"""
    return node.get('data', {}).get('blockType', node.get('type', 'unknown'))


def to_class_name(name: str) -> str:
    """Convert project name to valid Python class name"""
    import re
    # Remove special characters and convert to PascalCase
    name = re.sub(r'[^a-zA-Z0-9]', ' ', name)
    name = ''.join(word.capitalize() for word in name.split())
    if not name:
        return 'GeneratedModel'
    if name[0].isdigit():
        name = 'Model' + name
    return name
