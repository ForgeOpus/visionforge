"""
TensorFlow/Keras Code Generation Service
Generates tf.keras.Model code from architecture graphs with professional class-based structure
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import logging

# Import shared utilities and exceptions from PyTorch codegen (framework-agnostic)
from .pytorch_codegen import (
    GroupBlockShapeComputer,
    GroupDefinitionNotFoundError,
    ShapeMismatchError,
    CyclicDependencyError,
    UnsupportedNodeTypeError,
    ShapeInferenceError,
    MissingShapeDataError,
    safe_get_shape_data
)

# Configure logging
logger = logging.getLogger(__name__)


class TensorFlowBlockGenerator:
    """
    Generator for TensorFlow/Keras tf.keras.Model code for group blocks.
    
    Converts GroupBlockDefinition into reusable tf.keras.Model subclasses
    with proper initialization and call method logic.
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
        Generate tf.keras.Model subclass for a single block definition.
        
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

        # Generate call method
        call_method = self._generate_call_method(
            sorted_nodes, internal_edges, internal_shape_map, port_mappings
        )
        
        # Build class docstring
        docstring = self._generate_block_docstring(
            block_name, description, port_mappings, sorted_nodes
        )
        
        # Assemble the complete class
        class_code = f'''class {class_name}(keras.Model):
    """{docstring}"""

{init_method}

{call_method}'''
        
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
    
    def _generate_call_method(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        shape_map: Dict[str, Dict[str, Any]],
        port_mappings: List[Dict[str, Any]]
    ) -> str:
        """Generate call method with internal connection logic."""
        lines = []
        
        # Determine input parameters from port mappings
        input_ports = [pm for pm in port_mappings if pm['type'] == 'input']
        output_ports = [pm for pm in port_mappings if pm['type'] == 'output']
        
        # Generate method signature
        if len(input_ports) == 1:
            lines.append("    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:")
        else:
            param_names = [f"input_{i}" for i in range(len(input_ports))]
            params = ", ".join([f"{name}: tf.Tensor" for name in param_names])
            lines.append(f"    def call(self, {params}, training: Optional[bool] = None) -> tf.Tensor:")
        
        lines.append('        """')
        lines.append('        Forward pass through the block.')
        lines.append('')
        lines.append('        Args:')
        if len(input_ports) == 1:
            lines.append('            inputs: Input tensor in NHWC format')
        else:
            for i, port in enumerate(input_ports):
                label = port.get('externalPortLabel', f'input_{i}')
                lines.append(f'            input_{i}: {label}')
        lines.append('            training: Whether in training mode')
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
                var_map[internal_node_id] = 'inputs'
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
                    var_map[node_id] = 'inputs'
                continue
            
            # Get layer name
            layer_name = self._get_internal_layer_name(node_type, node_id, layer_count)
            
            # Get input variable(s)
            incoming = edge_map.get(node_id, [])
            if not incoming:
                # No incoming edges, might be an input node we missed
                input_var = 'inputs'
            elif len(incoming) == 1:
                input_var = var_map.get(incoming[0], 'inputs')
            else:
                # Multiple inputs (for concat, add, etc.)
                input_vars = [var_map.get(src, 'inputs') for src in incoming]
                input_var = f"[{', '.join(input_vars)}]"
            
            # Generate output variable name (sanitize node_id to avoid hyphens)
            output_var = f"x_{node_id[:8].replace('-', '_')}"
            var_map[node_id] = output_var
            
            # Generate forward line with training parameter for layers that need it
            if node_type in ('dropout', 'batchnorm', 'batchnorm2d'):
                lines.append(f"        {output_var} = self.{layer_name}({input_var}, training=training)")
            else:
                lines.append(f"        {output_var} = self.{layer_name}({input_var})")
        
        # Map output ports to return values
        if len(output_ports) == 1:
            output_node_id = output_ports[0]['internalNodeId']
            output_var = var_map.get(output_node_id, 'inputs')
            lines.append(f"        return {output_var}")
        else:
            output_vars = []
            for port in output_ports:
                output_node_id = port['internalNodeId']
                output_vars.append(var_map.get(output_node_id, 'inputs'))
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
        lines.append("Note: TensorFlow uses NHWC format (batch, height, width, channels)")
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
        """
        Generate layer instantiation line for TensorFlow/Keras layers.

        TensorFlow/Keras layer classes have all configuration baked into their
        class definitions, so __init__ methods take no parameters. This differs
        from PyTorch where layers need input dimensions in the constructor.
        """
        # TensorFlow layers don't need input shape parameters in constructor
        # All configuration is already baked into the layer class definition
        # Just instantiate with no arguments
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
        """Get the layer class name that will be used in the main model."""
        # These should match the class names generated by generate_layer_class
        type_name = node_type.replace('_', '').replace('2d', '2D').replace('3d', '3D').title()
        
        if node_type == 'conv2d':
            filters = config.get('filters', 64)
            kernel = config.get('kernel_size', 3)
            return f"{type_name}Layer_{filters}filters_{kernel}x{kernel}"
        elif node_type == 'linear':
            units = config.get('units', 128)
            return f"DenseLayer_{units}units"
        elif node_type in ('maxpool2d', 'maxpool'):
            pool_size = config.get('pool_size', 2)
            return f"MaxPool2DLayer_{pool_size}x{pool_size}"
        elif node_type == 'custom':
            name = config.get('name', 'CustomLayer')
            safe_name = name.replace(' ', '_').replace('-', '_')
            return f"CustomLayer_{safe_name}"
        else:
            # For other types, we'll need to generate a generic name
            # This will be handled by the main code generation
            return f"{type_name}Layer"
    
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


def generate_tensorflow_code(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    project_name: str = "GeneratedModel",
    group_definitions: Optional[List[Dict[str, Any]]] = None
) -> Tuple[Dict[str, str], List[Exception]]:
    """
    Generate complete TensorFlow/Keras code including model, training, and data loading.
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

    # Convert group_definitions list to dict for shape inference
    group_defs_dict = None
    if group_definitions:
        group_defs_dict = {defn['id']: defn for defn in group_definitions}

    # Infer shapes through the graph with group block support
    shape_map, shape_errors = infer_shapes(sorted_nodes, edges, group_defs_dict)

    # Validate computed shapes for critical issues
    validation_errors = validate_shape_map(sorted_nodes, shape_map)
    if validation_errors:
        logger.warning(f"Shape validation found {len(validation_errors)} potential issues")
        shape_errors.extend(validation_errors)

    # Initialize block generator if we have group definitions
    block_generator = None
    if group_definitions:
        # Create shape computer for block generator
        shape_computer = GroupBlockShapeComputer(group_defs_dict) if group_defs_dict else None
        block_generator = TensorFlowBlockGenerator(group_definitions, shape_computer)

    # Generate different components
    model_code = generate_model_file(sorted_nodes, edges, project_name, shape_map, block_generator)
    train_code = generate_training_script(project_name)
    dataset_code = generate_dataset_class(nodes)
    config_code = generate_config_file(nodes)

    # Return generated code with any shape inference errors
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
# Input shape (NHWC): {shape_info.get('out_channels', '?')} channels or {shape_info.get('out_units', '?')} units'''

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
    TensorFlow uses NHWC format (batch, height, width, channels).

    Args:
        node: Node dictionary

    Returns:
        Dictionary with shape information (in_channels, out_channels, in_units, out_units, etc.)
    """
    shape_info = {}
    node_type = get_node_type(node)
    config = node.get('data', {}).get('config', {})

    # Try to get shape from node metadata
    input_shape = node.get('data', {}).get('inputShape', {})
    output_shape = node.get('data', {}).get('outputShape', {})

    # Extract from inputShape/outputShape if available (NHWC format)
    if input_shape and isinstance(input_shape, dict):
        dims = input_shape.get('dims', [])
        if len(dims) >= 4:  # NHWC format
            shape_info['in_height'] = dims[1]
            shape_info['in_width'] = dims[2]
            shape_info['in_channels'] = dims[3]
        elif len(dims) >= 2:
            shape_info['in_units'] = dims[1]

    if output_shape and isinstance(output_shape, dict):
        dims = output_shape.get('dims', [])
        if len(dims) >= 4:  # NHWC format
            shape_info['out_height'] = dims[1]
            shape_info['out_width'] = dims[2]
            shape_info['out_channels'] = dims[3]
        elif len(dims) >= 2:
            shape_info['out_units'] = dims[1]

    # Infer from config if not in metadata
    if node_type == 'conv2d':
        if 'in_channels' not in shape_info:
            shape_info['in_channels'] = 3  # Default
        if 'out_channels' not in shape_info:
            shape_info['out_channels'] = config.get('filters', 64)
        # Try to estimate output dimensions if not provided
        if 'out_height' not in shape_info:
            shape_info['out_height'] = '?'
        if 'out_width' not in shape_info:
            shape_info['out_width'] = '?'

    elif node_type == 'linear':
        if 'in_units' not in shape_info:
            shape_info['in_units'] = 512  # Default
        if 'out_units' not in shape_info:
            shape_info['out_units'] = config.get('units', 128)

    elif node_type in ('batchnorm', 'batchnorm2d'):
        # BatchNorm preserves shape
        if 'out_channels' not in shape_info:
            shape_info['out_channels'] = shape_info.get('in_channels', 64)

    elif node_type == 'flatten':
        if 'out_units' not in shape_info:
            # Estimate based on typical conv output (NHWC)
            height = shape_info.get('in_height', 7)
            width = shape_info.get('in_width', 7)
            channels = shape_info.get('in_channels', 512)
            if isinstance(height, int) and isinstance(width, int) and isinstance(channels, int):
                shape_info['out_units'] = height * width * channels
            else:
                shape_info['out_units'] = '?'

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
    Extract output shape from node's frontend-provided metadata (TensorFlow/NHWC version).

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

    # TensorFlow uses NHWC format: [batch, height, width, channels]
    # Note: This is different from PyTorch's NCHW format!
    if len(dims) == 4:
        shape_info['out_height'] = dims[1]
        shape_info['out_width'] = dims[2]
        shape_info['out_channels'] = dims[3]
    elif len(dims) == 2:  # [batch, features] - for Dense/Flatten output
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
    TensorFlow uses NHWC format (batch, height, width, channels).
    Enhanced to handle group blocks properly.

    Args:
        nodes: List of node dictionaries
        edges: List of edge dictionaries
        group_definitions: Optional map of group definition IDs to definitions

    Returns:
        Tuple of (dictionary mapping node_id to shape info, list of errors)
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

    # Process nodes in order
    for node in nodes:
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
        # This is critical for layers like Conv2D, Dense, BatchNorm

        if node_type == 'input':
            # Input nodes have no upstream - parse from config if metadata doesn't exist
            if not metadata_shape:
                shape_str = config.get('shape', '[1, 224, 224, 3]')
                try:
                    import json
                    shape = json.loads(shape_str)
                    if len(shape) >= 4:
                        shape_info['out_height'] = shape[1]  # NHWC format
                        shape_info['out_width'] = shape[2]
                        shape_info['out_channels'] = shape[3]
                    elif len(shape) >= 2:
                        shape_info['out_units'] = shape[1]
                except (json.JSONDecodeError, ValueError, KeyError, IndexError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse input shape for node {node_id}: {e}. "
                        f"Using default shape [1, 224, 224, 3] (NHWC)"
                    )
                    errors.append(ShapeInferenceError(
                        node_id=node_id,
                        node_type=node_type,
                        reason=f"Failed to parse shape configuration: {str(e)}",
                        suggestion="Check that the input shape is a valid JSON array like [1, 224, 224, 3]"
                    ))
                    shape_info['out_height'] = 224
                    shape_info['out_width'] = 224
                    shape_info['out_channels'] = 3

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
                shape_info['out_channels'] = config.get('filters', 64)

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
                        strides = config.get('strides', 1)
                        padding = config.get('padding', 'valid')

                        if padding == 'same':
                            # Same padding preserves dimensions (with stride)
                            shape_info['out_height'] = (prev_shape['out_height'] + strides - 1) // strides
                            shape_info['out_width'] = (prev_shape['out_width'] + strides - 1) // strides
                        else:  # valid padding
                            shape_info['out_height'] = (prev_shape['out_height'] - kernel_size) // strides + 1
                            shape_info['out_width'] = (prev_shape['out_width'] - kernel_size) // strides + 1
                    except (MissingShapeDataError, ShapeInferenceError) as e:
                        logger.warning(f"Could not compute spatial dimensions for conv2d {node_id}: {e}")
                        errors.append(e)

        elif node_type in ('maxpool2d', 'maxpool'):
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
                        pool_size = config.get('pool_size', 2)
                        strides = config.get('strides', 2)
                        padding = config.get('padding', 'valid')

                        if padding == 'same':
                            shape_info['out_height'] = (prev_shape['out_height'] + strides - 1) // strides
                            shape_info['out_width'] = (prev_shape['out_width'] + strides - 1) // strides
                        else:  # valid padding
                            shape_info['out_height'] = (prev_shape['out_height'] - pool_size) // strides + 1
                            shape_info['out_width'] = (prev_shape['out_width'] - pool_size) // strides + 1
                    except (MissingShapeDataError, ShapeInferenceError) as e:
                        logger.warning(f"Could not compute spatial dimensions for maxpool {node_id}: {e}")
                        errors.append(e)

        elif node_type == 'flatten':
            # Flatten converts spatial dimensions to units
            # Use metadata if available, otherwise calculate from upstream
            if 'out_units' not in shape_info:
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
                        shape_info['out_units'] = channels * height * width
                    except (MissingShapeDataError, ShapeInferenceError) as e:
                        logger.warning(f"Shape inference warning for flatten {node_id}: {e}")
                        errors.append(e)
                        shape_info['out_units'] = 3136  # 64 * 7 * 7
                else:
                    shape_info['out_units'] = 3136  # Default

        elif node_type == 'linear':
            # Get input units from upstream layer (ALWAYS required)
            if incoming and incoming[0] in shape_map:
                try:
                    upstream_shape = safe_get_shape_data(
                        shape_map=shape_map,
                        node_id=node_id,
                        upstream_node_id=incoming[0],
                        required_keys=['out_units'],
                        default_values={'out_units': 512}
                    )
                    shape_info['in_units'] = upstream_shape['out_units']
                except (MissingShapeDataError, ShapeInferenceError) as e:
                    logger.warning(f"Shape inference warning for linear {node_id}: {e}")
                    errors.append(e)
                    shape_info['in_units'] = 512
            else:
                shape_info['in_units'] = 512

            # Output units: use metadata if available, otherwise config
            if 'out_units' not in shape_info:
                shape_info['out_units'] = config.get('units', 128)

        elif node_type in ('batchnorm', 'batchnorm2d'):
            # BatchNorm preserves all dimensions from upstream
            # Only copy upstream if metadata doesn't provide them
            if not metadata_shape and incoming and incoming[0] in shape_map:
                try:
                    prev_shape = safe_get_shape_data(
                        shape_map=shape_map,
                        node_id=node_id,
                        upstream_node_id=incoming[0],
                        required_keys=[],  # Accept whatever keys exist
                        default_values={}
                    )
                    shape_info.update(prev_shape)
                except (MissingShapeDataError, ShapeInferenceError) as e:
                    logger.warning(f"Shape inference warning for batchnorm {node_id}: {e}")
                    errors.append(e)

        elif node_type == 'group':
            # Group blocks: Use metadata if available, otherwise compute from internal structure
            if not metadata_shape:
                # No metadata - compute output shape using shape computer
                if shape_computer:
                    group_def_id = node.get('data', {}).get('groupDefinitionId')

                    if group_def_id and incoming and incoming[0] in shape_map:
                        # Get input shape from upstream node
                        input_shape = shape_map[incoming[0]]

                        # Compute output shape using internal structure
                        logger.debug(f"Computing shape for group block {node_id} (def: {group_def_id})")
                        output_shape, shape_errors = shape_computer.compute_output_shape(
                            group_def_id,
                            input_shape
                        )

                        # Collect any errors from shape computation
                        errors.extend(shape_errors)

                        if output_shape:
                            shape_info = output_shape
                            logger.debug(f"Group block {node_id} output shape: {output_shape}")
                        else:
                            # Fallback: copy input shape
                            shape_info = input_shape.copy()
                            logger.warning(f"Failed to compute shape for group block {node_id}, using input shape")
                    elif incoming and incoming[0] in shape_map:
                        # No definition found, copy input shape
                        shape_info = shape_map[incoming[0]].copy()
                        logger.warning(f"Group block {node_id} has no definition ID, using input shape")
                    else:
                        # No input, use default
                        shape_info = {'out_channels': 3, 'out_height': 224, 'out_width': 224}
                        logger.warning(f"Group block {node_id} has no incoming edges, using default shape")
                else:
                    # No shape computer available, fall back to old behavior
                    if incoming and incoming[0] in shape_map:
                        prev_shape = shape_map[incoming[0]]
                        # Copy input shape as default
                        shape_info.update(prev_shape)
                    else:
                        # Default starting shape
                        shape_info['out_channels'] = 3
                        shape_info['out_height'] = 224
                        shape_info['out_width'] = 224

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
    Validate computed shape map for common critical issues (TensorFlow version).

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
        if node_type == 'linear' or node_type == 'dense':
            # Linear/Dense MUST have in_features or in_units
            if 'in_features' not in shape_info and 'in_units' not in shape_info:
                errors.append(ShapeInferenceError(
                    node_id=node_id,
                    node_type=node_type,
                    reason="Missing required in_features/in_units for Linear/Dense layer",
                    suggestion="Check upstream Flatten or Linear layer output shape"
                ))
            # in_features/in_units must be positive
            in_val = shape_info.get('in_features') or shape_info.get('in_units', 0)
            if in_val <= 0:
                errors.append(ShapeInferenceError(
                    node_id=node_id,
                    node_type=node_type,
                    reason=f"Invalid in_features/in_units={in_val} (must be > 0)",
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
                    suggestion="Check upstream layer has spatial dimensions (NHWC format)"
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
    block_generator: Optional[TensorFlowBlockGenerator] = None
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


def generate_model_file(
    nodes: List[Dict],
    edges: List[Dict],
    project_name: str,
    shape_map: Dict[str, Dict[str, Any]],
    block_generator: Optional[TensorFlowBlockGenerator] = None
) -> str:
    """Generate complete model.py file with layer classes and main model class"""

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
                        # TensorFlow uses same parameter names as PyTorch for consistency
                        # Pass in_channels if the upstream outputs channels (convolutional layers)
                        if 'out_channels' in upstream_shape:
                            in_channels = upstream_shape['out_channels']
                            params.append(f"in_channels={in_channels}")
                            logger.debug(f"TF Block {node_id}: passing in_channels={in_channels} from upstream node {incoming[0]}")
                        
                        # Pass in_features if the upstream outputs features (linear layers)
                        # TensorFlow uses 'out_units' instead of 'out_features'
                        elif 'out_units' in upstream_shape:
                            in_units = upstream_shape['out_units']
                            params.append(f"in_features={in_units}")
                            logger.debug(f"TF Block {node_id}: passing in_features={in_units} from upstream node {incoming[0]}")
                        elif 'out_features' in upstream_shape:
                            in_features = upstream_shape['out_features']
                            params.append(f"in_features={in_features}")
                            logger.debug(f"TF Block {node_id}: passing in_features={in_features} from upstream node {incoming[0]}")
                        
                        # Pass num_features if the upstream outputs num_features (batch norm)
                        elif 'num_features' in upstream_shape:
                            num_features = upstream_shape['num_features']
                            params.append(f"num_features={num_features}")
                            logger.debug(f"TF Block {node_id}: passing num_features={num_features} from upstream node {incoming[0]}")
                        else:
                            # Upstream shape exists but doesn't have expected keys
                            logger.warning(f"TF Block {node_id}: upstream shape {upstream_shape} doesn't contain expected keys")
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
                                logger.debug(f"TF Block {node_id}: no upstream, using input shape in_channels={in_channels}")
                            elif 'out_units' in input_shape:
                                in_units = input_shape['out_units']
                                params.append(f"in_features={in_units}")
                                logger.debug(f"TF Block {node_id}: no upstream, using input shape in_features={in_units}")
                            elif 'out_features' in input_shape:
                                in_features = input_shape['out_features']
                                params.append(f"in_features={in_features}")
                                logger.debug(f"TF Block {node_id}: no upstream, using input shape in_features={in_features}")
                            else:
                                logger.warning(f"TF Block {node_id}: input shape {input_shape} doesn't contain expected keys")
                        else:
                            # No upstream and no input node, use defaults
                            logger.warning(f"TF Block {node_id}: no upstream connection and no input node found")

                    # Generate instantiation with computed parameters
                    # Each instance gets independent shape computation based on its position in the graph
                    if params:
                        layer_instantiations.append(f"self.{layer_name} = {block_class_name}({', '.join(params)})  # Instance at position {idx}")
                    else:
                        layer_instantiations.append(f"self.{layer_name} = {block_class_name}()  # Instance at position {idx}")

                    # Generate forward pass line
                    input_var = get_input_variable(incoming, var_map)
                    output_var = 'x'
                    forward_pass_lines.append(f"{output_var} = self.{layer_name}({input_var}, training=training)")
                    var_map[node_id] = output_var
                else:
                    # Block class not found, skip
                    logger.warning(f"TF Block class not found for group definition {group_def_id}")
                    var_map[node_id] = 'x'
            else:
                # No block generator or definition ID, skip
                logger.warning(f"TF No block generator or definition ID for node {node_id}")
                var_map[node_id] = 'x'
            continue

        # For regular nodes, we already generated the layer class above (no need to generate again)

        # Generate layer instantiation for __init__
        layer_name = get_layer_variable_name(node_type, idx, config)
        layer_class_name = get_layer_class_name(node_type, idx, config)
        layer_init = generate_layer_instantiation(layer_class_name, layer_name, shape_info)
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
Generated TensorFlow/Keras Model
Architecture: {class_name}
Generated by VisionForge

Note: TensorFlow uses NHWC format (batch, height, width, channels)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
class {class_name}(keras.Model):
    """
    Main model class combining all layers.

    This model was automatically generated from a visual architecture.
    Each layer is implemented as a separate class for clarity and reusability.

    Note: TensorFlow uses NHWC format (batch, height, width, channels)
    """

    def __init__(self):
        """Initialize all layers in the model."""
        super({class_name}, self).__init__()

'''

    # Add layer instantiations
    for init in layer_instantiations:
        code += f'        {init}\n'

    code += '''
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the model.

        Args:
            inputs: Input tensor in NHWC format
            training: Whether the model is in training mode

        Returns:
            Output tensor after passing through all layers
        """
        x = inputs
'''

    # Add forward pass lines
    for line in forward_pass_lines:
        code += f'        {line}\n'

    code += '''
        return x


def create_model() -> keras.Model:
    """
    Create and return an instance of the model.

    Returns:
        Initialized model ready for training or inference
    """
    model = {class_name}()
    return model


if __name__ == '__main__':
    # Create model
    model = create_model()
    print(f"Model: {class_name}")

    # Build the model with a sample input to initialize weights
    model.build(input_shape=(None, 224, 224, 3))  # NHWC format

    # Print model summary
    model.summary()

    # Test forward pass with dummy input
    dummy_input = tf.random.normal([1, 224, 224, 3])  # NHWC format
    output = model(dummy_input)
    print(f"\\nInput shape: {{dummy_input.shape}}")  # NHWC: [batch, height, width, channels]
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
        filters = config.get('filters', 64)
        kernel_size = config.get('kernel_size', 3)
        strides = config.get('strides', 1)
        padding = config.get('padding', 'valid')
        activation = config.get('activation', 'None')
        activation_str = f"'{activation}'" if activation != 'None' else 'None'

        # Calculate output shape
        out_h = shape_info.get('out_height', '?')
        out_w = shape_info.get('out_width', '?')
        out_c = filters

        return f'''class {class_name}(layers.Layer):
    """
    2D Convolutional Layer

    Applies a 2D convolution over an input signal.

    Parameters:
        - Filters (output channels): {filters}
        - Kernel size: {kernel_size}x{kernel_size}
        - Strides: {strides}
        - Padding: '{padding}'
        - Activation: {activation if activation != 'None' else 'None'}

    Shape:
        - Input: [batch, H, W, C] (NHWC format)
        - Output: [batch, {out_h}, {out_w}, {out_c}]
    """

    def __init__(self):
        """Initialize the convolutional layer."""
        super({class_name}, self).__init__()
        self.conv = layers.Conv2D(
            filters={filters},
            kernel_size={kernel_size},
            strides={strides},
            padding='{padding}',
            activation={activation_str}
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the convolutional layer.

        Args:
            inputs: Input tensor of shape [batch, H, W, C]
            training: Whether in training mode

        Returns:
            Output tensor of shape [batch, {out_h}, {out_w}, {out_c}]
        """
        # Apply convolution
        x = self.conv(inputs)
        return x'''

    elif node_type == 'linear':
        units = config.get('units', 128)
        activation = config.get('activation', 'None')
        use_bias = config.get('use_bias', True)
        activation_str = f"'{activation}'" if activation != 'None' else 'None'

        return f'''class {class_name}(layers.Layer):
    """
    Fully Connected (Dense) Layer

    Applies a linear transformation to the incoming data: y = xW + b

    Parameters:
        - Units (output size): {units}
        - Activation: {activation if activation != 'None' else 'None'}
        - Use bias: {use_bias}

    Shape:
        - Input: [batch, input_dim]
        - Output: [batch, {units}]
    """

    def __init__(self):
        """Initialize the dense layer."""
        super({class_name}, self).__init__()
        self.dense = layers.Dense(
            units={units},
            activation={activation_str},
            use_bias={use_bias}
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the dense layer.

        Args:
            inputs: Input tensor of shape [batch, input_dim]
            training: Whether in training mode

        Returns:
            Output tensor of shape [batch, {units}]
        """
        # Apply linear transformation
        x = self.dense(inputs)
        return x'''

    elif node_type in ('maxpool2d', 'maxpool'):
        pool_size = config.get('pool_size', 2)
        strides = config.get('strides', 2)
        padding = config.get('padding', 'valid')

        return f'''class {class_name}(layers.Layer):
    """
    2D Max Pooling Layer

    Applies a 2D max pooling over an input signal.
    Reduces spatial dimensions while preserving channel count.

    Parameters:
        - Pool size: {pool_size}x{pool_size}
        - Strides: {strides}
        - Padding: '{padding}'

    Shape:
        - Input: [batch, H, W, C] (NHWC format)
        - Output: [batch, H/{strides}, W/{strides}, C]
    """

    def __init__(self):
        """Initialize the max pooling layer."""
        super({class_name}, self).__init__()
        self.pool = layers.MaxPooling2D(
            pool_size={pool_size},
            strides={strides},
            padding='{padding}'
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the pooling layer.

        Args:
            inputs: Input tensor of shape [batch, H, W, C]
            training: Whether in training mode

        Returns:
            Output tensor with reduced spatial dimensions
        """
        # Apply max pooling
        x = self.pool(inputs)
        return x'''

    elif node_type == 'flatten':
        out_units = shape_info.get('out_units', '?')

        return f'''class {class_name}(layers.Layer):
    """
    Flatten Layer

    Flattens the input tensor to a 1D vector per batch sample.
    Commonly used to transition from convolutional layers to fully connected layers.

    Shape:
        - Input: [batch, H, W, C] (NHWC format)
        - Output: [batch, H*W*C] = [batch, {out_units}]
    """

    def __init__(self):
        """Initialize the flatten layer."""
        super({class_name}, self).__init__()
        self.flatten = layers.Flatten()

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the flatten layer.

        Args:
            inputs: Input tensor of shape [batch, H, W, C]
            training: Whether in training mode

        Returns:
            Output tensor of shape [batch, H*W*C]
        """
        # Flatten spatial and channel dimensions
        x = self.flatten(inputs)
        return x'''

    elif node_type == 'dropout':
        rate = config.get('rate', 0.5)

        return f'''class {class_name}(layers.Layer):
    """
    Dropout Regularization Layer

    Randomly sets input units to 0 with frequency rate during training.
    Helps prevent overfitting.

    Parameters:
        - Dropout rate: {rate}

    Shape:
        - Input: [batch, *] (any shape)
        - Output: [batch, *] (same shape as input)
    """

    def __init__(self):
        """Initialize the dropout layer."""
        super({class_name}, self).__init__()
        self.dropout = layers.Dropout(rate={rate})

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the dropout layer.

        Args:
            inputs: Input tensor
            training: Whether in training mode (dropout only active during training)

        Returns:
            Output tensor with dropout applied during training
        """
        # Apply dropout (only active during training)
        x = self.dropout(inputs, training=training)
        return x'''

    elif node_type in ('batchnorm', 'batchnorm2d'):
        momentum = config.get('momentum', 0.99)
        epsilon = config.get('epsilon', 0.001)

        return f'''class {class_name}(layers.Layer):
    """
    Batch Normalization Layer

    Normalizes the activations of the previous layer at each batch.
    Helps stabilize and accelerate training.

    Parameters:
        - Momentum: {momentum}
        - Epsilon: {epsilon}

    Shape:
        - Input: [batch, H, W, C] or [batch, features] (NHWC format)
        - Output: Same shape as input
    """

    def __init__(self):
        """Initialize the batch normalization layer."""
        super({class_name}, self).__init__()
        self.bn = layers.BatchNormalization(
            momentum={momentum},
            epsilon={epsilon}
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the batch normalization layer.

        Args:
            inputs: Input tensor
            training: Whether in training mode

        Returns:
            Normalized output tensor of same shape
        """
        # Apply batch normalization
        x = self.bn(inputs, training=training)
        return x'''

    elif node_type == 'concat':
        axis = config.get('axis', -1)

        return f'''class {class_name}(layers.Layer):
    """
    Concatenation Layer

    Concatenates multiple input tensors along a specified axis.
    Used for skip connections and multi-path architectures.

    Parameters:
        - Axis: {axis}

    Shape:
        - Input: List of tensors with compatible shapes
        - Output: Single concatenated tensor
    """

    def __init__(self):
        """Initialize the concatenation layer."""
        super({class_name}, self).__init__()
        self.concat = layers.Concatenate(axis={axis})

    def call(self, inputs: list, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the concatenation layer.

        Args:
            inputs: List of input tensors to concatenate
            training: Whether in training mode

        Returns:
            Concatenated output tensor
        """
        # Concatenate along specified axis
        x = self.concat(inputs)
        return x'''

    elif node_type == 'add':
        return f'''class {class_name}(layers.Layer):
    """
    Addition Layer

    Performs element-wise addition of multiple input tensors.
    Used for residual connections and multi-path architectures.

    Shape:
        - Input: List of tensors with identical shapes
        - Output: Single tensor with same shape as inputs
    """

    def __init__(self):
        """Initialize the addition layer."""
        super({class_name}, self).__init__()
        self.add = layers.Add()

    def call(self, inputs: list, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the addition layer.

        Args:
            inputs: List of input tensors to add
            training: Whether in training mode

        Returns:
            Sum of input tensors
        """
        # Element-wise addition
        x = self.add(inputs)
        return x'''

    elif node_type == 'custom':
        name = config.get('name', 'CustomLayer')
        description = config.get('description', 'User-defined custom layer')

        # Generate proper class name from user's layer name
        safe_name = name.replace(' ', '_').replace('-', '_')
        custom_class_name = f"CustomLayer_{safe_name}"

        return f'''class {custom_class_name}(layers.Layer):
    """
    Custom User-Defined Layer: {name}

    {description}

    TODO: Implement your custom layer logic below.
    This class provides the basic structure following TensorFlow/Keras conventions.
    Add your initialization and call method logic.

    Note: TensorFlow uses NHWC format (batch, height, width, channels)

    Shape:
        - Input: [batch, *] (Define your input shape in NHWC format)
        - Output: [batch, *] (Define your output shape)
    """

    def __init__(self):
        """Initialize the custom layer."""
        super({custom_class_name}, self).__init__()

        # TODO: Define your layer components here
        # Examples:
        # self.dense = layers.Dense(units=128)
        # self.conv = layers.Conv2D(filters=64, kernel_size=3)
        # self.activation = layers.ReLU()
        # self.dropout = layers.Dropout(rate=0.5)

        pass

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the custom layer.

        Args:
            inputs: Input tensor in NHWC format
            training: Whether in training mode

        Returns:
            Output tensor
        """
        # TODO: Implement your call method logic here
        # Examples:
        # x = self.dense(inputs)
        # x = self.activation(x)
        # x = self.dropout(x, training=training)

        # Placeholder: returns input unchanged
        # Replace this with your custom logic
        x = inputs
        return x'''

    # If we reach here, the node type is not supported
    raise UnsupportedNodeTypeError(
        node_id=node.get('id', 'unknown'),
        node_type=node_type,
        framework='TensorFlow'
    )


def generate_layer_instantiation(
    class_name: str,
    layer_name: str,
    shape_info: Dict[str, Any]
) -> str:
    """Generate layer instantiation line for __init__ method"""
    # TensorFlow layers don't need input size in constructor
    if 'in_channels' in shape_info:
        in_ch = shape_info['in_channels']
        return f"self.{layer_name} = {class_name}()  # Input: {in_ch} channels (NHWC)"
    elif 'in_units' in shape_info:
        in_units = shape_info['in_units']
        return f"self.{layer_name} = {class_name}()  # Input: {in_units} units"
    else:
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
        h = shape_info.get('out_height', '?')
        w = shape_info.get('out_width', '?')
        c = shape_info['out_channels']
        shape_comment = f"  # Shape: [batch, {h}, {w}, {c}] (NHWC)"
    elif 'out_units' in shape_info:
        u = shape_info['out_units']
        shape_comment = f"  # Shape: [batch, {u}]"

    # Handle layers that need training parameter
    if node_type in ('dropout', 'batchnorm', 'batchnorm2d'):
        return f"{output_var} = self.{layer_name}({input_var}, training=training){shape_comment}"
    # Handle merge layers
    elif node_type in ('concat', 'add'):
        return f"{output_var} = self.{layer_name}({input_var}){shape_comment}"
    else:
        return f"{output_var} = self.{layer_name}({input_var}){shape_comment}"


def get_layer_class_name(node_type: str, idx: int, config: Dict[str, Any]) -> str:
    """Generate descriptive class name for layer"""
    type_name = node_type.replace('_', '').replace('2d', '2D').replace('3d', '3D').title()

    # Add descriptive suffix based on config
    if node_type == 'conv2d':
        filters = config.get('filters', 64)
        kernel = config.get('kernel_size', 3)
        return f"{type_name}Layer_{filters}filters_{kernel}x{kernel}"
    elif node_type == 'linear':
        units = config.get('units', 128)
        return f"DenseLayer_{units}units"
    elif node_type in ('maxpool2d', 'maxpool'):
        pool_size = config.get('pool_size', 2)
        return f"MaxPool2DLayer_{pool_size}x{pool_size}"
    else:
        return f"{type_name}Layer_{idx}"


def get_layer_variable_name(node_type: str, idx: int, config: Dict[str, Any]) -> str:
    """Generate descriptive variable name for layer instance"""
    # Create readable names based on layer type
    if node_type == 'conv2d':
        filters = config.get('filters', 64)
        return f"conv_{filters}filters"
    elif node_type == 'linear':
        units = config.get('units', 128)
        return f"dense_{units}"
    elif node_type in ('maxpool2d', 'maxpool'):
        return f"maxpool_{idx}"
    elif node_type == 'flatten':
        return f"flatten"
    elif node_type == 'dropout':
        return f"dropout_{idx}"
    elif node_type in ('batchnorm', 'batchnorm2d'):
        return f"batchnorm_{idx}"
    elif node_type == 'concat':
        return f"concat_{idx}"
    elif node_type == 'add':
        return f"add_{idx}"
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

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional

from model import create_model
from dataset import CustomDataset


def train_model(
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    use_gpu: bool = True
) -> keras.callbacks.History:
    """
    Main training function.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: L2 regularization factor
        use_gpu: Whether to use GPU if available

    Returns:
        Training history object
    """
    # Configure GPU
    if use_gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f'Found {{len(gpus)}} GPU(s)')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            print('No GPU found, using CPU')
    else:
        tf.config.set_visible_devices([], 'GPU')
        print('Using CPU')

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Create model
    model = create_model()
    print(f'\\nModel created: {{model.__class__.__name__}}')

    # Build model with sample input (NHWC format)
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()

    # TODO: Replace with your actual dataset
    # Example:
    # train_dataset = CustomDataset('path/to/train', batch_size=batch_size)
    # val_dataset = CustomDataset('path/to/val', batch_size=batch_size)

    print('\\nCreating dummy datasets (replace with actual data)...')
    # Dummy data (NHWC format: batch, height, width, channels)
    train_data = np.random.randn(1000, 224, 224, 3).astype(np.float32)
    train_labels = np.random.randint(0, 10, (1000,))
    val_data = np.random.randn(200, 224, 224, 3).astype(np.float32)
    val_labels = np.random.randint(0, 10, (200,))

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]

    print(f'\\nStarting training for {{num_epochs}} epochs...\\n')

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save('{project_name}_final.keras')
    print(f"\\nFinal model saved to {project_name}_final.keras")

    # Print training summary
    print('\\n' + '=' * 60)
    print('Training completed!')
    print(f'Best validation loss: {{min(history.history["val_loss"]):.4f}}')
    print(f'Best validation accuracy: {{max(history.history["val_accuracy"]):.4f}}')
    print('=' * 60)

    return history


if __name__ == '__main__':
    # Train the model
    history = train_model(
        num_epochs=10,
        batch_size=32,
        learning_rate=0.001,
        weight_decay=1e-4,
        use_gpu=True
    )

    print('\\nTraining complete!')
'''


def generate_dataset_class(nodes: List[Dict]) -> str:
    """Generate dataset class for data loading"""

    return '''"""
Custom Dataset Class for TensorFlow
Generated by VisionForge
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class CustomDataset(keras.utils.PyDataset):
    """
    Custom dataset using tf.keras.utils.PyDataset for efficient data loading.

    This is a template - replace with your actual data loading logic.

    Args:
        data_path: Path to the dataset directory
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        split: Dataset split ('train', 'val', or 'test')
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        shuffle: bool = True,
        split: str = 'train',
        **kwargs
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to data directory
            batch_size: Batch size for loading
            shuffle: Whether to shuffle data
            split: Which split to load ('train', 'val', 'test')
        """
        super().__init__(**kwargs)
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split

        # TODO: Replace with your actual data loading
        # Example: Load file paths and labels
        # self.samples = self._load_samples()

        # For demonstration, create dummy data
        self.num_samples = 1000 if split == 'train' else 200
        print(f'Loaded {{self.num_samples}} samples for {{split}} split')

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return self.num_samples // self.batch_size

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one batch of data.

        Args:
            idx: Batch index

        Returns:
            Tuple of (inputs, targets) in NHWC format
        """
        # TODO: Replace with actual data loading
        # Example:
        # batch_samples = self.samples[idx*self.batch_size:(idx+1)*self.batch_size]
        # batch_x = []
        # batch_y = []
        # for sample in batch_samples:
        #     image = load_image(sample['path'])  # Load and preprocess
        #     batch_x.append(image)
        #     batch_y.append(sample['label'])
        # return np.array(batch_x), np.array(batch_y)

        # Generate dummy batch (NHWC format: batch, height, width, channels)
        batch_x = np.random.randn(self.batch_size, 224, 224, 3).astype(np.float32)
        batch_y = np.random.randint(0, 10, self.batch_size)

        return batch_x, batch_y

    def on_epoch_end(self):
        """Called at the end of each epoch."""
        if self.shuffle:
            # TODO: Implement shuffling logic
            pass

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
        # split_dir = self.data_path / self.split
        # for class_idx, class_name in enumerate(sorted(split_dir.iterdir())):
        #     if class_name.is_dir():
        #         for img_path in class_name.glob('*.jpg'):
        #             samples.append({{
        #                 'path': str(img_path),
        #                 'label': class_idx
        #             }})
        # return samples

        pass


# Example data preprocessing functions
def preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load and preprocess an image.

    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (height, width)

    Returns:
        Preprocessed image array in NHWC format
    """
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # Resize
    image = tf.image.resize(image, target_size)

    # Normalize to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    # Optional: Normalize with ImageNet mean and std
    # mean = tf.constant([0.485, 0.456, 0.406])
    # std = tf.constant([0.229, 0.224, 0.225])
    # image = (image - mean) / std

    return image.numpy()


def augment_image(image: np.ndarray) -> np.ndarray:
    """
    Apply data augmentation to an image.

    Args:
        image: Input image in NHWC format

    Returns:
        Augmented image
    """
    image = tf.constant(image)

    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)

    # Random brightness and contrast
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # Random rotation (small angles)
    # Note: Requires tf-addons for rotation
    # image = tfa.image.rotate(image, angles=tf.random.uniform([], -0.2, 0.2))

    # Clip values to [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image.numpy()


# Example usage
if __name__ == '__main__':
    # Create dataset instances
    train_dataset = CustomDataset('data/', batch_size=32, split='train')
    val_dataset = CustomDataset('data/', batch_size=32, split='val')

    print(f'Train dataset: {{len(train_dataset)}} batches')
    print(f'Val dataset: {{len(val_dataset)}} batches')

    # Get a sample batch
    batch_x, batch_y = train_dataset[0]
    print(f'\\nBatch X shape: {{batch_x.shape}}')  # Should be (32, 224, 224, 3) in NHWC format
    print(f'Batch Y shape: {{batch_y.shape}}')
'''


def generate_config_file(nodes: List[Dict]) -> str:
    """Generate configuration file with hyperparameters"""

    # Find input shape from nodes (NHWC format)
    input_shape = "[1, 224, 224, 3]"
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

Note: TensorFlow uses NHWC format (batch, height, width, channels)
"""

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
WEIGHT_DECAY = 1e-4

# Model Configuration (NHWC format: batch, height, width, channels)
INPUT_SHAPE = {input_shape}
NUM_CLASSES = 10  # TODO: Set to your number of classes

# Optimizer Settings
OPTIMIZER = 'adam'  # Options: 'adam', 'sgd', 'rmsprop', 'adamw'
MOMENTUM = 0.9  # For SGD
BETAS = (0.9, 0.999)  # For Adam/AdamW

# Learning Rate Scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = 'reduce_on_plateau'  # Options: 'reduce_on_plateau', 'exponential', 'cosine'
LR_PATIENCE = 3  # For ReduceLROnPlateau
LR_FACTOR = 0.5  # For ReduceLROnPlateau
DECAY_STEPS = 1000  # For ExponentialDecay
DECAY_RATE = 0.96  # For ExponentialDecay

# Early Stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5

# Data Augmentation (for training)
USE_AUGMENTATION = True
RANDOM_FLIP = True
RANDOM_ROTATION = True
RANDOM_ZOOM = True
RANDOM_BRIGHTNESS = True
RANDOM_CONTRAST = True

# Augmentation parameters
ROTATION_RANGE = 15
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
ZOOM_RANGE = 0.1

# Normalization (ImageNet statistics)
NORMALIZE = True
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Device Configuration
USE_GPU = True  # Use GPU if available
MEMORY_GROWTH = True  # Allow GPU memory to grow as needed

# Mixed Precision Training (for faster training on modern GPUs)
USE_MIXED_PRECISION = False

# Checkpointing
SAVE_BEST_ONLY = True
CHECKPOINT_DIR = './checkpoints'
SAVE_FREQUENCY = 1  # Save every N epochs

# Logging
USE_TENSORBOARD = True
TENSORBOARD_DIR = './logs'
LOG_HISTOGRAMS = True

# Data Loading
NUM_PARALLEL_CALLS = tf.data.AUTOTUNE if 'tf' in dir() else 4
PREFETCH_BUFFER = tf.data.AUTOTUNE if 'tf' in dir() else 2

# Paths
DATA_DIR = './data'
TRAIN_DIR = DATA_DIR + '/train'
VAL_DIR = DATA_DIR + '/val'
TEST_DIR = DATA_DIR + '/test'

# Model specific
DROPOUT_RATE = 0.5
BATCH_NORM_MOMENTUM = 0.99
BATCH_NORM_EPSILON = 0.001

# Import TensorFlow for AUTOTUNE constant
try:
    import tensorflow as tf
    NUM_PARALLEL_CALLS = tf.data.AUTOTUNE
    PREFETCH_BUFFER = tf.data.AUTOTUNE
except ImportError:
    pass
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
