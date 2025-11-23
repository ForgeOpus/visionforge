"""
TensorFlow/Keras Code Generation Service
Generates tf.keras.Model code from architecture graphs with professional class-based structure
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import deque


def generate_tensorflow_code(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    project_name: str = "GeneratedModel"
) -> Dict[str, str]:
    """
    Generate complete TensorFlow/Keras code including model, training, and data loading.
    Each layer gets its own reusable class, all combined in a main model class.

    Args:
        nodes: List of node dictionaries from architecture
        edges: List of edge dictionaries defining connections
        project_name: Name for the generated model class

    Returns:
        Dictionary with keys: 'model', 'train', 'dataset', 'config'
    """
    # Topologically sort nodes
    sorted_nodes = topological_sort(nodes, edges)

    # Infer shapes through the graph
    shape_map = infer_shapes(sorted_nodes, edges)

    # Generate different components
    model_code = generate_model_file(sorted_nodes, edges, project_name, shape_map)
    train_code = generate_training_script(project_name)
    dataset_code = generate_dataset_class(nodes)
    config_code = generate_config_file(nodes)

    return {
        'model': model_code,
        'train': train_code,
        'dataset': dataset_code,
        'config': config_code
    }


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


def infer_shapes(nodes: List[Dict], edges: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """
    Infer input/output shapes for each layer in the graph.
    TensorFlow uses NHWC format (batch, height, width, channels).

    Returns:
        Dictionary mapping node_id to shape info: {'in_channels', 'out_channels', 'in_units', 'out_units', etc.}
    """
    shape_map = {}

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

        # Initialize shape info for this node
        shape_info = {}

        if node_type == 'input':
            # Parse input shape (NHWC format)
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
            except:
                shape_info['out_height'] = 224
                shape_info['out_width'] = 224
                shape_info['out_channels'] = 3

        elif node_type == 'conv2d':
            # Get input channels from previous layer
            if incoming and incoming[0] in shape_map:
                shape_info['in_channels'] = shape_map[incoming[0]].get('out_channels', 3)
            else:
                shape_info['in_channels'] = 3

            # Output channels (filters) from config
            shape_info['out_channels'] = config.get('filters', 64)

            # Calculate output spatial dimensions
            if incoming and incoming[0] in shape_map:
                prev_shape = shape_map[incoming[0]]
                kernel_size = config.get('kernel_size', 3)
                strides = config.get('strides', 1)
                padding = config.get('padding', 'valid')

                if 'out_height' in prev_shape and 'out_width' in prev_shape:
                    if padding == 'same':
                        # Same padding preserves dimensions (with stride)
                        shape_info['out_height'] = (prev_shape['out_height'] + strides - 1) // strides
                        shape_info['out_width'] = (prev_shape['out_width'] + strides - 1) // strides
                    else:  # valid padding
                        shape_info['out_height'] = (prev_shape['out_height'] - kernel_size) // strides + 1
                        shape_info['out_width'] = (prev_shape['out_width'] - kernel_size) // strides + 1

        elif node_type in ('maxpool2d', 'maxpool'):
            # Preserve channels, reduce spatial dimensions
            if incoming and incoming[0] in shape_map:
                prev_shape = shape_map[incoming[0]]
                shape_info['in_channels'] = prev_shape.get('out_channels', 64)
                shape_info['out_channels'] = shape_info['in_channels']

                pool_size = config.get('pool_size', 2)
                strides = config.get('strides', 2)
                padding = config.get('padding', 'valid')

                if 'out_height' in prev_shape and 'out_width' in prev_shape:
                    if padding == 'same':
                        shape_info['out_height'] = (prev_shape['out_height'] + strides - 1) // strides
                        shape_info['out_width'] = (prev_shape['out_width'] + strides - 1) // strides
                    else:  # valid padding
                        shape_info['out_height'] = (prev_shape['out_height'] - pool_size) // strides + 1
                        shape_info['out_width'] = (prev_shape['out_width'] - pool_size) // strides + 1

        elif node_type == 'flatten':
            # Convert spatial dimensions to units
            if incoming and incoming[0] in shape_map:
                prev_shape = shape_map[incoming[0]]
                channels = prev_shape.get('out_channels', 64)
                height = prev_shape.get('out_height', 7)
                width = prev_shape.get('out_width', 7)
                shape_info['out_units'] = channels * height * width

        elif node_type == 'linear':
            # Get input units from previous layer
            if incoming and incoming[0] in shape_map:
                shape_info['in_units'] = shape_map[incoming[0]].get('out_units', 512)
            else:
                shape_info['in_units'] = 512

            # Output units from config
            shape_info['out_units'] = config.get('units', 128)

        elif node_type in ('batchnorm', 'batchnorm2d'):
            # Preserve dimensions
            if incoming and incoming[0] in shape_map:
                prev_shape = shape_map[incoming[0]]
                shape_info.update(prev_shape)

        elif node_type == 'avgpool2d':
            # Preserve channels, reduce spatial dimensions
            if incoming and incoming[0] in shape_map:
                prev_shape = shape_map[incoming[0]]
                shape_info['in_channels'] = prev_shape.get('out_channels', 64)
                shape_info['out_channels'] = shape_info['in_channels']

                pool_size = config.get('pool_size', 2)
                strides = config.get('strides', 2)
                padding = config.get('padding', 'valid')

                if 'out_height' in prev_shape and 'out_width' in prev_shape:
                    if padding == 'same':
                        shape_info['out_height'] = (prev_shape['out_height'] + strides - 1) // strides
                        shape_info['out_width'] = (prev_shape['out_width'] + strides - 1) // strides
                    else:  # valid padding
                        shape_info['out_height'] = (prev_shape['out_height'] - pool_size) // strides + 1
                        shape_info['out_width'] = (prev_shape['out_width'] - pool_size) // strides + 1

        elif node_type == 'adaptiveavgpool2d':
            # Global average pooling - output is [batch, channels] or [batch, 1, 1, channels]
            if incoming and incoming[0] in shape_map:
                prev_shape = shape_map[incoming[0]]
                shape_info['in_channels'] = prev_shape.get('out_channels', 64)

                keepdims = config.get('keepdims', False)
                if keepdims:
                    shape_info['out_channels'] = shape_info['in_channels']
                    shape_info['out_height'] = 1
                    shape_info['out_width'] = 1
                else:
                    shape_info['out_units'] = shape_info['in_channels']

        elif node_type == 'conv1d':
            # Get input channels from previous layer
            if incoming and incoming[0] in shape_map:
                shape_info['in_channels'] = shape_map[incoming[0]].get('out_channels', 1)
            else:
                shape_info['in_channels'] = 1

            # Output channels (filters) from config
            shape_info['out_channels'] = config.get('filters', 64)

        elif node_type == 'conv3d':
            # Get input channels from previous layer
            if incoming and incoming[0] in shape_map:
                shape_info['in_channels'] = shape_map[incoming[0]].get('out_channels', 1)
            else:
                shape_info['in_channels'] = 1

            # Output channels (filters) from config
            shape_info['out_channels'] = config.get('filters', 64)

        elif node_type in ('lstm', 'gru'):
            # Get input units from previous layer
            if incoming and incoming[0] in shape_map:
                prev_shape = shape_map[incoming[0]]
                shape_info['in_units'] = prev_shape.get('out_units', 128)
            else:
                shape_info['in_units'] = 128

            # Output units from config
            shape_info['out_units'] = config.get('units', 128)

        elif node_type == 'embedding':
            # Output units from output dimension
            shape_info['out_units'] = config.get('output_dim', 128)

        elif node_type == 'concat':
            # Sum channels/units along concat dimension
            if incoming:
                total = 0
                for src_id in incoming:
                    if src_id in shape_map:
                        prev_shape = shape_map[src_id]
                        # TensorFlow concat typically on last axis (channels for NHWC)
                        total += prev_shape.get('out_channels', prev_shape.get('out_units', 0))

                if total > 0:
                    # Check if spatial dimensions exist
                    if incoming[0] in shape_map and 'out_height' in shape_map[incoming[0]]:
                        shape_info['out_channels'] = total
                        first = shape_map[incoming[0]]
                        shape_info['out_height'] = first.get('out_height', 1)
                        shape_info['out_width'] = first.get('out_width', 1)
                    else:
                        shape_info['out_units'] = total

        elif node_type == 'add':
            # Preserve shape from first input (all inputs must have same shape)
            if incoming and incoming[0] in shape_map:
                prev_shape = shape_map[incoming[0]]
                shape_info.update(prev_shape)

        else:
            # For other layers, try to preserve shape from input
            if incoming and incoming[0] in shape_map:
                prev_shape = shape_map[incoming[0]]
                shape_info.update(prev_shape)

        shape_map[node_id] = shape_info

    return shape_map


def generate_model_file(
    nodes: List[Dict],
    edges: List[Dict],
    project_name: str,
    shape_map: Dict[str, Dict[str, Any]]
) -> str:
    """Generate complete model.py file with layer classes and main model class"""

    class_name = to_class_name(project_name)

    # Generate individual layer classes
    layer_classes = []
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

        # Generate layer class
        layer_class_code = generate_layer_class(node, idx, config, node_type, shape_info)
        if layer_class_code:
            layer_classes.append(layer_class_code)

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

    elif node_type == 'avgpool2d':
        pool_size = config.get('pool_size', 2)
        strides = config.get('strides', 2)
        padding = config.get('padding', 'valid')

        return f'''class {class_name}(layers.Layer):
    """
    2D Average Pooling Layer

    Applies a 2D average pooling over an input signal.
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
        """Initialize the average pooling layer."""
        super({class_name}, self).__init__()
        self.pool = layers.AveragePooling2D(
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
        # Apply average pooling
        x = self.pool(inputs)
        return x'''

    elif node_type == 'adaptiveavgpool2d':
        keepdims = config.get('keepdims', False)

        return f'''class {class_name}(layers.Layer):
    """
    Global Average Pooling Layer

    Applies global average pooling to produce a single value per channel.
    TensorFlow equivalent of PyTorch's AdaptiveAvgPool2d.

    Parameters:
        - Keep dimensions: {keepdims}

    Shape:
        - Input: [batch, H, W, C] (NHWC format)
        - Output: [batch, C] or [batch, 1, 1, C] if keepdims
    """

    def __init__(self):
        """Initialize the global average pooling layer."""
        super({class_name}, self).__init__()
        self.pool = layers.GlobalAveragePooling2D(keepdims={keepdims})

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the global pooling layer.

        Args:
            inputs: Input tensor of shape [batch, H, W, C]
            training: Whether in training mode

        Returns:
            Output tensor of shape [batch, C] or [batch, 1, 1, C]
        """
        # Apply global average pooling
        x = self.pool(inputs)
        return x'''

    elif node_type == 'conv1d':
        filters = config.get('filters', 64)
        kernel_size = config.get('kernel_size', 3)
        strides = config.get('strides', 1)
        padding = config.get('padding', 'valid')

        return f'''class {class_name}(layers.Layer):
    """
    1D Convolutional Layer

    Applies a 1D convolution over an input signal.
    Commonly used for sequence data like time series or text.

    Parameters:
        - Filters: {filters}
        - Kernel size: {kernel_size}
        - Strides: {strides}
        - Padding: '{padding}'

    Shape:
        - Input: [batch, steps, channels]
        - Output: [batch, steps_out, {filters}]
    """

    def __init__(self):
        """Initialize the 1D convolutional layer."""
        super({class_name}, self).__init__()
        self.conv = layers.Conv1D(
            filters={filters},
            kernel_size={kernel_size},
            strides={strides},
            padding='{padding}'
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the 1D convolutional layer.

        Args:
            inputs: Input tensor of shape [batch, steps, channels]
            training: Whether in training mode

        Returns:
            Output tensor of shape [batch, steps_out, {filters}]
        """
        # Apply 1D convolution
        x = self.conv(inputs)
        return x'''

    elif node_type == 'conv3d':
        filters = config.get('filters', 64)
        kernel_size = config.get('kernel_size', 3)
        strides = config.get('strides', 1)
        padding = config.get('padding', 'valid')

        return f'''class {class_name}(layers.Layer):
    """
    3D Convolutional Layer

    Applies a 3D convolution over an input signal.
    Commonly used for volumetric data like video or 3D medical imaging.

    Parameters:
        - Filters: {filters}
        - Kernel size: {kernel_size}x{kernel_size}x{kernel_size}
        - Strides: {strides}
        - Padding: '{padding}'

    Shape:
        - Input: [batch, D, H, W, C] (NDHWC format)
        - Output: [batch, D_out, H_out, W_out, {filters}]
    """

    def __init__(self):
        """Initialize the 3D convolutional layer."""
        super({class_name}, self).__init__()
        self.conv = layers.Conv3D(
            filters={filters},
            kernel_size={kernel_size},
            strides={strides},
            padding='{padding}'
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the 3D convolutional layer.

        Args:
            inputs: Input tensor of shape [batch, D, H, W, C]
            training: Whether in training mode

        Returns:
            Output tensor of shape [batch, D_out, H_out, W_out, {filters}]
        """
        # Apply 3D convolution
        x = self.conv(inputs)
        return x'''

    elif node_type == 'lstm':
        units = config.get('units', 128)
        return_sequences = config.get('return_sequences', False)
        dropout = config.get('dropout', 0.0)
        recurrent_dropout = config.get('recurrent_dropout', 0.0)

        output_shape = f"[batch, timesteps, {units}]" if return_sequences else f"[batch, {units}]"

        return f'''class {class_name}(layers.Layer):
    """
    Long Short-Term Memory (LSTM) Layer

    Applies an LSTM RNN to an input sequence.
    Learns long-term dependencies in sequential data.

    Parameters:
        - Units: {units}
        - Return sequences: {return_sequences}
        - Dropout: {dropout}
        - Recurrent dropout: {recurrent_dropout}

    Shape:
        - Input: [batch, timesteps, features]
        - Output: {output_shape}
    """

    def __init__(self):
        """Initialize the LSTM layer."""
        super({class_name}, self).__init__()
        self.lstm = layers.LSTM(
            units={units},
            return_sequences={return_sequences},
            dropout={dropout},
            recurrent_dropout={recurrent_dropout}
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the LSTM layer.

        Args:
            inputs: Input tensor of shape [batch, timesteps, features]
            training: Whether in training mode

        Returns:
            Output tensor of shape {output_shape}
        """
        # Apply LSTM
        x = self.lstm(inputs, training=training)
        return x'''

    elif node_type == 'gru':
        units = config.get('units', 128)
        return_sequences = config.get('return_sequences', False)
        dropout = config.get('dropout', 0.0)
        recurrent_dropout = config.get('recurrent_dropout', 0.0)

        output_shape = f"[batch, timesteps, {units}]" if return_sequences else f"[batch, {units}]"

        return f'''class {class_name}(layers.Layer):
    """
    Gated Recurrent Unit (GRU) Layer

    Applies a GRU RNN to an input sequence.
    Simpler alternative to LSTM with fewer parameters.

    Parameters:
        - Units: {units}
        - Return sequences: {return_sequences}
        - Dropout: {dropout}
        - Recurrent dropout: {recurrent_dropout}

    Shape:
        - Input: [batch, timesteps, features]
        - Output: {output_shape}
    """

    def __init__(self):
        """Initialize the GRU layer."""
        super({class_name}, self).__init__()
        self.gru = layers.GRU(
            units={units},
            return_sequences={return_sequences},
            dropout={dropout},
            recurrent_dropout={recurrent_dropout}
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the GRU layer.

        Args:
            inputs: Input tensor of shape [batch, timesteps, features]
            training: Whether in training mode

        Returns:
            Output tensor of shape {output_shape}
        """
        # Apply GRU
        x = self.gru(inputs, training=training)
        return x'''

    elif node_type == 'embedding':
        input_dim = config.get('input_dim', 10000)
        output_dim = config.get('output_dim', 128)
        mask_zero = config.get('mask_zero', False)

        return f'''class {class_name}(layers.Layer):
    """
    Embedding Layer

    Turns positive integers (indexes) into dense vectors of fixed size.
    Commonly used for text and categorical data.

    Parameters:
        - Input dimension (vocabulary size): {input_dim}
        - Output dimension (embedding size): {output_dim}
        - Mask zero: {mask_zero}

    Shape:
        - Input: [batch, sequence_length] (integer indices)
        - Output: [batch, sequence_length, {output_dim}]
    """

    def __init__(self):
        """Initialize the embedding layer."""
        super({class_name}, self).__init__()
        self.embedding = layers.Embedding(
            input_dim={input_dim},
            output_dim={output_dim},
            mask_zero={mask_zero}
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the embedding layer.

        Args:
            inputs: Input tensor of integer indices [batch, seq_len]
            training: Whether in training mode

        Returns:
            Output tensor of embeddings [batch, seq_len, {output_dim}]
        """
        # Look up embeddings
        x = self.embedding(inputs)
        return x'''

    return None


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
    if node_type in ('dropout', 'batchnorm', 'batchnorm2d', 'lstm', 'gru'):
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
    elif node_type == 'conv1d':
        filters = config.get('filters', 64)
        kernel = config.get('kernel_size', 3)
        return f"Conv1DLayer_{filters}filters_{kernel}"
    elif node_type == 'conv3d':
        filters = config.get('filters', 64)
        kernel = config.get('kernel_size', 3)
        return f"Conv3DLayer_{filters}filters_{kernel}x{kernel}x{kernel}"
    elif node_type == 'linear':
        units = config.get('units', 128)
        return f"DenseLayer_{units}units"
    elif node_type in ('maxpool2d', 'maxpool'):
        pool_size = config.get('pool_size', 2)
        return f"MaxPool2DLayer_{pool_size}x{pool_size}"
    elif node_type == 'avgpool2d':
        pool_size = config.get('pool_size', 2)
        return f"AvgPool2DLayer_{pool_size}x{pool_size}"
    elif node_type == 'adaptiveavgpool2d':
        return f"GlobalAvgPool2DLayer_{idx}"
    elif node_type == 'lstm':
        units = config.get('units', 128)
        return f"LSTMLayer_{units}units"
    elif node_type == 'gru':
        units = config.get('units', 128)
        return f"GRULayer_{units}units"
    elif node_type == 'embedding':
        output_dim = config.get('output_dim', 128)
        return f"EmbeddingLayer_{output_dim}dim"
    else:
        return f"{type_name}Layer_{idx}"


def get_layer_variable_name(node_type: str, idx: int, config: Dict[str, Any]) -> str:
    """Generate descriptive variable name for layer instance"""
    # Create readable names based on layer type
    if node_type == 'conv2d':
        filters = config.get('filters', 64)
        return f"conv_{filters}filters"
    elif node_type == 'conv1d':
        filters = config.get('filters', 64)
        return f"conv1d_{filters}filters"
    elif node_type == 'conv3d':
        filters = config.get('filters', 64)
        return f"conv3d_{filters}filters"
    elif node_type == 'linear':
        units = config.get('units', 128)
        return f"dense_{units}"
    elif node_type in ('maxpool2d', 'maxpool'):
        return f"maxpool_{idx}"
    elif node_type == 'avgpool2d':
        return f"avgpool_{idx}"
    elif node_type == 'adaptiveavgpool2d':
        return f"global_avgpool_{idx}"
    elif node_type == 'flatten':
        return f"flatten"
    elif node_type == 'dropout':
        return f"dropout_{idx}"
    elif node_type in ('batchnorm', 'batchnorm2d'):
        return f"batchnorm_{idx}"
    elif node_type == 'lstm':
        return f"lstm_{idx}"
    elif node_type == 'gru':
        return f"gru_{idx}"
    elif node_type == 'embedding':
        return f"embedding_{idx}"
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
