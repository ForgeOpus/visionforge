"""
PyTorch Code Generation Service
Generates PyTorch nn.Module code from architecture graphs with professional class-based structure
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import deque


def generate_pytorch_code(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    project_name: str = "GeneratedModel"
) -> Dict[str, str]:
    """
    Generate complete PyTorch code including model, training, and data loading.
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


def infer_shapes(nodes: List[Dict], edges: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """
    Infer input/output shapes for each layer in the graph.

    Returns:
        Dictionary mapping node_id to shape info: {'in_channels', 'out_channels', 'in_features', 'out_features', etc.}
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
            # Parse input shape
            shape_str = config.get('shape', '[1, 3, 224, 224]')
            try:
                # Try to parse shape
                import json
                shape = json.loads(shape_str)
                if len(shape) >= 4:
                    shape_info['out_channels'] = shape[1]  # NCHW format
                    shape_info['out_height'] = shape[2]
                    shape_info['out_width'] = shape[3]
                elif len(shape) >= 2:
                    shape_info['out_features'] = shape[1]
            except:
                shape_info['out_channels'] = 3
                shape_info['out_height'] = 224
                shape_info['out_width'] = 224

        elif node_type == 'conv2d':
            # Get input channels from previous layer
            if incoming and incoming[0] in shape_map:
                shape_info['in_channels'] = shape_map[incoming[0]].get('out_channels', 3)
            else:
                shape_info['in_channels'] = 3

            # Output channels from config
            shape_info['out_channels'] = config.get('out_channels', 64)

            # Calculate output spatial dimensions
            if incoming and incoming[0] in shape_map:
                prev_shape = shape_map[incoming[0]]
                kernel_size = config.get('kernel_size', 3)
                stride = config.get('stride', 1)
                padding = config.get('padding', 0)

                if 'out_height' in prev_shape and 'out_width' in prev_shape:
                    shape_info['out_height'] = (prev_shape['out_height'] + 2*padding - kernel_size) // stride + 1
                    shape_info['out_width'] = (prev_shape['out_width'] + 2*padding - kernel_size) // stride + 1

        elif node_type == 'maxpool2d':
            # Preserve channels, reduce spatial dimensions
            if incoming and incoming[0] in shape_map:
                prev_shape = shape_map[incoming[0]]
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
            if incoming and incoming[0] in shape_map:
                prev_shape = shape_map[incoming[0]]
                channels = prev_shape.get('out_channels', 64)
                height = prev_shape.get('out_height', 7)
                width = prev_shape.get('out_width', 7)
                shape_info['out_features'] = channels * height * width

        elif node_type == 'linear':
            # Get input features from previous layer
            if incoming and incoming[0] in shape_map:
                shape_info['in_features'] = shape_map[incoming[0]].get('out_features', 512)
            else:
                shape_info['in_features'] = 512

            # Output features from config
            shape_info['out_features'] = config.get('out_features', 128)

        elif node_type == 'batchnorm':
            # Preserve dimensions, just need num_features
            if incoming and incoming[0] in shape_map:
                prev_shape = shape_map[incoming[0]]
                shape_info['num_features'] = prev_shape.get('out_channels', 64)
                shape_info['out_channels'] = shape_info['num_features']
                if 'out_height' in prev_shape:
                    shape_info['out_height'] = prev_shape['out_height']
                if 'out_width' in prev_shape:
                    shape_info['out_width'] = prev_shape['out_width']

        elif node_type == 'avgpool2d':
            # Preserve channels, reduce spatial dimensions
            if incoming and incoming[0] in shape_map:
                prev_shape = shape_map[incoming[0]]
                shape_info['in_channels'] = prev_shape.get('out_channels', 64)
                shape_info['out_channels'] = shape_info['in_channels']

                kernel_size = config.get('kernel_size', 2)
                stride = config.get('stride', 2)
                padding = config.get('padding', 0)

                if 'out_height' in prev_shape and 'out_width' in prev_shape:
                    shape_info['out_height'] = (prev_shape['out_height'] + 2*padding - kernel_size) // stride + 1
                    shape_info['out_width'] = (prev_shape['out_width'] + 2*padding - kernel_size) // stride + 1

        elif node_type == 'adaptiveavgpool2d':
            # Preserve channels, set fixed output size
            if incoming and incoming[0] in shape_map:
                prev_shape = shape_map[incoming[0]]
                shape_info['in_channels'] = prev_shape.get('out_channels', 64)
                shape_info['out_channels'] = shape_info['in_channels']

                # Parse output size
                output_size_str = str(config.get('output_size', '1'))
                try:
                    if "," in output_size_str or "[" in output_size_str:
                        output_size_str = output_size_str.strip("[]()").replace(" ", "")
                        parts = output_size_str.split(",")
                        shape_info['out_height'] = int(parts[0])
                        shape_info['out_width'] = int(parts[1]) if len(parts) > 1 else int(parts[0])
                    else:
                        shape_info['out_height'] = int(output_size_str)
                        shape_info['out_width'] = int(output_size_str)
                except:
                    shape_info['out_height'] = 1
                    shape_info['out_width'] = 1

        elif node_type == 'conv1d':
            # Get input channels from previous layer
            if incoming and incoming[0] in shape_map:
                shape_info['in_channels'] = shape_map[incoming[0]].get('out_channels', 1)
            else:
                shape_info['in_channels'] = 1

            # Output channels from config
            shape_info['out_channels'] = config.get('out_channels', 64)

        elif node_type == 'conv3d':
            # Get input channels from previous layer
            if incoming and incoming[0] in shape_map:
                shape_info['in_channels'] = shape_map[incoming[0]].get('out_channels', 1)
            else:
                shape_info['in_channels'] = 1

            # Output channels from config
            shape_info['out_channels'] = config.get('out_channels', 64)

        elif node_type in ('lstm', 'gru'):
            # Get input features from previous layer
            if incoming and incoming[0] in shape_map:
                prev_shape = shape_map[incoming[0]]
                shape_info['in_features'] = prev_shape.get('out_features', 128)
            else:
                shape_info['in_features'] = 128

            # Output features based on hidden size and bidirectional
            hidden_size = config.get('hidden_size', 128)
            bidirectional = config.get('bidirectional', False)
            shape_info['out_features'] = hidden_size * (2 if bidirectional else 1)

        elif node_type == 'embedding':
            # Output features from embedding dimension
            shape_info['out_features'] = config.get('embedding_dim', 128)

        elif node_type == 'concat':
            # Sum channels/features along concat dimension
            if incoming:
                dim = config.get('dim', 1)
                total = 0
                for src_id in incoming:
                    if src_id in shape_map:
                        prev_shape = shape_map[src_id]
                        if dim == 1:  # Channel dimension for NCHW
                            total += prev_shape.get('out_channels', prev_shape.get('out_features', 0))
                        else:
                            total += prev_shape.get('out_features', prev_shape.get('out_channels', 0))

                if dim == 1 and total > 0:
                    shape_info['out_channels'] = total
                    # Copy spatial dims from first input
                    if incoming[0] in shape_map:
                        first = shape_map[incoming[0]]
                        if 'out_height' in first:
                            shape_info['out_height'] = first['out_height']
                        if 'out_width' in first:
                            shape_info['out_width'] = first['out_width']
                else:
                    shape_info['out_features'] = total if total > 0 else 256

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
Generated PyTorch Model
Architecture: {class_name}
Generated by VisionForge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


'''

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

    elif node_type == 'maxpool2d':
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

    elif node_type == 'avgpool2d':
        kernel_size = config.get('kernel_size', 2)
        stride = config.get('stride', 2)
        padding = config.get('padding', 0)

        return f'''class {class_name}(nn.Module):
    """
    2D Average Pooling Layer

    Applies a 2D average pooling over an input signal.
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
        """Initialize the average pooling layer."""
        super({class_name}, self).__init__()
        self.pool = nn.AvgPool2d(
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
        # Apply average pooling
        x = self.pool(x)
        return x'''

    elif node_type == 'adaptiveavgpool2d':
        output_size_str = str(config.get('output_size', '1'))

        # Parse output size
        if "," in output_size_str or "[" in output_size_str:
            output_size_str = output_size_str.strip("[]()").replace(" ", "")
            parts = output_size_str.split(",")
            out_h = int(parts[0])
            out_w = int(parts[1]) if len(parts) > 1 else out_h
            output_size = f"({out_h}, {out_w})"
        else:
            out_h = out_w = int(output_size_str)
            output_size = f"{out_h}"

        return f'''class {class_name}(nn.Module):
    """
    Adaptive 2D Average Pooling Layer

    Applies adaptive average pooling to produce output of specified size.
    Automatically calculates kernel size and stride based on input and output sizes.

    Parameters:
        - Output size: {out_h}x{out_w}

    Shape:
        - Input: [batch_size, C, H, W]
        - Output: [batch_size, C, {out_h}, {out_w}]
    """

    def __init__(self):
        """Initialize the adaptive average pooling layer."""
        super({class_name}, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d({output_size})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the adaptive pooling layer.

        Args:
            x: Input tensor of shape [batch, C, H, W]

        Returns:
            Output tensor of shape [batch, C, {out_h}, {out_w}]
        """
        # Apply adaptive average pooling
        x = self.pool(x)
        return x'''

    elif node_type == 'conv1d':
        in_channels = shape_info.get('in_channels', 1)
        out_channels = config.get('out_channels', 64)
        kernel_size = config.get('kernel_size', 3)
        stride = config.get('stride', 1)
        padding = config.get('padding', 0)
        dilation = config.get('dilation', 1)
        bias = config.get('bias', True)

        return f'''class {class_name}(nn.Module):
    """
    1D Convolutional Layer

    Applies a 1D convolution over an input signal composed of several input channels.
    Commonly used for sequence data like time series or text.

    Parameters:
        - Input channels: {in_channels}
        - Output channels: {out_channels}
        - Kernel size: {kernel_size}
        - Stride: {stride}
        - Padding: {padding}
        - Dilation: {dilation}

    Shape:
        - Input: [batch_size, {in_channels}, L]
        - Output: [batch_size, {out_channels}, L_out]
    """

    def __init__(self, in_channels: int = {in_channels}):
        """Initialize the 1D convolutional layer."""
        super({class_name}, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            {out_channels},
            kernel_size={kernel_size},
            stride={stride},
            padding={padding},
            dilation={dilation},
            bias={bias}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the 1D convolutional layer.

        Args:
            x: Input tensor of shape [batch, {in_channels}, L]

        Returns:
            Output tensor of shape [batch, {out_channels}, L_out]
        """
        # Apply 1D convolution
        x = self.conv(x)
        return x'''

    elif node_type == 'conv3d':
        in_channels = shape_info.get('in_channels', 1)
        out_channels = config.get('out_channels', 64)
        kernel_size = config.get('kernel_size', 3)
        stride = config.get('stride', 1)
        padding = config.get('padding', 0)
        dilation = config.get('dilation', 1)
        bias = config.get('bias', True)

        return f'''class {class_name}(nn.Module):
    """
    3D Convolutional Layer

    Applies a 3D convolution over an input signal composed of several input channels.
    Commonly used for volumetric data like video or 3D medical imaging.

    Parameters:
        - Input channels: {in_channels}
        - Output channels: {out_channels}
        - Kernel size: {kernel_size}x{kernel_size}x{kernel_size}
        - Stride: {stride}
        - Padding: {padding}
        - Dilation: {dilation}

    Shape:
        - Input: [batch_size, {in_channels}, D, H, W]
        - Output: [batch_size, {out_channels}, D_out, H_out, W_out]
    """

    def __init__(self, in_channels: int = {in_channels}):
        """Initialize the 3D convolutional layer."""
        super({class_name}, self).__init__()
        self.conv = nn.Conv3d(
            in_channels,
            {out_channels},
            kernel_size={kernel_size},
            stride={stride},
            padding={padding},
            dilation={dilation},
            bias={bias}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the 3D convolutional layer.

        Args:
            x: Input tensor of shape [batch, {in_channels}, D, H, W]

        Returns:
            Output tensor of shape [batch, {out_channels}, D_out, H_out, W_out]
        """
        # Apply 3D convolution
        x = self.conv(x)
        return x'''

    elif node_type == 'lstm':
        in_features = shape_info.get('in_features', 128)
        hidden_size = config.get('hidden_size', 128)
        num_layers = config.get('num_layers', 1)
        bias = config.get('bias', True)
        batch_first = config.get('batch_first', True)
        dropout = config.get('dropout', 0.0)
        bidirectional = config.get('bidirectional', False)

        output_size = hidden_size * (2 if bidirectional else 1)

        return f'''class {class_name}(nn.Module):
    """
    Long Short-Term Memory (LSTM) Layer

    Applies a multi-layer LSTM RNN to an input sequence.
    Learns long-term dependencies in sequential data.

    Parameters:
        - Input size: {in_features}
        - Hidden size: {hidden_size}
        - Number of layers: {num_layers}
        - Bidirectional: {bidirectional}
        - Dropout: {dropout}

    Shape:
        - Input: [batch_size, seq_len, {in_features}]
        - Output: [batch_size, seq_len, {output_size}]
    """

    def __init__(self, in_features: int = {in_features}):
        """Initialize the LSTM layer."""
        super({class_name}, self).__init__()
        self.lstm = nn.LSTM(
            in_features,
            {hidden_size},
            num_layers={num_layers},
            bias={bias},
            batch_first={batch_first},
            dropout={dropout},
            bidirectional={bidirectional}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM layer.

        Args:
            x: Input tensor of shape [batch, seq_len, {in_features}]

        Returns:
            Output tensor of shape [batch, seq_len, {output_size}]
        """
        # Apply LSTM (returns output, (h_n, c_n))
        x, _ = self.lstm(x)
        return x'''

    elif node_type == 'gru':
        in_features = shape_info.get('in_features', 128)
        hidden_size = config.get('hidden_size', 128)
        num_layers = config.get('num_layers', 1)
        bias = config.get('bias', True)
        batch_first = config.get('batch_first', True)
        dropout = config.get('dropout', 0.0)
        bidirectional = config.get('bidirectional', False)

        output_size = hidden_size * (2 if bidirectional else 1)

        return f'''class {class_name}(nn.Module):
    """
    Gated Recurrent Unit (GRU) Layer

    Applies a multi-layer GRU RNN to an input sequence.
    Simpler alternative to LSTM with fewer parameters.

    Parameters:
        - Input size: {in_features}
        - Hidden size: {hidden_size}
        - Number of layers: {num_layers}
        - Bidirectional: {bidirectional}
        - Dropout: {dropout}

    Shape:
        - Input: [batch_size, seq_len, {in_features}]
        - Output: [batch_size, seq_len, {output_size}]
    """

    def __init__(self, in_features: int = {in_features}):
        """Initialize the GRU layer."""
        super({class_name}, self).__init__()
        self.gru = nn.GRU(
            in_features,
            {hidden_size},
            num_layers={num_layers},
            bias={bias},
            batch_first={batch_first},
            dropout={dropout},
            bidirectional={bidirectional}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GRU layer.

        Args:
            x: Input tensor of shape [batch, seq_len, {in_features}]

        Returns:
            Output tensor of shape [batch, seq_len, {output_size}]
        """
        # Apply GRU (returns output, h_n)
        x, _ = self.gru(x)
        return x'''

    elif node_type == 'embedding':
        num_embeddings = config.get('num_embeddings', 10000)
        embedding_dim = config.get('embedding_dim', 128)
        padding_idx = config.get('padding_idx', -1)
        max_norm = config.get('max_norm', 0)
        scale_grad_by_freq = config.get('scale_grad_by_freq', False)

        # Build optional parameters
        optional_params = []
        if padding_idx >= 0:
            optional_params.append(f"padding_idx={padding_idx}")
        if max_norm > 0:
            optional_params.append(f"max_norm={max_norm}")
        if scale_grad_by_freq:
            optional_params.append(f"scale_grad_by_freq={scale_grad_by_freq}")

        optional_params_str = ",\n            ".join(optional_params)
        if optional_params_str:
            optional_params_str = ",\n            " + optional_params_str

        return f'''class {class_name}(nn.Module):
    """
    Embedding Layer

    A lookup table that stores embeddings of a fixed dictionary and size.
    Commonly used to store word embeddings for NLP tasks.

    Parameters:
        - Vocabulary size: {num_embeddings}
        - Embedding dimension: {embedding_dim}
        - Padding index: {padding_idx if padding_idx >= 0 else 'None'}

    Shape:
        - Input: [batch_size, seq_len] (LongTensor of indices)
        - Output: [batch_size, seq_len, {embedding_dim}]
    """

    def __init__(self):
        """Initialize the embedding layer."""
        super({class_name}, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings={num_embeddings},
            embedding_dim={embedding_dim}{optional_params_str}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the embedding layer.

        Args:
            x: Input tensor of indices [batch, seq_len]

        Returns:
            Output tensor of embeddings [batch, seq_len, {embedding_dim}]
        """
        # Look up embeddings
        x = self.embedding(x)
        return x'''

    elif node_type == 'concat':
        dim = config.get('dim', 1)

        return f'''class {class_name}(nn.Module):
    """
    Concatenation Layer

    Concatenates multiple input tensors along a specified dimension.
    Used for skip connections and multi-path architectures.

    Parameters:
        - Dimension: {dim}

    Shape:
        - Input: List of tensors with compatible shapes
        - Output: Single concatenated tensor
    """

    def __init__(self):
        """Initialize the concatenation layer."""
        super({class_name}, self).__init__()
        self.dim = {dim}

    def forward(self, inputs: list) -> torch.Tensor:
        """
        Forward pass through the concatenation layer.

        Args:
            inputs: List of input tensors to concatenate

        Returns:
            Concatenated output tensor
        """
        # Concatenate along specified dimension
        x = torch.cat(inputs, dim=self.dim)
        return x'''

    elif node_type == 'add':
        return f'''class {class_name}(nn.Module):
    """
    Element-wise Addition Layer

    Performs element-wise addition of multiple input tensors.
    Used for residual connections in architectures like ResNet.

    Shape:
        - Input: List of tensors with identical shapes
        - Output: Single tensor with same shape as inputs
    """

    def __init__(self):
        """Initialize the addition layer."""
        super({class_name}, self).__init__()

    def forward(self, inputs: list) -> torch.Tensor:
        """
        Forward pass through the addition layer.

        Args:
            inputs: List of input tensors to add

        Returns:
            Sum of input tensors
        """
        # Element-wise addition
        x = inputs[0]
        for tensor in inputs[1:]:
            x = x + tensor
        return x'''

    return None


def generate_layer_instantiation(
    class_name: str,
    layer_name: str,
    shape_info: Dict[str, Any]
) -> str:
    """Generate layer instantiation line for __init__ method"""
    # Determine if layer needs arguments
    if 'in_channels' in shape_info:
        in_ch = shape_info['in_channels']
        return f"self.{layer_name} = {class_name}(in_channels={in_ch})  # Input: {in_ch} channels"
    elif 'in_features' in shape_info:
        in_feat = shape_info['in_features']
        return f"self.{layer_name} = {class_name}(in_features={in_feat})  # Input: {in_feat} features"
    elif 'num_features' in shape_info:
        num_feat = shape_info['num_features']
        return f"self.{layer_name} = {class_name}(num_features={num_feat})  # {num_feat} features"
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
    """Generate descriptive class name for layer"""
    type_name = node_type.replace('_', '').title()

    # Add descriptive suffix based on config
    if node_type == 'conv2d':
        channels = config.get('out_channels', 64)
        kernel = config.get('kernel_size', 3)
        return f"{type_name}Layer_{channels}ch_{kernel}x{kernel}"
    elif node_type == 'conv1d':
        channels = config.get('out_channels', 64)
        kernel = config.get('kernel_size', 3)
        return f"{type_name}Layer_{channels}ch_{kernel}"
    elif node_type == 'conv3d':
        channels = config.get('out_channels', 64)
        kernel = config.get('kernel_size', 3)
        return f"{type_name}Layer_{channels}ch_{kernel}x{kernel}x{kernel}"
    elif node_type == 'linear':
        features = config.get('out_features', 128)
        return f"{type_name}Layer_{features}units"
    elif node_type == 'maxpool2d':
        kernel = config.get('kernel_size', 2)
        return f"{type_name}Layer_{kernel}x{kernel}"
    elif node_type == 'avgpool2d':
        kernel = config.get('kernel_size', 2)
        return f"AvgPool2DLayer_{kernel}x{kernel}"
    elif node_type == 'adaptiveavgpool2d':
        output_size = config.get('output_size', '1')
        return f"AdaptiveAvgPool2DLayer_{output_size}"
    elif node_type == 'lstm':
        hidden = config.get('hidden_size', 128)
        return f"LSTMLayer_{hidden}hidden"
    elif node_type == 'gru':
        hidden = config.get('hidden_size', 128)
        return f"GRULayer_{hidden}hidden"
    elif node_type == 'embedding':
        dim = config.get('embedding_dim', 128)
        return f"EmbeddingLayer_{dim}dim"
    elif node_type == 'concat':
        dim = config.get('dim', 1)
        return f"ConcatLayer_dim{dim}"
    elif node_type == 'add':
        return f"AddLayer_{idx}"
    else:
        return f"{type_name}Layer_{idx}"


def get_layer_variable_name(node_type: str, idx: int, config: Dict[str, Any]) -> str:
    """Generate descriptive variable name for layer instance"""
    # Create readable names based on layer type
    if node_type == 'conv2d':
        channels = config.get('out_channels', 64)
        return f"conv_{channels}ch"
    elif node_type == 'conv1d':
        channels = config.get('out_channels', 64)
        return f"conv1d_{channels}ch"
    elif node_type == 'conv3d':
        channels = config.get('out_channels', 64)
        return f"conv3d_{channels}ch"
    elif node_type == 'linear':
        features = config.get('out_features', 128)
        return f"fc_{features}"
    elif node_type == 'maxpool2d':
        return f"maxpool_{idx}"
    elif node_type == 'avgpool2d':
        return f"avgpool_{idx}"
    elif node_type == 'adaptiveavgpool2d':
        return f"adaptive_avgpool_{idx}"
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
