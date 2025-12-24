from typing import Any, Dict, List, Optional, Tuple
from collections import deque

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

# ===========================================
# CodeGen Logic Classes
# ===========================================

class ClassDefinitionGenerator():
    """
    Generates re-usable forward declarations for all unique layer types
    used in the architecture for DRY implementation.
    """

    group_class_mapper: Dict[str, str] = {}

    @classmethod 
    def get_node_type(cls, node: Dict) -> str:
        """Extract node type from node dictionary"""
        return node.get('data', {}).get('blockType', '')

    @classmethod 
    def _create_conv2d_class(cls) -> str:
        """
        Generate definition for a 2D convolution block.

        Returns:
            str: Definition for a 2D convolution block
        """
        return '''
        class Conv2DBlock(nn.Module):
            """
            2D Convolutional Layer

            Applies a 2D convolution over an input signal composed of several input channels.

            Parameters:
                - Input channels: in_channels
                - Output channels: out_channels
                - Kernel size: kernel_sizexkernel_size
                - Stride: stride
                - Padding: padding
                - Dilation: dilation

            Shape:
                - Input: [batch_size, in_channels, H, W]
                - Output: [batch_size, out_channels, H/stride, W/stride]
            """

            def __init__(self, in_channels: int, out_channels:int, kernel_size=3, stride=3, padding=0, dilation=1):
                """Initialize the convolutional layer."""
                super(Conv2DBlock, self).__init__()
                self.conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                Forward pass through the convolutional layer.

                Args:
                    x: Input tensor of shape [batch, in_channels, H, W]

                Returns:
                    Output tensor of shape [batch, out_channels, out_h, out_w]
                """
                # Apply convolution
                x = self.conv(x)
                return x
        '''
    
    @classmethod 
    def _create_add_class(cls) -> str:
        """
        Generate definition for an Add block for element-wise addition of tensors.

        Returns:
            str: Definition for an Add block
        """
        return '''
class AddBlock(nn.Module):
    """
    Addition block

    Element-wise addition block for tensors of same shape.

    Parameters:
        - tensor_list: list of input tensors of same shape
    
    """

    def __init__(self):
        """
        Initialize the addition layer (no learnable parameters).
        """
        super(AddBlock, self).__init__()
    
    def forward(self, tensor_list:List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the addition layer.
        Returns:
            Element-wise sum of all input tensors
        """
        return torch.stack(tensor_list).sum(dim=0)
        '''

    @classmethod 
    def _create_concat_class(cls) -> str:
        """
        Generate definition for a Concact block for concatination of tensors along a dimension.

        Returns:
            str: Definition for an Add block
        """
        return '''
class ConcatBlock(nn.Module):
    """
    Concatination Layer

    Concatenates multiple tensors along a specified dimension.
    Commonly used to merge feature maps from different paths in the network.

    Parameters:
        - Concatenation dimension: int

    Shape:
        - Input: List of tensors with compatible shapes
        - Output: Concatenated tensor along dimension dim
    """

    def __init__(self):
        """
        Initialize the concatenation layer (no learnable parameters).
        """
        super(ConcatBlock, self).__init__()
    
    def forward(self, tensor_list: List[torch.Tensor], concat_dim: int):
        """
        Forward pass through the addition layer.
        Returns:
            Element-wise sum of all input tensors
        """
        return torch.cat(tensor_list, dim=concat_dim)
        '''

    @classmethod 
    def _create_linear_class(cls) -> str:
        """
        Generate definition for a Fully Connected Linear Layer.

        Returns:
            str: Definition for a FCL layer
        """
        return '''
class LinearLayer(nn.Module):
    """
    Applies a linear transformation to the incoming data: y = xA^T + b

    Parameters:
    - Input features: in_features
    - Output features: out_features
    - Bias: bias

    Shape:
        - Input: [batch_size, in_features]
        - Output: [batch_size, out_features]
    """

    def __init__(self, in_features:int, out_features:int, bias:bool =False):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linear layer.

        Args:
            x: Input tensor of shape [batch, in_features]

        Returns:
            Output tensor of shape [batch, out_features]
        """
        x = self.linear(x)
        return x
        '''

    @classmethod 
    def _create_maxpool_class(cls) -> str:
        """
        Generate definition for a 2D maxpooling layer.

        Returns:
            str: Definition for a 2D maxpooling layer.
        """
        return '''
class MaxPoolBlock(nn.Module):
    """
    2D Max Pooling Layer

    Applies a 2D max pooling over an input signal.
    Reduces spatial dimensions while preserving channel count.

    Parameters:
        - Kernel size: kernel_sizexkernel_size
        - Stride: stride
        - Padding: padding

    Shape:
        - Input: [batch_size, C, H, W]
        - Output: [batch_size, C, H/stride, W/stride]
    """
    def __init__(self, kernel_size: int, stride: int, padding:int):
        """Initialize the max pooling layer."""
        super(MaxPoolBlock, self).__init__()
        self.pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
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
        return x 
        '''
    
    @classmethod 
    def _create_flatten_class(cls) -> str:
        """
        Generate definition for a flatten class.

        Returns:
            str: Definition for a flatten class.
        """
        return '''
class FlattenLayer(nn.Module):
    """
    Flatten Layer

    Flattens a contiguous range of dimensions into a tensor.
    Commonly used to transition from convolutional layers to fully connected layers.

    Shape:
        - Input: [batch_size, C, H, W]
        - Output: [batch_size, C*H*W] = [batch_size, out_features]
    """

    def __init__(self, start_dim:int = 1):
        """Initialize the flatten layer."""
        super(FlattenLayer, self).__init__()
        self.flatten = nn.Flatten(start_dim=start_dim)

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
        return x
        '''
    
    @classmethod 
    def _create_relu_class(cls) -> str:
        """
        Generate definition for a ReLU activation Layer class

        Returns:
            str: Definition for a ReLU class.
        """
        return '''
class ReLUBlock(nn.Module):
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
        super(ReLUBlock, self).__init__()
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
        return x
'''

    @classmethod 
    def _create_softmax_class(cls) -> str:
        """
        Generate definition for a Softmax activation Layer class.

        Returns:
            str: Definition for a Softmax activation Layer class.
        """
        return '''
class SoftmaxBlock(nn.Module):
    """
    Softmax Activation Layer

    Applies the softmax function to normalize outputs into a probability distribution.
    Commonly used in the final layer for classification tasks.

    Parameters:
        - Dimension: dim

    Shape:
        - Input: [batch_size, num_classes]
        - Output: [batch_size, num_classes] (sums to 1.0 along dimension {dim})
    """

    def __init__(self, dim:int = 1):
        """Initialize the softmax layer."""
        super(SoftmaxBlock, self).__init__()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the softmax layer.

        Args:
            x: Input tensor

        Returns:
            Probability distribution over dimension dim
        """
        # Apply softmax activation
        x = self.softmax(x)
        return x
'''

    @classmethod
    def _create_dropout_class(cls) -> str:
        """
        Generate definition for a Dropout Layer class

        Returns:
            str: Definition for a Dropout class.
        """
        return '''
class DropoutBlock(nn.Module):
    """
    Dropout Regularization Layer

    Randomly zeroes some elements of the input tensor with probability p during training.
    Helps prevent overfitting.

    Parameters:
        - Dropout probability: p

    Shape:
        - Input: [batch_size, *] (any shape)
        - Output: [batch_size, *] (same shape as input)
    """

    def __init__(self, p):
        """Initialize the dropout layer."""
        super(DropoutBlock, self).__init__()
        self.dropout = nn.Dropout(p=p)

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
        return x
        '''
    
    @classmethod 
    def _create_batchnorm_class(cls) -> str:
        """
        Generate definition for a Batch Normalization Layer class

        Returns:
            str: Definition for a Batch Normalization Layer class.
        """
        return '''
class BatchNormBlock(nn.Module):
    """
    Batch Normalization Layer

    Normalizes the input over a mini-batch for each feature channel.
    Helps stabilize and accelerate training.

    Parameters:
        - Number of features: num_features
        - Epsilon: eps
        - Momentum: momentum
        - Learnable parameters: affine

    Shape:
        - Input: [batch_size, num_features, H, W]
        - Output: [batch_size, num_features, H, W]
    """

    def __init__(self, num_features: int, eps:float = 1e-5, momentum:float = 0.1, affine: bool = True):
        """Initialize the batch normalization layer."""
        super(BatchNormBlock, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the batch normalization layer.

        Args:
            x: Input tensor of shape [batch, num_features, H, W]

        Returns:
            Normalized output tensor of same shape
        """
        # Apply batch normalization
        x = self.bn(x)
        return x
        '''

    @classmethod 
    def _create_self_attention_block(cls) -> str:
        """
        Generate definition for a Multi-Headed Self-Attention Layer class

        Returns:
            str: Definition for a Multi-Headed Self-Attention Layer class.
        """
        return '''
class MultiHeadSelfAttentionBlock(nn.Module):
    """
    Multi-Head Self-Attention Layer

    Applies multi-head self-attention mechanism to the input.
    Allows the model to jointly attend to information from different representation subspaces.

    Parameters:
        - Embedding dimension: embed_dim
        - Number of heads: num_heads
        - Dropout: dropout

    Shape:
        - Input: [batch_size, seq_len, embed_dim]
        - Output: [batch_size, seq_len, embed_dim]
    """

    def __init__(self, embed_dim:int = 1, num_heads:int = 8, dropout:float = 0.0):
        """Initialize the multi-head attention layer."""
        super(MultiHeadSelfAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention layer.

        Args:
            x: Input tensor of shape [batch, seq_len, embed_dim]

        Returns:
            Output tensor after applying multi-head attention
        """
        # Apply self-attention (query, key, value are all the same)
        x, _ = self.attention(x, x, x)
        return x
        '''

    @classmethod
    def _create_group_class(cls, group_definition) -> str:
        """
        Generate Pytorch group block class definition from group definition

        Args:
            group_definition (Dict): Group definition used in Architecture
        
        Returns:
            str: A class definition for the Pytorch module to reuse in architecture.
        """
        block_definition = f'''
class {group_definition['name']}(nn.Module):
    """
    Group Block: {group_definition['name']}
    {group_definition.get('description', 'No description provided.')}
    """ 
        '''
        init_method = '''
    def __init__(self, **args):
        super().__init__()
        '''
        layers = []
        for node in group_definition.get('internal_structure', {}).get('nodes', []):
            node_type = cls.get_node_type(node)
            class_name = cls.node_type_to_class_name(node_type)
            layer_name = f"self.{node['id'].replace('-','_')}_{class_name}"
            layers.append(layer_name)
            params = node.get('data', {}).get('config', {})
            init_method += f"\n        {layer_name} = {class_name}({params})"
        block_definition += init_method + "\n"
        forward_method = '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the group block.
        Args:
            x: Input tensor
        Returns:
            Output tensor after processing through the group block
        """
        '''
        for layer in layers:
            forward_method += f"\n        x = {layer}(x)"
        forward_method += "\n        return x\n"
        block_definition += forward_method + "\n"
        return block_definition


    @classmethod 
    def _create_defintion(cls, node_type) -> str:
        """
        Generate Pytorch layer class definition from class type

        Args:
            node_type (str): Node type used in Architecture
        
        Returns:
            str: A class definition for the Pytorch module to reuse in architecture.
        """
        if node_type == 'add':
            return cls._create_add_class()
        elif node_type == 'concat':
            return cls._create_concat_class()
        elif node_type == 'conv2d':
            return cls._create_conv2d_class()
        elif node_type == 'maxpool':
            return cls._create_maxpool_class()
        elif node_type == 'flatten':
            return cls._create_flatten_class()
        elif node_type == 'linear':
            return cls._create_linear_class()
        elif node_type == 'batchnorm':
            return cls._create_batchnorm_class()
        elif node_type == 'relu':
            return cls._create_relu_class()
        elif node_type == 'softmax':
            return cls._create_softmax_class()
        elif node_type == 'dropout':
            return cls._create_dropout_class()
        elif node_type == 'attention':
            return cls._create_self_attention_block()
        elif node_type == 'custom':
            pass
    
    @classmethod
    def node_type_to_class_name(cls, node_type: str) -> str:
        """
        Map node type to corresponding class name.

        Args:
            node_type (str): Node type used in Architecture

        Returns:
            str: Corresponding class name
        """
        mapping = {
            'conv2d': 'Conv2DBlock',
            'add': 'AddBlock',
            'concat': 'ConcatBlock',
            'linear': 'LinearLayer',
            'maxpool': 'MaxPoolBlock',
            'flatten': 'FlattenLayer',
            'relu': 'ReLUBlock',
            'softmax': 'SoftmaxBlock',
            'dropout': 'DropoutBlock',
            'batchnorm': 'BatchNormBlock',
            'attention': 'MultiHeadSelfAttentionBlock'
        }
        return mapping.get(node_type, 'UnknownBlock')
    
    @classmethod
    def group_node_type_to_class_name(cls, definition_id: str) -> str:
        """
        Map group definition ID to corresponding class name.

        Args:
            definition_id (str): Group definition ID used in Architecture

        Returns:
            str: Corresponding class name
        """
        print(cls.group_class_mapper)
        return cls.group_class_mapper.get(definition_id, 'UnknownGroupBlock')
    
    @classmethod
    def create_required_node_classes(cls, nodes: List[Dict[str, Any]], group_definitions: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Forward declare the required node type classes for reuse in model definition.
        Allows to keep clean model file

        Returns:
            str: All generated unique node type class definitions
        """
        node_types_required = set()

        for node in nodes:
            node_type = cls.get_node_type(node)
            if node_type != 'group':
                node_types_required.add(node_type)
        for group in group_definitions:
            cls.group_class_mapper[group['id'].replace('-','_')] = group['name']
            group_nodes = group.get('internal_structure', {}).get('nodes', [])
            for node in group_nodes:
                node_types_required.add(cls.get_node_type(node))

        class_declarations = '''
#==========================
#Layer Definitions:
#==========================
        '''
        for n_type in node_types_required:
            defn = cls._create_defintion(n_type)
            class_declarations += f"\n{defn}"
        for group in group_definitions:
            group_defn = cls._create_group_class(group)
            class_declarations += f"\n{group_defn}"

        return class_declarations
        
        
class LayerInitializationGenerator():
    """
    Initilizes dedicated objects of the forward-declared classes with correct parameters to form
    architecture layers
    """

    @classmethod
    def generate_layer_initializations(cls, nodes: List[Dict[str, Any]], group_definitions: Optional[List[Dict[str,Any]]] = None) -> str:
        """
        Generate layer initializations for all nodes in the architecture.

        Args:
            nodes (List[Dict[str, Any]]): List of node definitions.
            edges (List[Dict[str, Any]]): List of edge definitions.
            group_definitions (List[Dict[str,Any]]): List of group definitions.

        Returns:
            str: Generated layer initializations code.
        """
        layer_initializations = '''
        #==========================
        #Layer Initializations:
        #==========================
        '''
        for node in nodes:
            node_type = ClassDefinitionGenerator.get_node_type(node)
            node_id = node['id'].replace('-', '_')

            params = {}
            if node_type in ('input', 'dataloader', 'output'):
                continue
            elif node_type == 'group':
                group_type = node['data'].get('groupDefinitionId', '').replace('-', '_')
                class_name = ClassDefinitionGenerator.group_node_type_to_class_name(group_type)
                layer_name = f"self.{node_id}_{class_name}"

                #get params
                group_definition_id = node['data'].get('groupDefinitionId')
                group_params = {}
                for group_def in group_definitions:
                    if group_def['id'] == group_definition_id:
                        for node in group_def.get('internal_structure').get('nodes'):
                            group_params.update(node.get('data').get('config'))
                    params = group_params
                    break

            else:
                # Get layer names
                class_name = ClassDefinitionGenerator.node_type_to_class_name(node_type)
                layer_name = f"self.{node_id}_{class_name}"
                params = node.get('data', {}).get('config', {})

            layer_initializations += f"\n        {layer_name} = {class_name}({params})"
        return layer_initializations


# ============================================
# Helper Functions
# ============================================

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


# ============================================
# Model File Generation Helper
# ============================================

def generate_model_file(
    project_name: str,
    layer_classes: str,
    model_definition: str
) -> str:
    """
    Generate the model.py file with just the model architecture.

    Args:
        nodes: List of node definitions
        edges: List of edge definitions
        project_name: Name of the project/model
        layer_classes: Generated layer class definitions
        model_definition: Generated model class definition

    Returns:
        str: Complete model.py file contents
    """
    file = f'''"""
Generated PyTorch Model
Architecture: {project_name}
Generated by VisionForge

This file contains the model architecture with separate layer classes.
Each layer is implemented as a reusable class for clarity and maintainability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
{layer_classes}

{model_definition}
'''
    return file


# ============================================
# Adaptive Helper Functions for Training, Dataset, and Config
# ============================================

def generate_training_script(
    project_name: str,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]]
) -> str:
    """
    Generate adaptive training script based on model architecture.

    Analyzes the architecture to determine:
    - Input/output types and shapes
    - Whether it's a classification or regression task
    - Appropriate loss function
    - Model complexity for learning rate selection

    Args:
        project_name: Name of the project/model
        nodes: List of node definitions
        edges: List of edge definitions

    Returns:
        str: Complete training script code
    """
    # Analyze architecture to determine task type
    has_softmax = any(ClassDefinitionGenerator.get_node_type(n) == 'softmax' for n in nodes)

    # Count total layers for complexity estimation
    layer_count = sum(1 for n in nodes if ClassDefinitionGenerator.get_node_type(n) not in ('input', 'output', 'dataloader'))

    # Determine appropriate loss function and metrics based on output layer
    if has_softmax:
        loss_function = "nn.CrossEntropyLoss()"
        metric_name = "accuracy"
        task_type = "classification"
    else:
        loss_function = "nn.MSELoss()"
        metric_name = "mse"
        task_type = "regression"

    # Generate conditional code sections based on task type
    if has_softmax:
        train_metric_calc = """if task_type == "classification":
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()"""
        train_metric_result = "metric = 100. * correct / total"

        val_metric_calc = """pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()"""
        val_metric_result = "metric = 100. * correct / total"
    else:
        train_metric_calc = "# For regression tasks, metric tracking can be added here"
        train_metric_result = "metric = avg_loss"
        val_metric_calc = "# Metric calculation for regression"
        val_metric_result = "metric = avg_loss"

    return f'''"""
Training Script for {project_name}
Generated by VisionForge
Architecture Type: {task_type.capitalize()}
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple, Dict
import time

from model import {project_name}
from dataset import CustomDataset
from config import *


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: The model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Tuple of (average loss, metric value)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate metric
        {train_metric_calc}
        total += target.size(0)

    avg_loss = total_loss / len(dataloader)
    {train_metric_result}

    return avg_loss, metric


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate for one epoch.

    Args:
        model: The model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Tuple of (average loss, metric value)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()

            # Calculate metric
            {val_metric_calc}
            total += target.size(0)

    avg_loss = total_loss / len(dataloader)
    {val_metric_result}

    return avg_loss, metric


def main():
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {{device}}")

    # Create model
    model = {project_name}().to(device)
    print(f"Model created with {{sum(p.numel() for p in model.parameters()):,}} parameters")

    # Create datasets
    train_dataset = CustomDataset(train=True)
    val_dataset = CustomDataset(train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Setup training
    criterion = {loss_function}
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss, train_metric = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_metric = validate_epoch(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        epoch_time = time.time() - start_time

        # Print progress
        print(f"Epoch {{epoch+1}}/{{NUM_EPOCHS}} | "
              f"Time: {{epoch_time:.2f}}s | "
              f"Train Loss: {{train_loss:.4f}} | Train {metric_name.upper()}: {{train_metric:.2f}} | "
              f"Val Loss: {{val_loss:.4f}} | Val {metric_name.upper()}: {{val_metric:.2f}}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  → New best model saved (val_loss: {{val_loss:.4f}})")

    print("\\nTraining completed!")
    print(f"Best validation loss: {{best_val_loss:.4f}}")


if __name__ == "__main__":
    main()
'''


def generate_dataset_class(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]]
) -> str:
    """
    Generate adaptive dataset class based on input architecture.

    Analyzes the input node to determine:
    - Input shape and dimensions
    - Whether data is image, sequence, or tabular
    - Appropriate data loading and preprocessing

    Args:
        nodes: List of node definitions
        edges: List of edge definitions

    Returns:
        str: Complete dataset class code
    """
    # Find input node
    input_node = next((n for n in nodes if ClassDefinitionGenerator.get_node_type(n) == 'input'), None)

    # Extract input shape
    input_shape = (1, 3, 224, 224)  # Default
    data_type = "image"

    if input_node:
        config = input_node.get('data', {}).get('config', {})
        shape_str = config.get('shape', '[1, 3, 224, 224]')
        try:
            import json
            shape = json.loads(shape_str) if isinstance(shape_str, str) else shape_str
            if isinstance(shape, list) and len(shape) >= 2:
                input_shape = tuple(shape)
                # Determine data type from shape
                if len(shape) == 4:
                    data_type = "image"
                    channels, height, width = shape[1], shape[2], shape[3]
                elif len(shape) == 2:
                    data_type = "tabular"
                    features = shape[1]
                elif len(shape) == 3:
                    data_type = "sequence"
        except:
            pass

    # Generate appropriate dataset class based on data type
    if data_type == "image":
        channels, height, width = input_shape[1], input_shape[2], input_shape[3]
        return f'''"""
Custom Dataset Class
Generated by VisionForge
Data Type: Image ({channels} channels, {height}x{width})
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    """
    Custom dataset for loading and preprocessing image data.

    Expected input: Images with shape [{channels}, {height}, {width}]
    Channels: {channels} ({'RGB' if channels == 3 else 'Grayscale' if channels == 1 else 'Multi-channel'})
    """

    def __init__(self, data_dir: str = './data', train: bool = True):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing the data
            train: Whether this is training data
        """
        self.data_dir = Path(data_dir)
        self.train = train

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(({height}, {width})),
            {'transforms.Grayscale(),' if channels == 1 else ''}
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5] * {channels},
                std=[0.5] * {channels}
            )
        ])

        # TODO: Load your data here
        # Example: self.images = list((self.data_dir / ('train' if train else 'val')).glob('*.jpg'))
        # For now, we'll use dummy data
        self.data = []
        self.labels = []

    def __len__(self) -> int:
        """Return the total number of samples"""
        return len(self.data) if self.data else 100  # Dummy size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image tensor, label tensor)
        """
        # TODO: Replace with actual data loading
        # Example:
        # image = Image.open(self.images[idx])
        # image = self.transform(image)
        # label = self.labels[idx]

        # Dummy data for now
        image = torch.randn({channels}, {height}, {width})
        label = torch.randint(0, 10, (1,)).item()

        return image, label
'''
    else:
        # Fallback for other data types
        return f'''"""
Custom Dataset Class
Generated by VisionForge
Data Type: {data_type.capitalize()}
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple


class CustomDataset(Dataset):
    """
    Custom dataset for loading and preprocessing data.

    Expected input shape: {input_shape}
    """

    def __init__(self, data_dir: str = './data', train: bool = True):
        """Initialize the dataset"""
        self.data_dir = Path(data_dir)
        self.train = train

        # TODO: Load your data here
        self.data = []
        self.labels = []

    def __len__(self) -> int:
        """Return the total number of samples"""
        return len(self.data) if self.data else 100

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample"""
        # TODO: Replace with actual data loading
        data = torch.randn(*{input_shape[1:]})
        label = torch.randint(0, 10, (1,)).item()

        return data, label
'''


def generate_config_file(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]]
) -> str:
    """
    Generate adaptive configuration file based on architecture.

    Analyzes the architecture to set:
    - Appropriate batch size based on model size
    - Learning rate based on complexity
    - Input shape from input node
    - Number of epochs based on layer count

    Args:
        nodes: List of node definitions
        edges: List of edge definitions

    Returns:
        str: Complete configuration file code
    """
    # Extract input shape
    input_node = next((n for n in nodes if ClassDefinitionGenerator.get_node_type(n) == 'input'), None)
    input_shape = [1, 3, 224, 224]  # Default

    if input_node:
        config = input_node.get('data', {}).get('config', {})
        shape_str = config.get('shape', '[1, 3, 224, 224]')
        try:
            import json
            shape = json.loads(shape_str) if isinstance(shape_str, str) else shape_str
            if isinstance(shape, list):
                input_shape = shape
        except:
            pass

    # Count layers to estimate model complexity
    layer_count = sum(1 for n in nodes if ClassDefinitionGenerator.get_node_type(n) not in ('input', 'output', 'dataloader'))

    # Adaptive hyperparameters based on complexity
    if layer_count > 20:
        batch_size = 16
        learning_rate = 1e-4
        epochs = 100
        comment_complexity = "Deep"
    elif layer_count > 10:
        batch_size = 32
        learning_rate = 1e-3
        epochs = 50
        comment_complexity = "Medium"
    else:
        batch_size = 64
        learning_rate = 1e-3
        epochs = 30
        comment_complexity = "Shallow"

    # Check if model has attention layers (might need lower LR)
    has_attention = any(ClassDefinitionGenerator.get_node_type(n) == 'self_attention' for n in nodes)
    if has_attention:
        learning_rate = learning_rate * 0.1
        batch_size = max(8, batch_size // 2)

    return f'''"""
Configuration File
Generated by VisionForge
Architecture Complexity: {comment_complexity} ({layer_count} layers)
"""

# Training Configuration
BATCH_SIZE = {batch_size}  # Adjusted for {comment_complexity.lower()} network
LEARNING_RATE = {learning_rate}  # {'Reduced for attention layers' if has_attention else 'Standard for architecture'}
NUM_EPOCHS = {epochs}
WEIGHT_DECAY = 1e-4

# Model Configuration (NCHW format: batch, channels, height, width)
INPUT_SHAPE = {input_shape}

# Data Configuration
DATA_DIR = './data'
NUM_WORKERS = 0  # Set to 0 for debugging, increase for faster data loading

# Device Configuration
DEVICE = 'cuda'  # Change to 'cpu' if no GPU available

# Logging Configuration
LOG_INTERVAL = 10  # Print every N batches
SAVE_INTERVAL = 5  # Save checkpoint every N epochs

# Architecture Info
NUM_LAYERS = {layer_count}
HAS_ATTENTION = {has_attention}
'''


# ===========================================
# CodeGen Driver
# ===========================================

def generate_pytorch_code(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    project_name: str = "GeneratedModel",
    group_definitions: Optional[List[Dict[str, Any]]] = None
) -> Tuple[Dict[str, str], List[Exception]]:
    """
    Generate PyTorch code from the given nodes and edges.

    Args:
        nodes (List[Dict[str, Any]]): List of node definitions.
        edges (List[Dict[str, Any]]): List of edge definitions.
        project_name (str): Name of the project/model.
        group_definitions (Optional[List[Dict[str, Any]]]): List of group definitions.

    Returns:
        Tuple[Dict[str, str], List[Exception]]: A tuple containing a dictionary of generated
        code files (with keys: 'model', 'train', 'dataset', 'config') and a list of shape
        errors encountered during generation.
    """
    # Generate layer class definitions
    layer_classes = ClassDefinitionGenerator.create_required_node_classes(nodes, group_definitions)

    # Generate layer initializations
    layer_initializations = LayerInitializationGenerator.generate_layer_initializations(nodes, group_definitions)

    # Start model definition
    model_definition = f'''
class {project_name}(nn.Module):
    """
    PyTorch Model for {project_name}

    This model is auto-generated from the VisionForge architecture.
    """

    def __init__(self):
        super({project_name}, self).__init__()
{layer_initializations}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor after processing through the model
        """
'''

    # Sort nodes topologically based on edges
    sorted_nodes = topological_sort(nodes, edges)

    # Build edge map to track incoming connections
    edge_map = {}
    for edge in edges:
        target = edge.get('target')
        source = edge.get('source')
        if target not in edge_map:
            edge_map[target] = []
        edge_map[target].append(source)

    # Generate forward pass with variable tracking
    var_map = {}  # Maps node_id -> variable_name

    for node in sorted_nodes:
        node_id = node['id'].replace('-', '_')
        node_type = ClassDefinitionGenerator.get_node_type(node)

        print(f"Processing node: {node_id} of type {node_type}")

        # Skip input/output nodes
        if node_type in ('input', 'dataloader', 'output'):
            var_map[node_id] = 'x'
            continue
        elif node_type == 'group':
            group_type = node['data'].get('groupDefinitionId', '').replace('-', '_')
            class_name = ClassDefinitionGenerator.group_node_type_to_class_name(group_type)
            layer_name = f"self.{node_id}_{class_name}"
        else:
            # Get layer names
            class_name = ClassDefinitionGenerator.node_type_to_class_name(node_type)
            layer_name = f"self.{node_id}_{class_name}"

        # Determine input variable(s)
        incoming = edge_map.get(node_id, [])
        input_var = get_input_variable(incoming, var_map)

        # Generate forward pass line
        output_var = 'x'
        model_definition += f"\n        {output_var} = {layer_name}({input_var})"
        var_map[node_id] = output_var

    model_definition += "\n        return x\n"

    # Generate all file components
    model_code = generate_model_file(
        project_name=project_name,
        layer_classes=layer_classes,
        model_definition=model_definition
    )

    # Generate adaptive training, dataset, and config files
    train_code = generate_training_script(project_name, nodes, edges)
    dataset_code = generate_dataset_class(nodes, edges)
    config_code = generate_config_file(nodes, edges)

    # Return in the expected format with 4 file keys
    return {
        'model': model_code,
        'train': train_code,
        'dataset': dataset_code,
        'config': config_code
    }, []

