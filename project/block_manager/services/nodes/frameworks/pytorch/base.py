"""
PyTorch-specific base class for node definitions.
Provides PyTorch-specific utilities and conventions.
"""

from typing import Any, Dict, List, Optional

from ...core.base import (
    NodeDefinition,
    SourceNodeDefinition,
    TerminalNodeDefinition,
    MergeNodeDefinition,
    PassthroughNodeDefinition,
)
from ...core.types import Framework, TensorShape


class PyTorchNodeDefinition(NodeDefinition):
    """
    Base class for PyTorch node definitions.

    Provides PyTorch-specific conventions and helper methods.
    All PyTorch nodes should inherit from this class.

    The framework is automatically set to PYTORCH.
    """

    def get_template_context(
        self,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape] = None
    ) -> Dict[str, Any]:
        """
        Build PyTorch-specific template context.

        Adds common PyTorch patterns like in_features/in_channels
        derived from input shape.
        """
        context = super().get_template_context(config, input_shape)

        # Add PyTorch-specific derived values
        if input_shape:
            dims = input_shape.dims

            # For 2D tensors (batch, features) - Linear layers
            if len(dims) == 2:
                context["in_features"] = dims[1]

            # For 4D tensors (batch, channels, height, width) - Conv layers
            elif len(dims) == 4:
                context["in_channels"] = dims[1]
                context["in_height"] = dims[2]
                context["in_width"] = dims[3]
                context["num_features"] = dims[1]  # For BatchNorm

            # For 3D tensors (batch, seq, features) - RNN layers
            elif len(dims) == 3:
                context["in_features"] = dims[2]
                context["seq_length"] = dims[1]

        return context


class PyTorchSourceNode(PyTorchNodeDefinition, SourceNodeDefinition):
    """Base for PyTorch source nodes (Input, DataLoader)"""
    pass


class PyTorchTerminalNode(PyTorchNodeDefinition, TerminalNodeDefinition):
    """Base for PyTorch terminal nodes (Output, Loss)"""
    pass


class PyTorchMergeNode(PyTorchNodeDefinition, MergeNodeDefinition):
    """Base for PyTorch merge nodes (Concat, Add)"""
    pass


class PyTorchPassthroughNode(PyTorchNodeDefinition, PassthroughNodeDefinition):
    """Base for PyTorch passthrough nodes (ReLU, Dropout)"""
    pass
