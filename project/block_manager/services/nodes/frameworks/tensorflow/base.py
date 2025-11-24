"""
TensorFlow/Keras-specific base class for node definitions.
Provides TensorFlow-specific utilities and conventions.
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


class TensorFlowNodeDefinition(NodeDefinition):
    """
    Base class for TensorFlow/Keras node definitions.

    Provides TensorFlow-specific conventions and helper methods.
    All TensorFlow nodes should inherit from this class.

    Note: TensorFlow uses channels-last format by default (NHWC)
    while PyTorch uses channels-first (NCHW).
    """

    def get_template_context(
        self,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape] = None
    ) -> Dict[str, Any]:
        """
        Build TensorFlow-specific template context.

        Adds common Keras patterns like units/filters
        derived from input shape.
        """
        context = super().get_template_context(config, input_shape)

        # Add TensorFlow-specific derived values
        if input_shape:
            dims = input_shape.dims

            # For 2D tensors (batch, features) - Dense layers
            if len(dims) == 2:
                context["input_dim"] = dims[1]

            # For 4D tensors - TensorFlow uses NHWC (batch, height, width, channels)
            elif len(dims) == 4:
                # Assume input is in NHWC format
                context["input_height"] = dims[1]
                context["input_width"] = dims[2]
                context["input_channels"] = dims[3]

            # For 3D tensors (batch, seq, features) - RNN layers
            elif len(dims) == 3:
                context["input_dim"] = dims[2]
                context["seq_length"] = dims[1]

        return context


class TensorFlowSourceNode(TensorFlowNodeDefinition, SourceNodeDefinition):
    """Base for TensorFlow source nodes (Input, DataLoader)"""
    pass


class TensorFlowTerminalNode(TensorFlowNodeDefinition, TerminalNodeDefinition):
    """Base for TensorFlow terminal nodes (Output, Loss)"""
    pass


class TensorFlowMergeNode(TensorFlowNodeDefinition, MergeNodeDefinition):
    """Base for TensorFlow merge nodes (Concatenate, Add)"""
    pass


class TensorFlowPassthroughNode(TensorFlowNodeDefinition, PassthroughNodeDefinition):
    """Base for TensorFlow passthrough nodes (ReLU, Dropout)"""
    pass
