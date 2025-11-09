"""Validation and shape computation rules for nodes."""

from .shape import (
    TensorShape,
    compute_activation_output,
    compute_add_output,
    compute_batchnorm_output,
    compute_concat_output,
    compute_conv2d_output,
    compute_dropout_output,
    compute_flatten_output,
    compute_linear_output,
    compute_maxpool_output,
)
from .validation import (
    ValidationError,
    validate_config,
    validate_connection,
    validate_graph_acyclic,
    validate_multi_input_connection,
    validate_single_input_node,
)

__all__ = [
    # Shape functions
    "TensorShape",
    "compute_activation_output",
    "compute_add_output",
    "compute_batchnorm_output",
    "compute_concat_output",
    "compute_conv2d_output",
    "compute_dropout_output",
    "compute_flatten_output",
    "compute_linear_output",
    "compute_maxpool_output",
    # Validation functions
    "ValidationError",
    "validate_config",
    "validate_connection",
    "validate_graph_acyclic",
    "validate_multi_input_connection",
    "validate_single_input_node",
]
