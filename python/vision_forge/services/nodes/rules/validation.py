"""Validation utilities for node connections and configurations."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..core import Framework
from ..core.base import NodeDefinition
from .shape import TensorShape


class ValidationError(Exception):
    """Raised when validation fails."""

    pass


def validate_connection(
    source_spec: NodeDefinition,
    target_spec: NodeDefinition,
    source_output_shape: Optional[TensorShape],
) -> tuple[bool, Optional[str]]:
    """
    Validate if a connection between two nodes is valid.

    Returns:
        (is_valid, error_message) - error_message is None if valid
    """
    # Input nodes can connect to anything
    if source_spec.metadata.type == "input":
        return True, None

    # Output nodes can be connected from anything
    if target_spec.metadata.type == "output":
        return True, None

    # Check framework compatibility
    if source_spec.metadata.framework != target_spec.metadata.framework:
        return (
            False,
            f"Framework mismatch: {source_spec.metadata.framework.value} → {target_spec.metadata.framework.value}",
        )

    # Shape-based validation
    if source_output_shape:
        dims = source_output_shape.get("dims", [])
        ndim = len(dims)

        # Conv2D requires 4D input
        if target_spec.metadata.type == "conv2d" and ndim != 4:
            return (
                False,
                f"Conv2D requires 4D input, got {ndim}D. Consider adding Flatten if coming from 2D layer.",
            )

        # Linear/Dense requires 2D input
        if target_spec.metadata.type == "linear" and ndim != 2:
            if ndim == 4:
                return (
                    False,
                    f"Linear layer requires 2D input, got {ndim}D. Add Flatten layer before Linear.",
                )
            return False, f"Linear layer requires 2D input, got {ndim}D."

        # MaxPool requires 4D input
        if target_spec.metadata.type == "maxpool" and ndim != 4:
            return False, f"MaxPool requires 4D input, got {ndim}D."

        # BatchNorm dimension requirements
        if target_spec.metadata.type == "batchnorm":
            if source_spec.metadata.framework == Framework.PYTORCH and ndim not in (2, 3, 4):
                return False, f"BatchNorm requires 2D, 3D, or 4D input, got {ndim}D."
            if source_spec.metadata.framework == Framework.TENSORFLOW and ndim < 2:
                return False, f"BatchNorm requires at least 2D input, got {ndim}D."

    return True, None


def validate_multi_input_connection(
    input_shapes: list[Optional[TensorShape]],
    target_spec: NodeDefinition,
) -> tuple[bool, Optional[str]]:
    """
    Validate connections for multi-input nodes (concat, add).

    Returns:
        (is_valid, error_message) - error_message is None if valid
    """
    if not target_spec.allows_multiple_inputs():
        return False, f"{target_spec.metadata.label} does not support multiple inputs"

    valid_shapes = [s for s in input_shapes if s is not None]
    
    if len(valid_shapes) < 2:
        return False, "Multi-input nodes require at least 2 inputs"

    # For Add nodes, all shapes must match exactly
    if target_spec.metadata.type == "add":
        first_dims = valid_shapes[0].get("dims", [])
        for i, shape in enumerate(valid_shapes[1:], 1):
            dims = shape.get("dims", [])
            if dims != first_dims:
                return (
                    False,
                    f"Add requires identical shapes. Input 0: {first_dims}, Input {i}: {dims}",
                )
        return True, None

    # For Concat nodes, all dimensions except concat axis must match
    if target_spec.metadata.type == "concat":
        first_dims = valid_shapes[0].get("dims", [])
        ndim = len(first_dims)

        for i, shape in enumerate(valid_shapes[1:], 1):
            dims = shape.get("dims", [])
            if len(dims) != ndim:
                return (
                    False,
                    f"Concat requires same number of dimensions. Input 0: {ndim}D, Input {i}: {len(dims)}D",
                )

        return True, None

    return True, None


def validate_config(
    node_spec: NodeDefinition,
    config: Dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Validate node configuration against its schema.

    Returns:
        (is_valid, error_messages) - empty list if valid
    """
    errors = []

    for field_spec in node_spec.config_schema:
        field_name = field_spec.name
        value = config.get(field_name)

        # Check required fields
        if field_spec.required and (value is None or value == ""):
            errors.append(f"Field '{field_spec.label}' is required")
            continue

        # Skip validation for None values on non-required fields
        if value is None:
            continue

        # Type-specific validation
        if field_spec.type == "number":
            try:
                num_value = float(value)

                # Check min/max
                if field_spec.min is not None and num_value < field_spec.min:
                    errors.append(
                        f"'{field_spec.label}' must be at least {field_spec.min}"
                    )
                if field_spec.max is not None and num_value > field_spec.max:
                    errors.append(
                        f"'{field_spec.label}' must be at most {field_spec.max}"
                    )
            except (ValueError, TypeError):
                errors.append(f"'{field_spec.label}' must be a number")

        elif field_spec.type == "boolean":
            if not isinstance(value, bool):
                errors.append(f"'{field_spec.label}' must be true or false")

        elif field_spec.type == "select":
            if field_spec.options:
                valid_values = [opt.value for opt in field_spec.options]
                if value not in valid_values:
                    errors.append(
                        f"'{field_spec.label}' must be one of: {', '.join(str(v) for v in valid_values)}"
                    )

    return len(errors) == 0, errors


def validate_graph_acyclic(
    edges: list[tuple[str, str]],
) -> tuple[bool, Optional[str]]:
    """
    Check if graph is acyclic (DAG).

    Args:
        edges: List of (source_id, target_id) tuples

    Returns:
        (is_valid, error_message) - error_message is None if valid
    """
    # Build adjacency list
    graph: Dict[str, list[str]] = {}
    all_nodes = set()
    
    for source, target in edges:
        all_nodes.add(source)
        all_nodes.add(target)
        if source not in graph:
            graph[source] = []
        graph[source].append(target)

    # DFS cycle detection
    visited = set()
    rec_stack = set()

    def has_cycle(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if has_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    for node in all_nodes:
        if node not in visited:
            if has_cycle(node):
                return False, "Graph contains a cycle. Neural networks must be acyclic."

    return True, None


def validate_single_input_node(node_id: str, edges: list[tuple[str, str]]) -> tuple[bool, Optional[str]]:
    """
    Validate that a non-multi-input node has at most one input connection.

    Args:
        node_id: ID of the node to check
        edges: List of (source_id, target_id) tuples

    Returns:
        (is_valid, error_message) - error_message is None if valid
    """
    input_count = sum(1 for _, target in edges if target == node_id)
    
    if input_count > 1:
        return False, f"Node can only accept one input, but has {input_count} connections"
    
    return True, None
