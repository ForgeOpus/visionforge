"""
Base classes for node definitions.
Provides abstract interfaces and shared functionality for all node types.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .types import (
    ConfigField,
    Framework,
    InputPort,
    NodeMetadata,
    TensorShape,
)


class ShapeComputerMixin:
    """Mixin providing shape computation utilities for common operations"""

    def compute_conv2d_output(
        self,
        input_height: int,
        input_width: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int = 1
    ) -> Tuple[int, int]:
        """Compute 2D convolution output dimensions"""
        effective_kernel = dilation * (kernel_size - 1) + 1
        output_height = (input_height + 2 * padding - effective_kernel) // stride + 1
        output_width = (input_width + 2 * padding - effective_kernel) // stride + 1
        return (output_height, output_width)

    def compute_pool2d_output(
        self,
        input_height: int,
        input_width: int,
        kernel_size: int,
        stride: int,
        padding: int
    ) -> Tuple[int, int]:
        """Compute 2D pooling output dimensions"""
        output_height = (input_height + 2 * padding - kernel_size) // stride + 1
        output_width = (input_width + 2 * padding - kernel_size) // stride + 1
        return (output_height, output_width)

    def compute_conv1d_output(
        self,
        input_length: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int = 1
    ) -> int:
        """Compute 1D convolution output length"""
        effective_kernel = dilation * (kernel_size - 1) + 1
        return (input_length + 2 * padding - effective_kernel) // stride + 1

    def parse_shape_string(self, shape_str: str) -> Optional[List[int]]:
        """Parse shape from JSON string like '[1, 3, 224, 224]'"""
        import json
        try:
            parsed = json.loads(shape_str)
            if isinstance(parsed, list) and all(isinstance(d, int) and d > 0 for d in parsed):
                return parsed
        except (json.JSONDecodeError, ValueError):
            return None
        return None


class ValidatorMixin:
    """Mixin providing validation utilities for connections and shapes"""

    def validate_dimensions(
        self,
        shape: Optional[TensorShape],
        required_dims: Any,  # int, list of ints, or 'any'
        description: str = ""
    ) -> Optional[str]:
        """
        Validate input tensor dimensions against requirements.

        Args:
            shape: Input tensor shape to validate
            required_dims: Required number of dimensions (int, list, or 'any')
            description: Human-readable shape format (e.g., '[batch, features]')

        Returns:
            Error message if invalid, None if valid
        """
        if not shape:
            return "Input shape is not defined"

        actual_dims = len(shape.dims)

        if required_dims == 'any':
            return None

        if isinstance(required_dims, int):
            if actual_dims != required_dims:
                return f"Requires {required_dims}D input {description}, got {actual_dims}D"
        elif isinstance(required_dims, list):
            if actual_dims not in required_dims:
                dims_str = ' or '.join(map(str, required_dims))
                return f"Requires {dims_str}D input {description}, got {actual_dims}D"

        return None

    def shapes_match(self, shape1: TensorShape, shape2: TensorShape) -> bool:
        """Check if all dimensions in shapes match"""
        return shape1.matches(shape2)


class NodeDefinition(ABC, ShapeComputerMixin, ValidatorMixin):
    """
    Abstract base class for all node definitions.

    Defines the interface that all nodes must implement, including:
    - Metadata (type, label, category, etc.)
    - Configuration schema
    - Shape computation
    - Connection validation
    - Code generation template

    Subclasses must implement:
    - metadata property
    - config_schema property
    - compute_output_shape method
    - template_name property (for code generation)

    Example:
        class LinearNode(NodeDefinition):
            @property
            def metadata(self) -> NodeMetadata:
                return NodeMetadata(
                    type="linear",
                    label="Linear",
                    ...
                )

            @property
            def config_schema(self) -> List[ConfigField]:
                return [
                    ConfigField(name="out_features", ...)
                ]

            def compute_output_shape(self, input_shape, config):
                ...

            @property
            def template_name(self) -> str:
                return "linear"
    """

    @property
    @abstractmethod
    def metadata(self) -> NodeMetadata:
        """Return node metadata"""
        pass

    @property
    @abstractmethod
    def config_schema(self) -> List[ConfigField]:
        """Return configuration schema"""
        pass

    @abstractmethod
    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        """
        Compute output shape given input shape and configuration.

        Args:
            input_shape: Shape of input tensor (None if not connected)
            config: Node configuration dictionary

        Returns:
            Output tensor shape, or None if cannot be computed
        """
        pass

    @property
    def template_name(self) -> str:
        """
        Name of the Jinja2 template file (without extension).

        By default, uses the node type from metadata.
        Template file should be located at:
        block_manager/services/nodes/frameworks/{framework}/templates/{template_name}.jinja2

        Override this if the template name differs from node type.
        """
        return self.metadata.type

    @property
    def input_ports(self) -> Optional[List[InputPort]]:
        """
        Named input ports for nodes with multiple distinct inputs.

        Most nodes have a single unnamed input, so this returns None by default.
        Override for nodes like Loss that need named ports.
        """
        return None

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        """
        Validate whether this node can receive a connection from a source node.

        Args:
            source_node_type: Type of the source node
            source_output_shape: Output shape of the source node
            target_config: Configuration of this (target) node

        Returns:
            Error message if invalid, None if valid
        """
        return None  # Default: allow all connections

    def allows_multiple_inputs(self) -> bool:
        """Check if this node type allows multiple input connections"""
        return False  # Default: single input only

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate node configuration parameters.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of error messages, empty if valid
        """
        errors = []

        for field in self.config_schema:
            value = config.get(field.name)

            # Check required fields
            if field.required and (value is None or value == ''):
                errors.append(f"{field.label} is required")

            # Validate numeric ranges
            if field.type == 'number' and value is not None:
                try:
                    num_val = float(value)
                    if field.min is not None and num_val < field.min:
                        errors.append(f"{field.label} must be at least {field.min}")
                    if field.max is not None and num_val > field.max:
                        errors.append(f"{field.label} must be at most {field.max}")
                except (TypeError, ValueError):
                    errors.append(f"{field.label} must be a number")

        return errors

    def get_default_config(self) -> Dict[str, Any]:
        """Generate default configuration from schema"""
        config = {}
        for field in self.config_schema:
            if field.default is not None:
                config[field.name] = field.default
        return config

    def get_template_context(
        self,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape] = None
    ) -> Dict[str, Any]:
        """
        Build context dictionary for template rendering.

        Args:
            config: Node configuration
            input_shape: Input tensor shape (for computing derived values)

        Returns:
            Context dictionary with config, computed values, and metadata
        """
        # Merge defaults with provided config
        final_config = {**self.get_default_config(), **config}

        # Build base context
        context = {
            "config": final_config,
            "metadata": self.metadata.to_dict(),
        }

        # Add input shape info if available
        if input_shape:
            context["input"] = {
                "dims": input_shape.dims,
                "ndim": len(input_shape.dims),
            }
            # Add common derived values
            if len(input_shape.dims) >= 2:
                context["input"]["batch_size"] = input_shape.dims[0]
                context["input"]["features"] = input_shape.dims[1]
            if len(input_shape.dims) >= 4:
                context["input"]["channels"] = input_shape.dims[1]
                context["input"]["height"] = input_shape.dims[2]
                context["input"]["width"] = input_shape.dims[3]

        return context

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node definition to dictionary for API"""
        result = {
            "metadata": self.metadata.to_dict(),
            "configSchema": [field.to_dict() for field in self.config_schema],
            "allowsMultipleInputs": self.allows_multiple_inputs(),
        }
        if self.input_ports:
            result["inputPorts"] = [port.to_dict() for port in self.input_ports]
        return result


class SourceNodeDefinition(NodeDefinition):
    """
    Base class for input/source nodes that don't receive connections.

    Examples: Input, DataLoader
    """

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        return f"{self.metadata.label} blocks cannot receive connections (they are source nodes)"


class TerminalNodeDefinition(NodeDefinition):
    """
    Base class for output/terminal nodes that accept any input.

    Examples: Output, Loss
    """

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        return None  # Always valid


class MergeNodeDefinition(NodeDefinition):
    """
    Base class for merge nodes that accept multiple inputs.

    Examples: Concat, Add
    """

    def allows_multiple_inputs(self) -> bool:
        return True


class PassthroughNodeDefinition(NodeDefinition):
    """
    Base class for nodes that don't change tensor shape.

    Examples: ReLU, Dropout, BatchNorm (mostly)
    """

    def compute_output_shape(
        self,
        input_shape: Optional[TensorShape],
        config: Dict[str, Any]
    ) -> Optional[TensorShape]:
        return input_shape

    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        return None  # Always valid
