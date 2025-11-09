"""
Base classes for node definitions in the backend.
Provides abstract interfaces and shared functionality for all node types.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class Framework(str, Enum):
    """Supported backend frameworks"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


class TensorShape:
    """Represents tensor shape with dimensions and description"""
    
    def __init__(self, dims: List[int], description: str = ""):
        self.dims = dims
        self.description = description
    
    def __repr__(self) -> str:
        return f"TensorShape(dims={self.dims}, description='{self.description}')"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dims": self.dims,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TensorShape':
        return cls(dims=data.get("dims", []), description=data.get("description", ""))
    
    def matches(self, other: 'TensorShape') -> bool:
        """Check if two shapes match exactly"""
        return self.dims == other.dims


class NodeMetadata:
    """Metadata describing a node's properties"""
    
    def __init__(
        self,
        type: str,
        label: str,
        category: str,
        color: str,
        icon: str,
        description: str,
        framework: Framework
    ):
        self.type = type
        self.label = label
        self.category = category
        self.color = color
        self.icon = icon
        self.description = description
        self.framework = framework
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "label": self.label,
            "category": self.category,
            "color": self.color,
            "icon": self.icon,
            "description": self.description,
            "framework": self.framework.value
        }


class ConfigField:
    """Configuration field definition"""
    
    def __init__(
        self,
        name: str,
        label: str,
        type: str,
        required: bool = False,
        default: Any = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
        options: Optional[List[Dict[str, Any]]] = None,
        description: str = "",
        placeholder: str = ""
    ):
        self.name = name
        self.label = label
        self.type = type
        self.required = required
        self.default = default
        self.min = min
        self.max = max
        self.options = options or []
        self.description = description
        self.placeholder = placeholder
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "label": self.label,
            "type": self.type,
            "description": self.description
        }
        if self.required:
            data["required"] = self.required
        if self.default is not None:
            data["default"] = self.default
        if self.min is not None:
            data["min"] = self.min
        if self.max is not None:
            data["max"] = self.max
        if self.options:
            data["options"] = self.options
        if self.placeholder:
            data["placeholder"] = self.placeholder
        return data


class ShapeComputerMixin:
    """Mixin providing shape computation utilities"""
    
    def compute_conv2d_output(
        self,
        input_height: int,
        input_width: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int
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
    
    def parse_shape_string(self, shape_str: str) -> Optional[List[int]]:
        """Parse shape from JSON string"""
        import json
        try:
            parsed = json.loads(shape_str)
            if isinstance(parsed, list) and all(isinstance(d, int) and d > 0 for d in parsed):
                return parsed
        except (json.JSONDecodeError, ValueError):
            return None
        return None


class ValidatorMixin:
    """Mixin providing validation utilities"""
    
    def validate_dimensions(
        self,
        shape: Optional[TensorShape],
        required_dims: Any,  # int, list of ints, or 'any'
        description: str = ""
    ) -> Optional[str]:
        """Validate input tensor dimensions against requirements"""
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
    Defines the interface that all nodes must implement.
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
        """Compute output shape given input shape and configuration"""
        pass
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        """
        Validate whether this node can receive a connection from a source node.
        Returns error message if invalid, None if valid.
        """
        return None  # Default: allow all connections
    
    def allows_multiple_inputs(self) -> bool:
        """Check if this node type allows multiple input connections"""
        return False  # Default: single input only
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate node configuration parameters.
        Returns list of error messages, empty if valid.
        """
        errors = []
        
        for field in self.config_schema:
            value = config.get(field.name)
            
            # Check required fields
            if field.required and (value is None or value == ''):
                errors.append(f"{field.label} is required")
            
            # Validate numeric ranges
            if field.type == 'number' and value is not None:
                if field.min is not None and value < field.min:
                    errors.append(f"{field.label} must be at least {field.min}")
                if field.max is not None and value > field.max:
                    errors.append(f"{field.label} must be at most {field.max}")
        
        return errors
    
    def get_default_config(self) -> Dict[str, Any]:
        """Generate default configuration from schema"""
        config = {}
        for field in self.config_schema:
            if field.default is not None:
                config[field.name] = field.default
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize node definition to dictionary"""
        return {
            "metadata": self.metadata.to_dict(),
            "configSchema": [field.to_dict() for field in self.config_schema]
        }


class SourceNodeDefinition(NodeDefinition):
    """Base class for input/source nodes that don't receive connections"""
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        return f"{self.metadata.label} blocks cannot receive connections (they are source nodes)"


class TerminalNodeDefinition(NodeDefinition):
    """Base class for output/terminal nodes that accept any input"""
    
    def validate_incoming_connection(
        self,
        source_node_type: str,
        source_output_shape: Optional[TensorShape],
        target_config: Dict[str, Any]
    ) -> Optional[str]:
        return None  # Always valid


class MergeNodeDefinition(NodeDefinition):
    """Base class for merge nodes that accept multiple inputs"""
    
    def allows_multiple_inputs(self) -> bool:
        return True


class PassthroughNodeDefinition(NodeDefinition):
    """Base class for passthrough/utility nodes"""
    
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
