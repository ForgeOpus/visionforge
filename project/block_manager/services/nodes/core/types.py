"""
Type definitions for the node system.
Contains all data structures used across nodes, registry, and code generation.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


class Framework(str, Enum):
    """Supported ML frameworks"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


@dataclass
class TensorShape:
    """
    Represents tensor shape with dimensions and description.

    Attributes:
        dims: List of dimension sizes (e.g., [batch, channels, height, width])
        description: Human-readable description of the shape meaning
    """
    dims: List[int]
    description: str = ""

    def __repr__(self) -> str:
        return f"TensorShape(dims={self.dims}, description='{self.description}')"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dims": self.dims,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TensorShape':
        return cls(
            dims=data.get("dims", []),
            description=data.get("description", "")
        )

    def matches(self, other: 'TensorShape') -> bool:
        """Check if two shapes match exactly"""
        return self.dims == other.dims

    @property
    def ndim(self) -> int:
        """Number of dimensions"""
        return len(self.dims)

    @property
    def batch_size(self) -> Optional[int]:
        """Get batch size (first dimension)"""
        return self.dims[0] if self.dims else None

    @property
    def channels(self) -> Optional[int]:
        """Get channels (second dimension for NCHW format)"""
        return self.dims[1] if len(self.dims) >= 2 else None

    @property
    def features(self) -> Optional[int]:
        """Get features (second dimension for 2D tensors)"""
        return self.dims[1] if len(self.dims) == 2 else None


@dataclass
class ConfigOption:
    """
    Option for a select/dropdown configuration field.

    Attributes:
        value: The actual value to use in code
        label: Human-readable display label
    """
    value: Any
    label: str


@dataclass
class ConfigField:
    """
    Configuration field definition for a node.

    Attributes:
        name: Field identifier (used in code)
        label: Human-readable display name
        type: Field type ('number', 'text', 'boolean', 'select')
        required: Whether the field must be set
        default: Default value if not provided
        min: Minimum value (for number fields)
        max: Maximum value (for number fields)
        options: Available options (for select fields)
        description: Help text for the field
        placeholder: Placeholder text for input
    """
    name: str
    label: str
    type: str
    required: bool = False
    default: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    options: List[ConfigOption] = field(default_factory=list)
    description: str = ""
    placeholder: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API serialization"""
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
            data["options"] = [
                {"value": opt.value, "label": opt.label}
                for opt in self.options
            ]
        if self.placeholder:
            data["placeholder"] = self.placeholder
        return data


@dataclass
class InputPort:
    """
    Named input port for nodes with multiple inputs.

    Used by nodes like Loss that need distinct inputs
    (e.g., predictions vs ground truth).

    Attributes:
        id: Port identifier
        label: Human-readable name
        description: Help text
    """
    id: str
    label: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description
        }


@dataclass
class NodeMetadata:
    """
    Metadata describing a node's visual and identification properties.

    Attributes:
        type: Unique identifier for this node type
        label: Display name shown in UI
        category: Category for grouping in palette
        color: CSS color for node appearance
        icon: Icon name from icon library
        description: Brief description of functionality
        framework: Which ML framework this node belongs to
    """
    type: str
    label: str
    category: str
    color: str
    icon: str
    description: str
    framework: Framework

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
