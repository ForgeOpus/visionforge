from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class Framework(str, Enum):
    """Supported backend frameworks."""

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


@dataclass(frozen=True)
class ConfigOptionSpec:
    """Selectable option for a configuration field."""

    value: Any
    label: str


@dataclass(frozen=True)
class ConfigFieldSpec:
    """Schema definition for a single configuration field."""

    name: str
    label: str
    field_type: str
    required: bool = False
    default: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    description: str = ""
    placeholder: str = ""
    options: Tuple[ConfigOptionSpec, ...] = field(default_factory=tuple)
    accept: Optional[str] = None


@dataclass(frozen=True)
class NodeTemplateSpec:
    """Rendering template configuration for a node."""

    name: str
    engine: str
    content: str
    default_context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeSpec:
    """Complete specification for a node definition."""

    type: str
    label: str
    category: str
    color: str
    icon: str
    description: str
    framework: Framework
    config_schema: Tuple[ConfigFieldSpec, ...]
    allows_multiple_inputs: bool = False
    shape_fn: Optional[str] = None
    validation_fn: Optional[str] = None
    template: Optional[NodeTemplateSpec] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def default_config(self) -> Dict[str, Any]:
        """Compute default configuration dictionary for the node."""

        defaults: Dict[str, Any] = {}
        for field_spec in self.config_schema:
            if field_spec.default is not None:
                defaults[field_spec.name] = field_spec.default
        return defaults
