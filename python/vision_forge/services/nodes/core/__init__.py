"""
Core node system components.
Provides base classes, types, and utilities for the node definition system.
"""

from .types import (
    Framework,
    TensorShape,
    NodeMetadata,
    ConfigField,
    ConfigOption,
    InputPort,
)

from .base import (
    NodeDefinition,
    SourceNodeDefinition,
    TerminalNodeDefinition,
    MergeNodeDefinition,
    PassthroughNodeDefinition,
)

from .registry import (
    NodeRegistry,
    get_node_definition,
    get_all_node_definitions,
    get_node_definitions_by_category,
    has_node_definition,
    get_available_node_types,
)

from .codegen import (
    TemplateRenderer,
    render_node_template,
)

__all__ = [
    # Types
    'Framework',
    'TensorShape',
    'NodeMetadata',
    'ConfigField',
    'ConfigOption',
    'InputPort',
    # Base classes
    'NodeDefinition',
    'SourceNodeDefinition',
    'TerminalNodeDefinition',
    'MergeNodeDefinition',
    'PassthroughNodeDefinition',
    # Registry
    'NodeRegistry',
    'get_node_definition',
    'get_all_node_definitions',
    'get_node_definitions_by_category',
    'has_node_definition',
    'get_available_node_types',
    # Code generation
    'TemplateRenderer',
    'render_node_template',
]
