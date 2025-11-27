"""
Node definitions package for VisionForge backend.
Provides modular, extensible node definitions for all supported frameworks.
"""

from .base import (
    Framework,
    TensorShape,
    NodeMetadata,
    ConfigField,
    NodeDefinition,
    SourceNodeDefinition,
    TerminalNodeDefinition,
    MergeNodeDefinition,
    PassthroughNodeDefinition,
    ShapeComputerMixin,
    ValidatorMixin
)

__all__ = [
    'Framework',
    'TensorShape',
    'NodeMetadata',
    'ConfigField',
    'NodeDefinition',
    'SourceNodeDefinition',
    'TerminalNodeDefinition',
    'MergeNodeDefinition',
    'PassthroughNodeDefinition',
    'ShapeComputerMixin',
    'ValidatorMixin'
]
