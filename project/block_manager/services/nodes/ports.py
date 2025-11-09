"""
Port Definition System for Node Connections
Defines typed ports for inputs and outputs with semantic meaning
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class PortSemantic(str, Enum):
    """Semantic types for ports"""
    DATA = "data"              # Regular data flow (activations, features)
    LABELS = "labels"          # Ground truth labels
    LOSS = "loss"              # Loss value output
    PREDICTIONS = "predictions"  # Model predictions
    ANCHOR = "anchor"          # Anchor for triplet loss
    POSITIVE = "positive"      # Positive example for triplet loss
    NEGATIVE = "negative"      # Negative example for triplet loss
    INPUT1 = "input1"          # Generic first input
    INPUT2 = "input2"          # Generic second input
    WEIGHTS = "weights"        # Model weights/parameters


@dataclass(frozen=True)
class PortDefinition:
    """Definition of a single input or output port"""
    id: str
    label: str
    type: str  # 'input' or 'output'
    semantic: PortSemantic
    required: bool
    description: str
    accepts_multiple: bool = False  # For inputs that can receive multiple connections


@dataclass
class NodePortSchema:
    """Complete port schema for a node"""
    inputs: List[PortDefinition]
    outputs: List[PortDefinition]


def are_ports_compatible(source: PortDefinition, target: PortDefinition) -> bool:
    """
    Check if two ports are semantically compatible for connection
    
    Args:
        source: Output port from source node
        target: Input port from target node
        
    Returns:
        True if ports can be connected
    """
    # Ground truth/labels can only connect to label inputs
    if source.semantic == PortSemantic.LABELS:
        return target.semantic == PortSemantic.LABELS
    
    # Loss output should only connect to optimizer (not implemented yet, allow for now)
    if source.semantic == PortSemantic.LOSS:
        return True  # Will be restricted when optimizer nodes are added
    
    # Predictions can connect to loss or other prediction inputs
    if source.semantic == PortSemantic.PREDICTIONS:
        return target.semantic in [
            PortSemantic.PREDICTIONS,
            PortSemantic.LOSS,
            PortSemantic.DATA
        ]
    
    # Data outputs can connect to most inputs
    if source.semantic == PortSemantic.DATA:
        return target.semantic in [
            PortSemantic.DATA,
            PortSemantic.ANCHOR,
            PortSemantic.POSITIVE,
            PortSemantic.NEGATIVE,
            PortSemantic.PREDICTIONS,
            PortSemantic.INPUT1,
            PortSemantic.INPUT2
        ]
    
    # Generic compatibility - same semantic types can connect
    return source.semantic == target.semantic


def validate_port_connection(
    source_port: PortDefinition,
    target_port: PortDefinition
) -> tuple[bool, Optional[str]]:
    """
    Validate if a connection between two specific ports is allowed
    
    Args:
        source_port: Output port from source node
        target_port: Input port from target node
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check type compatibility
    if source_port.type != 'output':
        return False, 'Source must be an output port'
    
    if target_port.type != 'input':
        return False, 'Target must be an input port'
    
    # Check semantic compatibility
    if not are_ports_compatible(source_port, target_port):
        return False, f'Cannot connect {source_port.semantic.value} to {target_port.semantic.value}'
    
    return True, None


# Default ports for standard nodes
DEFAULT_INPUT_PORT = PortDefinition(
    id='default',
    label='Input',
    type='input',
    semantic=PortSemantic.DATA,
    required=True,
    description='Default input port'
)

DEFAULT_OUTPUT_PORT = PortDefinition(
    id='default',
    label='Output',
    type='output',
    semantic=PortSemantic.DATA,
    required=False,
    description='Default output port'
)
