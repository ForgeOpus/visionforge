"""
TensorFlow/Keras Code Generation Service
Generates tf.keras.Model code from architecture graphs with professional class-based structure
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import logging

# New template-based code generation
from .codegen.tensorflow_orchestrator import TensorFlowCodeOrchestrator

# NOTE: Legacy imports removed - all code generation now delegated to TensorFlowCodeOrchestrator
# The classes below were only used in legacy code that no longer executes:
# - GroupBlockShapeComputer, GroupDefinitionNotFoundError, ShapeMismatchError
# - CyclicDependencyError, UnsupportedNodeTypeError, ShapeInferenceError
# - MissingShapeDataError, safe_get_shape_data

# Configure logging
logger = logging.getLogger(__name__)


# ==================== LEGACY CLASS REMOVED ====================
# TensorFlowBlockGenerator class has been removed and replaced with:
# - TensorFlowGroupBlockGenerator in codegen/tensorflow_group_generator.py (for group blocks)
# - TensorFlowCodeOrchestrator in codegen/tensorflow_orchestrator.py (for overall code generation)
# ===============================================================


def generate_tensorflow_code(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    project_name: str = "GeneratedModel",
    group_definitions: Optional[List[Dict[str, Any]]] = None
) -> Tuple[Dict[str, str], List[Exception]]:
    """
    Generate complete TensorFlow/Keras code including model, training, and data loading.
    Each layer gets its own reusable class, all combined in a main model class.

    Args:
        nodes: List of node dictionaries from architecture
        edges: List of edge dictionaries defining connections
        project_name: Name for the generated model class
        group_definitions: Optional list of GroupBlockDefinition dictionaries

    Returns:
        Tuple of (dictionary with keys: 'model', 'train', 'dataset', 'config', list of errors)
    """
    # Delegate to new template-based orchestrator
    orchestrator = TensorFlowCodeOrchestrator()
    return orchestrator.generate(nodes, edges, project_name, group_definitions)

# ==================== LEGACY CODE REMOVED ====================
# All legacy code after the return statement has been removed.
# The new implementation is in:
# - codegen/tensorflow_orchestrator.py
# - codegen/tensorflow_group_generator.py
# ==============================================================
