"""
Node Definition Registry for Backend
Dynamically loads and manages node definitions for all supported frameworks
"""

import importlib
import pkgutil
from typing import Dict, List, Optional, Type
from .base import NodeDefinition, Framework


class NodeRegistry:
    """Registry for managing node definitions across frameworks"""
    
    def __init__(self):
        self._registry: Dict[Framework, Dict[str, NodeDefinition]] = {
            Framework.PYTORCH: {},
            Framework.TENSORFLOW: {}
        }
        self._initialized = False
    
    def _initialize(self):
        """Initialize the registry by loading all node definitions"""
        if self._initialized:
            return
        
        # Load PyTorch nodes
        self._load_framework_nodes(Framework.PYTORCH, 'block_manager.services.nodes.pytorch')
        
        # Load TensorFlow nodes
        self._load_framework_nodes(Framework.TENSORFLOW, 'block_manager.services.nodes.tensorflow')
        
        self._initialized = True
    
    def _load_framework_nodes(self, framework: Framework, package_name: str):
        """Load all node definitions from a framework package"""
        try:
            package = importlib.import_module(package_name)
            
            # Iterate through all modules in the package
            for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
                if modname == '__init__':
                    continue
                
                # Import the module
                module = importlib.import_module(f"{package_name}.{modname}")
                
                # Find all NodeDefinition subclasses in the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    
                    # Check if it's a class and a subclass of NodeDefinition
                    if (isinstance(attr, type) and 
                        issubclass(attr, NodeDefinition) and 
                        attr is not NodeDefinition):
                        
                        # Instantiate and register
                        try:
                            instance = attr()
                            node_type = instance.metadata.type
                            self._registry[framework][node_type] = instance
                        except Exception as e:
                            print(f"Error instantiating {attr_name}: {e}")
        
        except ImportError as e:
            print(f"Could not load {package_name}: {e}")
    
    def get_node_definition(
        self,
        node_type: str,
        framework: Framework = Framework.PYTORCH
    ) -> Optional[NodeDefinition]:
        """
        Get a specific node definition by type and framework
        
        Args:
            node_type: The type of node to retrieve
            framework: The target framework (defaults to PyTorch)
        
        Returns:
            The node definition or None if not found
        """
        self._initialize()
        return self._registry.get(framework, {}).get(node_type)
    
    def get_all_node_definitions(
        self,
        framework: Framework = Framework.PYTORCH
    ) -> List[NodeDefinition]:
        """
        Get all node definitions for a specific framework
        
        Args:
            framework: The target framework (defaults to PyTorch)
        
        Returns:
            List of all node definitions for the framework
        """
        self._initialize()
        return list(self._registry.get(framework, {}).values())
    
    def get_node_definitions_by_category(
        self,
        framework: Framework = Framework.PYTORCH
    ) -> Dict[str, List[NodeDefinition]]:
        """
        Get node definitions grouped by category
        
        Args:
            framework: The target framework (defaults to PyTorch)
        
        Returns:
            Dictionary mapping category to node definitions
        """
        self._initialize()
        by_category: Dict[str, List[NodeDefinition]] = {}
        
        for node in self.get_all_node_definitions(framework):
            category = node.metadata.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(node)
        
        return by_category
    
    def has_node_definition(
        self,
        node_type: str,
        framework: Framework = Framework.PYTORCH
    ) -> bool:
        """
        Check if a node type exists in the registry
        
        Args:
            node_type: The type of node to check
            framework: The target framework (defaults to PyTorch)
        
        Returns:
            True if the node type exists
        """
        self._initialize()
        return node_type in self._registry.get(framework, {})
    
    def get_available_node_types(
        self,
        framework: Framework = Framework.PYTORCH
    ) -> List[str]:
        """
        Get all available node types for a framework
        
        Args:
            framework: The target framework (defaults to PyTorch)
        
        Returns:
            List of node type strings
        """
        self._initialize()
        return list(self._registry.get(framework, {}).keys())
    
    def reset(self):
        """Reset the registry (useful for testing)"""
        self._registry = {
            Framework.PYTORCH: {},
            Framework.TENSORFLOW: {}
        }
        self._initialized = False


# Global registry instance
_global_registry = NodeRegistry()


# Convenience functions for accessing the global registry

def get_node_definition(
    node_type: str,
    framework: Framework = Framework.PYTORCH
) -> Optional[NodeDefinition]:
    """Get a specific node definition"""
    return _global_registry.get_node_definition(node_type, framework)


def get_all_node_definitions(
    framework: Framework = Framework.PYTORCH
) -> List[NodeDefinition]:
    """Get all node definitions for a framework"""
    return _global_registry.get_all_node_definitions(framework)


def get_node_definitions_by_category(
    framework: Framework = Framework.PYTORCH
) -> Dict[str, List[NodeDefinition]]:
    """Get node definitions grouped by category"""
    return _global_registry.get_node_definitions_by_category(framework)


def has_node_definition(
    node_type: str,
    framework: Framework = Framework.PYTORCH
) -> bool:
    """Check if a node type exists"""
    return _global_registry.has_node_definition(node_type, framework)


def get_available_node_types(
    framework: Framework = Framework.PYTORCH
) -> List[str]:
    """Get all available node types"""
    return _global_registry.get_available_node_types(framework)


def reset_registry():
    """Reset the global registry"""
    _global_registry.reset()
