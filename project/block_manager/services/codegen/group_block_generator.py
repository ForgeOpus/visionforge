"""
Group Block Code Generator
Abstract base class for generating group block code across frameworks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import re

from ..nodes.base import Framework, LayerCodeSpec
from ..nodes.templates.manager import TemplateManager
from .base import topological_sort


class GroupBlockGenerator(ABC):
    """
    Abstract base class for generating group block code across frameworks.

    Responsibilities:
    - Parse group definition internal structure
    - Detect multi-I/O patterns from portMappings
    - Handle dependency ordering for nested internal nodes
    - Coordinate with TemplateManager for rendering
    """

    def __init__(self, framework: Framework):
        self.framework = framework
        self.template_manager = TemplateManager()

    @abstractmethod
    def generate_group_block_spec(
        self,
        group_definition: Dict[str, Any],
        node_id: str,
        instance_config: Optional[Dict[str, Any]] = None
    ) -> LayerCodeSpec:
        """
        Generate LayerCodeSpec for a group block instance.

        Args:
            group_definition: The GroupBlockDefinition dict with internal_structure
            node_id: The node ID of the group instance in the main graph
            instance_config: Optional per-instance config overrides

        Returns:
            LayerCodeSpec containing all info needed to render the group block
        """
        pass

    @abstractmethod
    def generate_group_class_code(
        self,
        group_definition: Dict[str, Any]
    ) -> str:
        """
        Generate the complete class code for a group block definition.

        Args:
            group_definition: The GroupBlockDefinition dict

        Returns:
            Rendered class code as string
        """
        pass

    def _parse_port_mappings(
        self,
        port_mappings: List[Dict[str, Any]]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Parse port mappings into input and output ports.

        Args:
            port_mappings: List of port mapping dicts from internal_structure

        Returns:
            Tuple of (input_ports, output_ports)
        """
        input_ports = [pm for pm in port_mappings if pm.get('type') == 'input']
        output_ports = [pm for pm in port_mappings if pm.get('type') == 'output']
        return input_ports, output_ports

    def _topologically_sort_internal_nodes(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Topologically sort internal nodes using existing base.topological_sort.

        Args:
            nodes: List of internal nodes
            edges: List of internal edges

        Returns:
            Topologically sorted list of nodes
        """
        return topological_sort(nodes, edges)

    def _detect_nested_groups(
        self,
        internal_nodes: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Detect which internal nodes are themselves group blocks.

        Args:
            internal_nodes: List of internal nodes

        Returns:
            List of group definition IDs referenced by internal nodes
        """
        nested_groups = []
        for node in internal_nodes:
            node_type = node.get('data', {}).get('blockType')
            if node_type == 'group':
                group_def_id = node.get('data', {}).get('groupDefinitionId')
                if group_def_id:
                    nested_groups.append(group_def_id)
        return nested_groups

    def _build_template_context(
        self,
        group_definition: Dict[str, Any],
        internal_nodes: List[Dict[str, Any]],
        input_ports: List[Dict],
        output_ports: List[Dict]
    ) -> Dict[str, Any]:
        """
        Build the base template context for rendering group block class.

        Args:
            group_definition: The group definition dict
            internal_nodes: Sorted list of internal nodes
            input_ports: List of input port mappings
            output_ports: List of output port mappings

        Returns:
            Dict with keys: class_name, description, layers, input_ports,
            output_ports, has_multi_input, has_multi_output, etc.
        """
        return {
            'class_name': self._sanitize_class_name(group_definition['name']),
            'group_name': group_definition['name'],
            'description': group_definition.get('description', ''),
            'input_ports': input_ports,
            'output_ports': output_ports,
            'has_multi_input': len(input_ports) > 1,
            'has_multi_output': len(output_ports) > 1,
            'num_inputs': len(input_ports),
            'num_outputs': len(output_ports),
            'internal_nodes': internal_nodes,
            'framework': self.framework.value
        }

    def _sanitize_class_name(self, name: str) -> str:
        """
        Convert group name to valid class name (PascalCase).

        Args:
            name: Original group name (may contain spaces, special chars)

        Returns:
            Valid PascalCase class name
        """
        # Remove special chars, capitalize words
        clean = re.sub(r'[^a-zA-Z0-9_]', '', name.replace(' ', '_'))
        parts = clean.split('_')
        return ''.join(word.capitalize() for word in parts if word)

    def _build_edge_map(
        self,
        edges: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Build a map of node_id -> list of incoming node_ids.

        Args:
            edges: List of edge dicts

        Returns:
            Dict mapping target node ID to list of source node IDs
        """
        edge_map = {}
        for edge in edges:
            target = edge.get('target')
            source = edge.get('source')
            if target and source:
                if target not in edge_map:
                    edge_map[target] = []
                edge_map[target].append(source)
        return edge_map
