"""
Base Code Generation Orchestrator
Shared functionality between PyTorch and TensorFlow orchestrators
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import json
from abc import ABC, abstractmethod

from .base import topological_sort, get_input_variable, get_node_type, get_node_config
from ..nodes.registry import get_node_definition
from ..nodes.base import Framework, LayerCodeSpec
from ..nodes.templates.manager import TemplateManager


class UnsupportedNodeTypeError(Exception):
    """Raised when a node type is not supported"""
    pass


class BaseCodeOrchestrator(ABC):
    """
    Base orchestrator for code generation.
    Provides common functionality for both PyTorch and TensorFlow.
    """

    def __init__(self):
        self.template_manager = TemplateManager()

    @property
    @abstractmethod
    def framework(self) -> Framework:
        """Return the framework this orchestrator targets"""
        pass

    @abstractmethod
    def _get_code_spec_method_name(self) -> str:
        """Return the method name for getting code specs (e.g., 'get_pytorch_code_spec')"""
        pass

    @abstractmethod
    def _generate_layer_call(
        self,
        layer_var: str,
        input_var: str,
        node_type: str,
        spec: LayerCodeSpec
    ) -> str:
        """Generate the code for calling a layer (framework-specific)"""
        pass

    @abstractmethod
    def _get_default_input_shape(self) -> Tuple[int, ...]:
        """Get default input shape for this framework"""
        pass

    def generate(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        project_name: str = "GeneratedModel",
        group_definitions: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, str], List[Exception]]:
        """
        Generate complete project files.

        Args:
            nodes: List of node definitions from the frontend
            edges: List of edge definitions
            project_name: Name for the generated model class
            group_definitions: Optional group definitions

        Returns:
            Tuple of (files dict, errors list)
        """
        errors = []

        try:
            sorted_nodes = topological_sort(nodes, edges)
            edge_map = self._build_edge_map(edges)
            code_specs, spec_errors = self._generate_code_specs(sorted_nodes, edge_map)
            errors.extend(spec_errors)

            layer_classes = self._render_layer_classes(code_specs)
            model_definition = self._generate_model_definition(
                project_name, code_specs, sorted_nodes, edge_map
            )

            input_shape = self._extract_input_shape(nodes)
            test_code = self._generate_test_code(project_name, input_shape)
            model_code = self._render_model_file(
                project_name, layer_classes, model_definition, test_code
            )

            train_code = self._generate_training_script(project_name, nodes)
            dataset_code = self._generate_dataset_script(nodes)
            config_code = self._generate_config_file(nodes)

            return {
                'model': model_code,
                'train': train_code,
                'dataset': dataset_code,
                'config': config_code
            }, errors

        except Exception as e:
            errors.append(e)
            return {}, errors

    def _build_edge_map(self, edges: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Build a map of node_id -> list of incoming node_ids"""
        edge_map = defaultdict(list)
        for edge in edges:
            target = edge.get('target')
            source = edge.get('source')
            if target and source:
                edge_map[target].append(source)
        return dict(edge_map)

    def _generate_code_specs(
        self,
        sorted_nodes: List[Dict[str, Any]],
        edge_map: Dict[str, List[str]]
    ) -> Tuple[List[LayerCodeSpec], List[Exception]]:
        """Generate code specifications for all nodes"""
        code_specs = []
        errors = []

        processable_nodes = [
            n for n in sorted_nodes
            if get_node_type(n) not in ('input', 'dataloader', 'output')
        ]

        for node in processable_nodes:
            try:
                node_id = node['id']
                node_type = get_node_type(node)
                config = get_node_config(node)

                node_def = get_node_definition(node_type, self.framework)

                if not node_def:
                    raise UnsupportedNodeTypeError(
                        f"Node type '{node_type}' (id: {node_id}) is not supported for {self.framework.value}"
                    )

                # Call the appropriate method dynamically
                method = getattr(node_def, self._get_code_spec_method_name())
                code_spec = method(
                    node_id=node_id,
                    config=config,
                    input_shape=None,
                    output_shape=None
                )

                code_specs.append(code_spec)

            except Exception as e:
                errors.append(e)

        return code_specs, errors

    def _render_layer_classes(self, code_specs: List[LayerCodeSpec]) -> str:
        """Render all unique layer class definitions"""
        unique_classes = {}

        for spec in code_specs:
            if spec.node_type not in unique_classes:
                try:
                    template_path = spec.get_template_path(self.framework)
                    rendered = self.template_manager.render(
                        template_path,
                        spec.template_context
                    )
                    unique_classes[spec.node_type] = rendered
                except Exception:
                    pass

        return '\n\n'.join(unique_classes.values())

    def _generate_forward_pass(
        self,
        sorted_nodes: List[Dict[str, Any]],
        edge_map: Dict[str, List[str]],
        code_specs: List[LayerCodeSpec]
    ) -> Tuple[List[str], Set[str]]:
        """Generate forward pass logic with skip connection support"""
        forward_lines = []
        var_map = {}
        skip_connections = set()
        spec_map = {spec.node_id: spec for spec in code_specs}

        processable_nodes = [
            n for n in sorted_nodes
            if get_node_type(n) not in ('output',)
        ]

        for node in processable_nodes:
            node_id = node['id']
            node_type = get_node_type(node)

            if node_type in ('input', 'dataloader'):
                var_map[node_id] = 'x'
                continue

            incoming = edge_map.get(node_id, [])
            input_var = get_input_variable(incoming, var_map)

            spec = spec_map.get(node_id)
            if not spec:
                continue

            output_var = f"x_{node_id.replace('-', '_')}"

            # Generate the layer call (framework-specific)
            layer_call = self._generate_layer_call(
                spec.layer_variable_name,
                input_var,
                node_type,
                spec
            )
            forward_lines.append(f"{output_var} = {layer_call}")

            var_map[node_id] = output_var

            if len(incoming) > 1:
                skip_connections.add(output_var)

        # Ensure final output is assigned to 'x'
        if processable_nodes:
            last_node_id = processable_nodes[-1]['id']
            last_var = var_map.get(last_node_id, 'x')
            if last_var != 'x':
                forward_lines.append(f"x = {last_var}")

        return forward_lines, skip_connections

    def _extract_input_shape(self, nodes: List[Dict[str, Any]]) -> Tuple[int, ...]:
        """Extract input shape from input node"""
        input_node = next((n for n in nodes if get_node_type(n) == 'input'), None)

        if input_node:
            config = get_node_config(input_node)
            shape_str = config.get('shape', '')
            try:
                shape = json.loads(shape_str) if isinstance(shape_str, str) else shape_str
                if isinstance(shape, list):
                    return tuple(shape)
            except (ValueError, TypeError):
                pass

        return self._get_default_input_shape()

    def _generate_training_script(self, project_name: str, nodes: List[Dict[str, Any]]) -> str:
        """Generate training script using template"""
        has_softmax = any(get_node_type(n) == 'softmax' for n in nodes)
        is_classification = has_softmax

        context = self._get_training_context(project_name, is_classification)
        template_path = f"{self.framework.value}/files/train.py.jinja2"
        return self.template_manager.render(template_path, context)

    def _generate_dataset_script(self, nodes: List[Dict[str, Any]]) -> str:
        """Generate dataset script using template"""
        input_shape = self._extract_input_shape(nodes)
        context = self._get_dataset_context(input_shape)
        template_path = f"{self.framework.value}/files/dataset.py.jinja2"
        return self.template_manager.render(template_path, context)

    def _generate_config_file(self, nodes: List[Dict[str, Any]]) -> str:
        """Generate config file using template"""
        input_shape = self._extract_input_shape(nodes)
        layer_count = sum(
            1 for n in nodes
            if get_node_type(n) not in ('input', 'output', 'dataloader', 'loss')
        )

        if layer_count > 20:
            batch_size, learning_rate, epochs, complexity = 16, 1e-4, 100, "Deep"
        elif layer_count > 10:
            batch_size, learning_rate, epochs, complexity = 32, 1e-3, 50, "Medium"
        else:
            batch_size, learning_rate, epochs, complexity = 64, 1e-3, 30, "Shallow"

        has_attention = any(get_node_type(n) in ('self_attention', 'attention') for n in nodes)
        if has_attention:
            learning_rate *= 0.1
            batch_size = max(8, batch_size // 2)

        context = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': epochs,
            'input_shape': list(input_shape),
            'complexity': complexity,
            'layer_count': layer_count,
            'has_attention': has_attention
        }

        template_path = f"{self.framework.value}/files/config.py.jinja2"
        return self.template_manager.render(template_path, context)

    @abstractmethod
    def _generate_model_definition(
        self,
        project_name: str,
        code_specs: List[LayerCodeSpec],
        sorted_nodes: List[Dict[str, Any]],
        edge_map: Dict[str, List[str]]
    ) -> str:
        """Generate the main model class definition (framework-specific)"""
        pass

    @abstractmethod
    def _generate_test_code(self, project_name: str, input_shape: Tuple[int, ...]) -> str:
        """Generate test code (framework-specific)"""
        pass

    @abstractmethod
    def _render_model_file(
        self,
        project_name: str,
        layer_classes: str,
        model_definition: str,
        test_code: str
    ) -> str:
        """Render the complete model file (framework-specific)"""
        pass

    @abstractmethod
    def _get_training_context(self, project_name: str, is_classification: bool) -> Dict[str, Any]:
        """Get template context for training script (framework-specific)"""
        pass

    @abstractmethod
    def _get_dataset_context(self, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Get template context for dataset script (framework-specific)"""
        pass
