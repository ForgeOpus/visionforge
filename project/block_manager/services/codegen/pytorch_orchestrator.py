"""
PyTorch Code Generation Orchestrator
Coordinates the generation of complete PyTorch project files
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import json

from .base import topological_sort, get_input_variable, get_node_type, get_node_config
from ..nodes.registry import get_node_definition
from ..nodes.base import Framework, LayerCodeSpec
from ..nodes.templates.manager import TemplateManager


class UnsupportedNodeTypeError(Exception):
    """Raised when a node type is not supported"""
    pass


class PyTorchCodeOrchestrator:
    """
    Orchestrator for PyTorch code generation.
    Delegates code generation to individual node classes and assembles the final output.
    """

    def __init__(self):
        self.template_manager = TemplateManager()

    def generate(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        project_name: str = "GeneratedModel",
        group_definitions: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, str], List[Exception]]:
        """
        Generate complete PyTorch project files.

        Args:
            nodes: List of node definitions from the frontend
            edges: List of edge definitions
            project_name: Name for the generated model class
            group_definitions: Optional group definitions (not yet implemented)

        Returns:
            Tuple of (files dict, errors list)
            files dict contains: {'model': str, 'train': str, 'dataset': str, 'config': str}
        """
        errors = []

        try:
            # Sort nodes topologically
            sorted_nodes = topological_sort(nodes, edges)

            # Build edge map for quick lookups
            edge_map = self._build_edge_map(edges)

            # Generate code specifications for each node
            code_specs, spec_errors = self._generate_code_specs(sorted_nodes, edge_map)
            errors.extend(spec_errors)

            # Render layer classes from templates
            layer_classes = self._render_layer_classes(code_specs)

            # Generate model class definition
            model_definition = self._generate_model_definition(
                project_name,
                code_specs,
                sorted_nodes,
                edge_map
            )

            # Generate test code
            input_shape = self._extract_input_shape(nodes)
            test_code = self._generate_test_code(project_name, input_shape)

            # Render complete model file
            model_code = self._render_model_file(
                project_name,
                layer_classes,
                model_definition,
                test_code
            )

            # Generate training script
            train_code = self._generate_training_script(project_name, nodes)

            # Generate dataset script
            dataset_code = self._generate_dataset_script(nodes)

            # Generate config file
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
        """
        Generate code specifications for all nodes.

        Returns:
            Tuple of (list of code specs, list of errors)
        """
        code_specs = []
        errors = []

        # Skip input/dataloader/output nodes - they don't generate layers
        processable_nodes = [
            n for n in sorted_nodes
            if get_node_type(n) not in ('input', 'dataloader', 'output')
        ]

        for node in processable_nodes:
            try:
                node_id = node['id']
                node_type = get_node_type(node)
                config = get_node_config(node)

                # Get node definition from registry
                node_def = get_node_definition(node_type, Framework.PYTORCH)

                if not node_def:
                    raise UnsupportedNodeTypeError(
                        f"Node type '{node_type}' (id: {node_id}) is not supported for PyTorch"
                    )

                # Generate code specification
                # Note: Shape inference would ideally happen here
                # For now, we pass None and let the node handle it
                code_spec = node_def.get_pytorch_code_spec(
                    node_id=node_id,
                    config=config,
                    input_shape=None,  # TODO: Add shape inference
                    output_shape=None
                )

                code_specs.append(code_spec)

            except Exception as e:
                errors.append(e)

        return code_specs, errors

    def _render_layer_classes(self, code_specs: List[LayerCodeSpec]) -> str:
        """
        Render all unique layer class definitions.

        Returns:
            String containing all layer class definitions
        """
        unique_classes = {}

        for spec in code_specs:
            # Use node_type as key to ensure we only define each class type once
            if spec.node_type not in unique_classes:
                try:
                    template_path = spec.get_template_path(Framework.PYTORCH)
                    rendered = self.template_manager.render(
                        template_path,
                        spec.template_context
                    )
                    unique_classes[spec.node_type] = rendered
                except Exception:
                    # If template doesn't exist, skip this layer
                    # Error will be caught during model generation
                    pass

        # Join all classes with blank lines
        return '\n\n'.join(unique_classes.values())

    def _generate_model_definition(
        self,
        project_name: str,
        code_specs: List[LayerCodeSpec],
        sorted_nodes: List[Dict[str, Any]],
        edge_map: Dict[str, List[str]]
    ) -> str:
        """Generate the main model class definition"""
        # Generate layer initializations
        layer_inits = []
        for spec in code_specs:
            params_str = ', '.join(
                f"{k}={repr(v)}" for k, v in spec.init_params.items()
            )
            layer_inits.append(
                f"self.{spec.layer_variable_name} = {spec.class_name}({params_str})"
            )

        # Generate forward pass logic with skip connection support
        forward_lines, skip_connections = self._generate_forward_pass(
            sorted_nodes,
            edge_map,
            code_specs
        )

        model_class = f'''class {project_name}(nn.Module):
    """
    PyTorch Model for {project_name}

    This model is auto-generated from the VisionForge architecture.
    """

    def __init__(self):
        super({project_name}, self).__init__()
        #==========================
        #Layer Initializations:
        #==========================
{chr(10).join("        " + line for line in layer_inits)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor after processing through the model
        """
{chr(10).join("        " + line for line in forward_lines)}
        return x
'''
        return model_class

    def _generate_forward_pass(
        self,
        sorted_nodes: List[Dict[str, Any]],
        edge_map: Dict[str, List[str]],
        code_specs: List[LayerCodeSpec]
    ) -> Tuple[List[str], Set[str]]:
        """
        Generate forward pass logic, handling skip connections properly.

        Returns:
            Tuple of (forward pass lines, set of skip connection var names)
        """
        forward_lines = []
        var_map = {}  # Maps node_id to variable name
        skip_connections = set()
        spec_map = {spec.node_id: spec for spec in code_specs}

        # Process nodes in topological order
        processable_nodes = [
            n for n in sorted_nodes
            if get_node_type(n) not in ('output',)  # Keep input/dataloader for var mapping
        ]

        for node in processable_nodes:
            node_id = node['id']
            node_type = get_node_type(node)

            # Input and dataloader nodes set up the initial 'x' variable
            if node_type in ('input', 'dataloader'):
                var_map[node_id] = 'x'
                continue

            # Get incoming connections
            incoming = edge_map.get(node_id, [])

            # Determine input variable
            input_var = get_input_variable(incoming, var_map)

            # Get code spec for this node
            spec = spec_map.get(node_id)
            if not spec:
                continue

            # Generate forward pass line
            output_var = f"x_{node_id.replace('-', '_')}"

            # Handle multi-input nodes (add, concat)
            if node_type in ('add', 'concat'):
                if node_type == 'concat':
                    dim = spec.template_context.get('dim', 1)
                    forward_lines.append(
                        f"{output_var} = self.{spec.layer_variable_name}({input_var}, concat_dim={dim})"
                    )
                else:
                    forward_lines.append(
                        f"{output_var} = self.{spec.layer_variable_name}({input_var})"
                    )
            else:
                # Regular single-input node
                forward_lines.append(
                    f"{output_var} = self.{spec.layer_variable_name}({input_var})"
                )

            # Update variable map
            var_map[node_id] = output_var

            # Track skip connections (nodes with multiple outgoing edges)
            # This helps identify which variables need to be preserved
            if len(incoming) > 1:
                skip_connections.add(output_var)

        # Ensure final output is assigned to 'x' for return statement
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
            shape_str = config.get('shape', '[1, 3, 224, 224]')
            try:
                shape = json.loads(shape_str) if isinstance(shape_str, str) else shape_str
                if isinstance(shape, list):
                    return tuple(shape)
            except (ValueError, TypeError):
                pass

        return (1, 3, 224, 224)

    def _generate_test_code(self, project_name: str, input_shape: Tuple[int, ...]) -> str:
        """Generate test code for model validation"""
        return f'''if __name__ == "__main__":
    # Test the model with random input
    model = {project_name}()
    model.eval()
    test_input = torch.randn({input_shape})
    print(f"Input shape: {{test_input.shape}}")
    output = model(test_input)
    print(f"Output shape: {{output.shape}}")
    print(f"Model has {{sum(p.numel() for p in model.parameters()):,}} parameters")
'''

    def _render_model_file(
        self,
        project_name: str,
        layer_classes: str,
        model_definition: str,
        test_code: str
    ) -> str:
        """Render the complete model.py file"""
        context = {
            'project_name': project_name,
            'layer_classes': layer_classes,
            'model_class_name': project_name,
            'layer_initializations': [],  # Handled in model_definition
            'forward_pass_lines': [],  # Handled in model_definition
            'test_code': test_code
        }

        # For now, use string formatting since we're embedding pre-rendered content
        # TODO: Convert model_definition to use template as well
        return f'''"""
Generated PyTorch Model
Architecture: {project_name}
Generated by VisionForge

This file contains the model architecture with separate layer classes.
Each layer is implemented as a reusable class for clarity and maintainability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


#==========================
#Layer Definitions:
#==========================
{layer_classes}

{model_definition}

{test_code}
'''

    def _generate_training_script(self, project_name: str, nodes: List[Dict[str, Any]]) -> str:
        """Generate training script using template"""
        # Determine task type based on architecture
        has_softmax = any(get_node_type(n) == 'softmax' for n in nodes)
        is_classification = has_softmax

        context = {
            'project_name': project_name,
            'model_class_name': project_name,
            'task_type': 'classification' if is_classification else 'regression',
            'is_classification': is_classification,
            'loss_function': 'nn.CrossEntropyLoss()' if is_classification else 'nn.MSELoss()',
            'metric_name': 'accuracy' if is_classification else 'mse'
        }

        return self.template_manager.render('pytorch/files/train.py.jinja2', context)

    def _generate_dataset_script(self, nodes: List[Dict[str, Any]]) -> str:
        """Generate dataset script using template"""
        input_shape = self._extract_input_shape(nodes)

        context = {
            'data_type': 'image',  # Default to image
            'input_shape': input_shape,
            'input_channels': input_shape[1] if len(input_shape) > 1 else 3,
            'input_height': input_shape[2] if len(input_shape) > 2 else 224,
            'input_width': input_shape[3] if len(input_shape) > 3 else 224,
            'channel_type': 'RGB' if input_shape[1] == 3 else 'Grayscale' if input_shape[1] == 1 else f'{input_shape[1]}-channel'
        }

        return self.template_manager.render('pytorch/files/dataset.py.jinja2', context)

    def _generate_config_file(self, nodes: List[Dict[str, Any]]) -> str:
        """Generate config file using template"""
        input_shape = self._extract_input_shape(nodes)

        # Count layers
        layer_count = sum(
            1 for n in nodes
            if get_node_type(n) not in ('input', 'output', 'dataloader')
        )

        # Determine complexity and hyperparameters
        if layer_count > 20:
            batch_size = 16
            learning_rate = 1e-4
            epochs = 100
            complexity = "Deep"
        elif layer_count > 10:
            batch_size = 32
            learning_rate = 1e-3
            epochs = 50
            complexity = "Medium"
        else:
            batch_size = 64
            learning_rate = 1e-3
            epochs = 30
            complexity = "Shallow"

        # Check for attention layers
        has_attention = any(get_node_type(n) in ('self_attention', 'attention') for n in nodes)
        if has_attention:
            learning_rate = learning_rate * 0.1
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

        return self.template_manager.render('pytorch/files/config.py.jinja2', context)
