"""
TensorFlow Code Generation Orchestrator
Coordinates the generation of complete TensorFlow/Keras project files
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


class TensorFlowCodeOrchestrator:
    """
    Orchestrator for TensorFlow/Keras code generation.
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
        Generate complete TensorFlow/Keras project files.

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

                node_def = get_node_definition(node_type, Framework.TENSORFLOW)

                if not node_def:
                    raise UnsupportedNodeTypeError(
                        f"Node type '{node_type}' (id: {node_id}) is not supported for TensorFlow"
                    )

                code_spec = node_def.get_tensorflow_code_spec(
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
                    template_path = spec.get_template_path(Framework.TENSORFLOW)
                    rendered = self.template_manager.render(
                        template_path,
                        spec.template_context
                    )
                    unique_classes[spec.node_type] = rendered
                except Exception:
                    pass

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

        # Generate forward pass logic
        forward_lines, _ = self._generate_forward_pass(
            sorted_nodes,
            edge_map,
            code_specs
        )

        model_class = f'''class {project_name}(keras.Model):
    """
    TensorFlow/Keras Model for {project_name}

    This model is auto-generated from the VisionForge architecture.
    """

    def __init__(self):
        super({project_name}, self).__init__()
        #==========================
        #Layer Initializations:
        #==========================
{chr(10).join("        " + line for line in layer_inits)}

    def call(self, inputs, training=None):
        """
        Forward pass through the model.

        Args:
            inputs: Input tensor (NHWC format)
            training: Whether in training mode

        Returns:
            Output tensor after processing through the model
        """
        x = inputs
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

            # Handle multi-input nodes
            if node_type in ('add', 'concat'):
                forward_lines.append(
                    f"{output_var} = self.{spec.layer_variable_name}.call({input_var}, training=training)"
                )
            else:
                forward_lines.append(
                    f"{output_var} = self.{spec.layer_variable_name}.call({input_var}, training=training)"
                )

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
            shape_str = config.get('shape', '[1, 224, 224, 3]')
            try:
                shape = json.loads(shape_str) if isinstance(shape_str, str) else shape_str
                if isinstance(shape, list):
                    return tuple(shape)
            except (ValueError, TypeError):
                pass

        return (1, 224, 224, 3)  # NHWC format default

    def _generate_test_code(self, project_name: str, input_shape: Tuple[int, ...]) -> str:
        """Generate test code for model validation"""
        return f'''if __name__ == "__main__":
    # Test the model with random input
    model = {project_name}()
    test_input = tf.random.normal({input_shape})
    print(f"Input shape: {{test_input.shape}}")
    output = model(test_input, training=False)
    print(f"Output shape: {{output.shape}}")
    print(f"Model has {{model.count_params():,}} parameters")
'''

    def _render_model_file(
        self,
        project_name: str,
        layer_classes: str,
        model_definition: str,
        test_code: str
    ) -> str:
        """Render the complete model.py file"""
        return f'''"""
Generated TensorFlow/Keras Model
Architecture: {project_name}
Generated by VisionForge

This file contains the model architecture with separate layer classes.
Each layer is implemented as a reusable class for clarity and maintainability.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
        has_softmax = any(get_node_type(n) == 'softmax' for n in nodes)
        is_classification = has_softmax

        context = {
            'project_name': project_name,
            'model_class_name': project_name,
            'task_type': 'classification' if is_classification else 'regression',
            'is_classification': is_classification,
            'loss_function': 'keras.losses.SparseCategoricalCrossentropy()' if is_classification else 'keras.losses.MeanSquaredError()',
            'metric_name': 'accuracy' if is_classification else 'mse'
        }

        return self.template_manager.render('tensorflow/files/train.py.jinja2', context)

    def _generate_dataset_script(self, nodes: List[Dict[str, Any]]) -> str:
        """Generate dataset script using template"""
        input_shape = self._extract_input_shape(nodes)

        context = {
            'data_type': 'image',
            'input_shape': input_shape,
            'input_height': input_shape[1] if len(input_shape) > 1 else 224,
            'input_width': input_shape[2] if len(input_shape) > 2 else 224,
            'input_channels': input_shape[3] if len(input_shape) > 3 else 3,
            'channel_type': 'RGB' if input_shape[3] == 3 else 'Grayscale' if input_shape[3] == 1 else f'{input_shape[3]}-channel'
        }

        return self.template_manager.render('tensorflow/files/dataset.py.jinja2', context)

    def _generate_config_file(self, nodes: List[Dict[str, Any]]) -> str:
        """Generate config file using template"""
        input_shape = self._extract_input_shape(nodes)

        layer_count = sum(
            1 for n in nodes
            if get_node_type(n) not in ('input', 'output', 'dataloader')
        )

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

        return self.template_manager.render('tensorflow/files/config.py.jinja2', context)
