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
            group_definitions: Optional list of group block definitions

        Returns:
            Tuple of (files dict, errors list)
            files dict contains: {'model': str, 'train': str, 'dataset': str, 'config': str}
        """
        errors = []

        try:
            # Initialize group block generator if needed
            group_generator = None
            if group_definitions:
                from .tensorflow_group_generator import TensorFlowGroupBlockGenerator
                group_generator = TensorFlowGroupBlockGenerator()

            # Sort nodes topologically
            sorted_nodes = topological_sort(nodes, edges)

            # Build edge map for quick lookups
            edge_map = self._build_edge_map(edges)

            # Generate code specifications for each node
            code_specs, spec_errors = self._generate_code_specs(
                sorted_nodes, edge_map, group_generator, group_definitions
            )
            errors.extend(spec_errors)

            # Generate code specs for internal layers in group blocks
            if group_definitions:
                internal_specs, internal_errors = self._generate_internal_layer_specs(
                    group_definitions
                )
                code_specs.extend(internal_specs)
                errors.extend(internal_errors)

            # Render layer classes from templates (includes internal layers)
            layer_classes = self._render_layer_classes(code_specs)

            # Generate group block class definitions
            group_classes = ""
            if group_generator and group_definitions:
                group_classes = self._generate_group_block_classes(
                    group_definitions, group_generator
                )

            # Combine regular layers + group classes
            all_classes = layer_classes
            if group_classes:
                all_classes += "\n\n" + group_classes

            # Generate model class definition
            model_definition = self._generate_model_definition(
                project_name,
                code_specs,
                sorted_nodes,
                edges,
                edge_map
            )

            # Generate test code
            input_shape = self._extract_input_shape(nodes)
            test_code = self._generate_test_code(project_name, input_shape)

            # Render complete model file
            model_code = self._render_model_file(
                project_name,
                all_classes,
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

    def _build_outgoing_edge_map(self, edges: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Build a map of outgoing edges from each node.

        Returns:
            Dict mapping source_node_id -> [target_node_ids]
        """
        outgoing_map = defaultdict(list)
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            if source and target:
                outgoing_map[source].append(target)
        return dict(outgoing_map)

    def _generate_code_specs(
        self,
        sorted_nodes: List[Dict[str, Any]],
        edge_map: Dict[str, List[str]],
        group_generator=None,
        group_definitions: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[LayerCodeSpec], List[Exception]]:
        """Generate code specifications for all nodes including group blocks"""
        code_specs = []
        errors = []

        # Build group definition lookup
        group_def_map = {}
        if group_definitions:
            group_def_map = {gd['id']: gd for gd in group_definitions}

        processable_nodes = [
            n for n in sorted_nodes
            if get_node_type(n) not in ('input', 'dataloader', 'output', 'loss')
        ]

        for node in processable_nodes:
            try:
                node_id = node['id']
                node_type = get_node_type(node)
                config = get_node_config(node)

                # Handle group blocks
                if node_type == 'group':
                    if not group_generator:
                        raise UnsupportedNodeTypeError(
                            f"Group node {node_id} found but no group_definitions provided"
                        )

                    group_def_id = node.get('data', {}).get('groupDefinitionId')
                    group_def = group_def_map.get(group_def_id)

                    if not group_def:
                        raise ValueError(
                            f"Group definition {group_def_id} not found for node {node_id}"
                        )

                    # Generate spec for this group instance
                    code_spec = group_generator.generate_group_block_spec(
                        group_definition=group_def,
                        node_id=node_id,
                        instance_config=config
                    )
                    code_specs.append(code_spec)

                else:
                    # Regular node
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

    def _generate_internal_layer_specs(
        self,
        group_definitions: List[Dict[str, Any]]
    ) -> Tuple[List[LayerCodeSpec], List[Exception]]:
        """
        Generate LayerCodeSpecs for all unique internal layers used in group blocks.
        This ensures internal layer classes are defined before group blocks use them.

        Args:
            group_definitions: List of group block definitions

        Returns:
            Tuple of (list of internal layer specs, list of errors)
        """
        internal_specs = []
        errors = []
        seen_node_types = set()

        for group_def in group_definitions:
            internal_structure = group_def.get('internal_structure', {})
            internal_nodes = internal_structure.get('nodes', [])

            for node in internal_nodes:
                node_type = get_node_type(node)

                # Skip special nodes
                if node_type in ('input', 'output', 'dataloader', 'group', 'loss'):
                    continue

                # Only generate each node type once
                if node_type in seen_node_types:
                    continue

                seen_node_types.add(node_type)

                try:
                    node_id = node['id']
                    config = get_node_config(node)

                    # Get node definition from registry
                    node_def = get_node_definition(node_type, Framework.TENSORFLOW)

                    if not node_def:
                        raise UnsupportedNodeTypeError(
                            f"Internal node type '{node_type}' not supported in group block"
                        )

                    # Generate code specification
                    code_spec = node_def.get_tensorflow_code_spec(
                        node_id=node_id,
                        config=config,
                        input_shape=None,
                        output_shape=None
                    )

                    internal_specs.append(code_spec)

                except Exception as e:
                    errors.append(e)

        return internal_specs, errors

    def _render_layer_classes(self, code_specs: List[LayerCodeSpec]) -> str:
        """Render all unique layer class definitions"""
        unique_classes = {}

        for spec in code_specs:
            if spec.node_type not in unique_classes:
                try:
                    template_path = spec.get_template_path(Framework.TENSORFLOW)
                    # Merge class_name, init_params, and template_context for rendering
                    context = {
                        'class_name': spec.class_name,
                        **spec.init_params,
                        **spec.template_context
                    }
                    rendered = self.template_manager.render(template_path, context)
                    unique_classes[spec.node_type] = rendered
                except Exception:
                    pass

        return '\n\n'.join(unique_classes.values())

    def _generate_group_block_classes(
        self,
        group_definitions: List[Dict[str, Any]],
        group_generator
    ) -> str:
        """
        Generate class definitions for all group blocks.

        Args:
            group_definitions: List of group block definitions
            group_generator: TensorFlowGroupBlockGenerator instance

        Returns:
            String containing all group block class definitions
        """
        # Detect dependency order for nested groups
        ordered_definitions = self._order_group_definitions(group_definitions)

        class_codes = []
        for group_def in ordered_definitions:
            try:
                class_code = group_generator.generate_group_class_code(group_def)
                class_codes.append(class_code)
            except Exception as e:
                # Log error but continue with other groups
                print(f"Error generating group {group_def.get('name')}: {e}")

        return '\n\n'.join(class_codes)

    def _order_group_definitions(
        self,
        group_definitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Order group definitions so nested groups are defined before their parents.
        Uses topological sort based on group dependencies.

        Args:
            group_definitions: List of group block definitions

        Returns:
            Topologically sorted list of group definitions
        """
        # Build dependency graph
        graph = {gd['id']: [] for gd in group_definitions}

        for group_def in group_definitions:
            internal_nodes = group_def.get('internal_structure', {}).get('nodes', [])
            for node in internal_nodes:
                if node.get('data', {}).get('blockType') == 'group':
                    nested_group_id = node.get('data', {}).get('groupDefinitionId')
                    if nested_group_id in graph:
                        # This group depends on the nested group
                        graph[group_def['id']].append(nested_group_id)

        # Topological sort (simple implementation)
        visited = set()
        result = []

        def visit(gd_id):
            if gd_id in visited:
                return
            visited.add(gd_id)
            for dep in graph.get(gd_id, []):
                visit(dep)
            result.append(gd_id)

        for gd in group_definitions:
            visit(gd['id'])

        # Map back to group definitions
        gd_map = {gd['id']: gd for gd in group_definitions}
        return [gd_map[gd_id] for gd_id in result if gd_id in gd_map]

    def _generate_model_definition(
        self,
        project_name: str,
        code_specs: List[LayerCodeSpec],
        sorted_nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
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
            edges,
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

    def _needs_named_variable(
        self,
        node_id: str,
        node_type: str,
        outgoing_edge_map: Dict[str, List[str]],
        num_outputs: int
    ) -> bool:
        """
        Determine if a node's output requires a named variable.

        A named variable is needed when:
        1. Node has multiple outgoing edges (output used multiple times - skip connections)
        2. Node has multiple outputs (e.g., group blocks)
        3. Node is a merge operation (add, concat) - for readability

        Args:
            node_id: Node identifier
            node_type: Type of node (e.g., 'conv2d', 'add')
            outgoing_edge_map: Map of node_id -> list of target node IDs
            num_outputs: Number of outputs for this node

        Returns:
            True if a named variable should be created
        """
        # Multi-output nodes always need named variables
        if num_outputs > 1:
            return True

        # Nodes with multiple outgoing edges need named variables (skip connections)
        outgoing_edges = outgoing_edge_map.get(node_id, [])
        if len(outgoing_edges) > 1:
            return True

        # Merge operations benefit from named variables for readability
        if node_type in ('add', 'concat'):
            return True

        return False

    def _generate_forward_pass(
        self,
        sorted_nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        edge_map: Dict[str, List[str]],
        code_specs: List[LayerCodeSpec]
    ) -> Tuple[List[str], Set[str]]:
        """
        Generate forward pass logic with optimized variable usage.

        Only creates named variables when needed:
        - Skip connections (nodes with multiple outgoing edges)
        - Multi-output nodes (group blocks)
        - Merge operations (add, concat)

        Otherwise reuses 'x' variable for memory efficiency.

        Returns:
            Tuple of (forward pass lines, set of skip connection var names)
        """
        forward_lines = []
        var_map = {}
        skip_connections = set()
        spec_map = {spec.node_id: spec for spec in code_specs}

        # Build outgoing edge map to detect skip connections
        outgoing_edge_map = self._build_outgoing_edge_map(edges)

        processable_nodes = [
            n for n in sorted_nodes
            if get_node_type(n) not in ('output', 'loss')
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

            # Handle group blocks with multiple outputs
            if node_type == 'group' and spec.template_context.get('has_multi_output'):
                num_outputs = spec.template_context.get('num_outputs', 1)
                output_vars = [
                    f"{node_id.replace('-', '_')}_out{i}"
                    for i in range(num_outputs)
                ]
                forward_lines.append(
                    f"{', '.join(output_vars)} = self.{spec.layer_variable_name}({input_var}, training=training)"
                )
                # Map first output as primary variable for this node
                var_map[node_id] = output_vars[0]
                skip_connections.add(output_vars[0])

            else:
                # Determine if this node needs a named variable
                num_outputs = 1
                needs_named_var = self._needs_named_variable(
                    node_id,
                    node_type,
                    outgoing_edge_map,
                    num_outputs
                )

                if needs_named_var:
                    # Create named variable for skip connections, merge ops, etc.
                    output_var = f"x_{node_id.replace('-', '_')}"
                    forward_lines.append(
                        f"{output_var} = self.{spec.layer_variable_name}({input_var}, training=training)"
                    )
                    var_map[node_id] = output_var

                    # Track if this is a skip connection source
                    if len(outgoing_edge_map.get(node_id, [])) > 1:
                        skip_connections.add(output_var)

                else:
                    # Reuse 'x' variable for linear chains (memory efficient)
                    forward_lines.append(
                        f"x = self.{spec.layer_variable_name}({input_var}, training=training)"
                    )
                    var_map[node_id] = 'x'

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

    def _extract_loss_config(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract loss configuration from loss node (REQUIRED).

        Args:
            nodes: List of node definitions

        Returns:
            Dictionary with loss configuration

        Raises:
            ValueError: If no loss node is found
        """
        loss_node = next((n for n in nodes if get_node_type(n) == 'loss'), None)

        if not loss_node:
            raise ValueError(
                "No loss function node found in architecture. "
                "Please add a Loss Function node from the 'Output' category "
                "to specify the training loss."
            )

        config = get_node_config(loss_node)
        loss_type = config.get('loss_type', 'cross_entropy')
        reduction = config.get('reduction', 'sum_over_batch_size')
        from_logits = config.get('from_logits', True)

        return {
            'loss_type': loss_type,
            'reduction': reduction,
            'from_logits': from_logits
        }

    def _generate_training_script(self, project_name: str, nodes: List[Dict[str, Any]]) -> str:
        """Generate training script using template"""
        # Extract loss configuration from loss node
        loss_config = self._extract_loss_config(nodes)

        # Map loss types to TensorFlow/Keras loss classes
        loss_map = {
            'cross_entropy': 'keras.losses.SparseCategoricalCrossentropy',
            'mse': 'keras.losses.MeanSquaredError',
            'mae': 'keras.losses.MeanAbsoluteError',
            'bce': 'keras.losses.BinaryCrossentropy',
            'categorical_crossentropy': 'keras.losses.CategoricalCrossentropy',
            'kl_div': 'keras.losses.KLDivergence',
            'hinge': 'keras.losses.Hinge',
        }

        loss_class = loss_map.get(loss_config['loss_type'], 'keras.losses.SparseCategoricalCrossentropy')

        # Build loss function instantiation with parameters
        loss_params = []
        if loss_config['from_logits'] is not None and loss_config['loss_type'] in ['cross_entropy', 'bce', 'categorical_crossentropy']:
            loss_params.append(f"from_logits={loss_config['from_logits']}")
        if loss_config['reduction'] and loss_config['reduction'] != 'sum_over_batch_size':
            loss_params.append(f"reduction='{loss_config['reduction']}'")

        loss_function = f"{loss_class}({', '.join(loss_params)})" if loss_params else f"{loss_class}()"

        # Determine if classification based on loss type
        is_classification = loss_config['loss_type'] in ['cross_entropy', 'bce', 'categorical_crossentropy']

        model_class_name = project_name.replace(project_name, "".join(c if c.isalnum() else "_" for c in project_name))

        context = {
            'project_name': project_name,
            'model_class_name': model_class_name,
            'task_type': 'classification' if is_classification else 'regression',
            'is_classification': is_classification,
            'loss_function': loss_function,
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

        # Count layers (exclude special nodes)
        layer_count = sum(
            1 for n in nodes
            if get_node_type(n) not in ('input', 'output', 'dataloader', 'loss')
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

        # Check for attention layers (affects learning rate)
        has_attention = any(get_node_type(n) in ('self_attention', 'attention') for n in nodes)
        if has_attention:
            learning_rate = learning_rate * 0.1
            batch_size = max(8, batch_size // 2)

        # Get loss configuration for reference in config
        loss_config = self._extract_loss_config(nodes)
        is_classification = loss_config['loss_type'] in ['cross_entropy', 'bce', 'categorical_crossentropy']

        context = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': epochs,
            'input_shape': list(input_shape),
            'complexity': complexity,
            'layer_count': layer_count,
            'has_attention': has_attention,
            'loss_type': loss_config['loss_type'],
            'is_classification': is_classification
        }

        return self.template_manager.render('tensorflow/files/config.py.jinja2', context)
