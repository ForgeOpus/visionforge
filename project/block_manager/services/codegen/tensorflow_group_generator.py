"""
TensorFlow Group Block Generator
Generates TensorFlow keras.Model code for group block definitions.
"""

from typing import Dict, List, Any, Optional
from .group_block_generator import GroupBlockGenerator
from ..nodes.base import Framework, LayerCodeSpec
from ..nodes.registry import get_node_definition
from .base import get_node_type, get_node_config


class TensorFlowGroupBlockGenerator(GroupBlockGenerator):
    """TensorFlow-specific group block code generation"""

    def __init__(self):
        super().__init__(Framework.TENSORFLOW)

    def generate_group_block_spec(
        self,
        group_definition: Dict[str, Any],
        node_id: str,
        instance_config: Optional[Dict[str, Any]] = None
    ) -> LayerCodeSpec:
        """
        Generate LayerCodeSpec for TensorFlow group block instance.

        Args:
            group_definition: The GroupBlockDefinition dict
            node_id: The node ID of the group instance in main graph
            instance_config: Optional per-instance config overrides

        Returns:
            LayerCodeSpec for this group instance
        """
        class_name = self._sanitize_class_name(group_definition['name'])
        sanitized_id = node_id.replace('-', '_')
        layer_var_name = f"{sanitized_id}_{class_name}"

        # Extract init params from internal structure
        init_params = self._extract_init_params(group_definition, instance_config)

        # Build template context
        internal_structure = group_definition.get('internal_structure', {})
        port_mappings = internal_structure.get('portMappings', [])
        input_ports, output_ports = self._parse_port_mappings(port_mappings)

        return LayerCodeSpec(
            class_name=class_name,
            layer_variable_name=layer_var_name,
            node_type='group',
            node_id=node_id,
            init_params=init_params,
            config_params=instance_config or {},
            template_context={
                'group_definition_id': group_definition['id'],
                'has_multi_output': len(output_ports) > 1,
                'num_outputs': len(output_ports),
                'num_inputs': len(input_ports)
            }
        )

    def generate_group_class_code(
        self,
        group_definition: Dict[str, Any]
    ) -> str:
        """
        Generate TensorFlow group block class using template.

        Args:
            group_definition: The GroupBlockDefinition dict

        Returns:
            Rendered class code as string
        """
        internal_structure = group_definition.get('internal_structure', {})
        internal_nodes = internal_structure.get('nodes', [])
        internal_edges = internal_structure.get('edges', [])
        port_mappings = internal_structure.get('portMappings', [])

        # Sort internal nodes
        sorted_nodes = self._topologically_sort_internal_nodes(
            internal_nodes, internal_edges
        )

        # Parse ports
        input_ports, output_ports = self._parse_port_mappings(port_mappings)

        # Generate LayerCodeSpecs for internal nodes
        internal_specs = self._generate_internal_node_specs(sorted_nodes)

        # Build edge map
        edge_map = self._build_edge_map(internal_edges)

        # Generate forward pass lines
        forward_lines = self._generate_call_method(
            sorted_nodes,
            internal_specs,
            edge_map,
            input_ports,
            output_ports
        )

        # Build template context
        context = self._build_tensorflow_template_context(
            group_definition,
            sorted_nodes,
            internal_specs,
            input_ports,
            output_ports,
            forward_lines
        )

        # Render template
        template_path = 'tensorflow/layers/group_block.py.jinja2'
        return self.template_manager.render(template_path, context)

    def _generate_internal_node_specs(
        self,
        internal_nodes: List[Dict[str, Any]]
    ) -> List[LayerCodeSpec]:
        """
        Generate code specs for each internal node.

        Args:
            internal_nodes: Sorted list of internal nodes

        Returns:
            List of LayerCodeSpec for processable internal nodes
        """
        specs = []

        for node in internal_nodes:
            node_type = get_node_type(node)

            # Skip special nodes
            if node_type in ('input', 'output', 'dataloader', 'loss'):
                continue

            node_id = node['id']
            config = get_node_config(node)

            # Check if this is a nested group
            if node_type == 'group':
                # For Phase 1-2, we can defer nested groups
                # TODO: Implement nested group support
                continue

            # Get node definition from registry
            node_def = get_node_definition(node_type, Framework.TENSORFLOW)
            if node_def:
                spec = node_def.get_tensorflow_code_spec(
                    node_id=node_id,
                    config=config,
                    input_shape=None,  # Shape inference deferred
                    output_shape=None
                )
                specs.append(spec)

        return specs

    def _generate_call_method(
        self,
        sorted_nodes: List[Dict[str, Any]],
        internal_specs: List[LayerCodeSpec],
        edge_map: Dict[str, List[str]],
        input_ports: List[Dict],
        output_ports: List[Dict]
    ) -> List[str]:
        """
        Generate call method lines for internal graph.

        Args:
            sorted_nodes: Topologically sorted internal nodes
            internal_specs: LayerCodeSpecs for internal nodes
            edge_map: Map of node_id -> [incoming_node_ids]
            input_ports: Input port mappings
            output_ports: Output port mappings

        Returns:
            List of call method code lines
        """
        call_lines = []
        var_map = {}
        spec_map = {spec.node_id: spec for spec in internal_specs}

        # Map input ports to their external parameter names
        input_node_map = {
            port['internalNodeId']: port
            for port in input_ports
        }

        # Track which nodes are outputs (need to preserve their variables)
        output_node_ids = {port['internalNodeId'] for port in output_ports}

        for node in sorted_nodes:
            node_id = node['id']
            node_type = get_node_type(node)

            # Handle input nodes
            if node_type == 'input':
                # Map input node to its external parameter
                if node_id in input_node_map:
                    # For TensorFlow, inputs come as list or single value
                    if len(input_ports) == 1:
                        var_map[node_id] = 'inputs'
                    else:
                        # Multi-input: unpack from inputs list
                        idx = list(input_node_map.keys()).index(node_id)
                        var_map[node_id] = f'inputs[{idx}]'
                continue

            # Skip output, dataloader, and loss nodes
            if node_type in ('output', 'dataloader', 'loss'):
                continue

            # Get the spec for this node
            spec = spec_map.get(node_id)
            if not spec:
                continue

            # Determine input variable
            incoming = edge_map.get(node_id, [])
            if not incoming:
                input_var = 'inputs'
            elif len(incoming) == 1:
                input_var = var_map.get(incoming[0], 'inputs')
            else:
                # Multiple inputs (add, concat nodes)
                input_vars = [var_map.get(src, 'inputs') for src in incoming]
                input_var = f"[{', '.join(input_vars)}]"

            # Generate output variable
            output_var = f"x_{node_id.replace('-', '_')}"

            # Generate layer call with training parameter
            if node_type in ('add', 'concat'):
                # Merge nodes don't use training parameter
                if node_type == 'concat':
                    axis = spec.template_context.get('axis', -1)
                    call_lines.append(
                        f"{output_var} = self.{spec.layer_variable_name}({input_var}, axis={axis})"
                    )
                else:
                    call_lines.append(
                        f"{output_var} = self.{spec.layer_variable_name}({input_var})"
                    )
            elif node_type in ('dropout', 'batchnorm'):
                # Layers that use training parameter
                call_lines.append(
                    f"{output_var} = self.{spec.layer_variable_name}({input_var}, training=training)"
                )
            else:
                # Standard layer call
                call_lines.append(
                    f"{output_var} = self.{spec.layer_variable_name}({input_var})"
                )

            var_map[node_id] = output_var

        return call_lines

    def _build_tensorflow_template_context(
        self,
        group_definition: Dict[str, Any],
        sorted_nodes: List[Dict[str, Any]],
        internal_specs: List[LayerCodeSpec],
        input_ports: List[Dict],
        output_ports: List[Dict],
        call_lines: List[str]
    ) -> Dict[str, Any]:
        """
        Build comprehensive template context for TensorFlow group block.

        Args:
            group_definition: The group definition
            sorted_nodes: Sorted internal nodes
            internal_specs: LayerCodeSpecs for internal nodes
            input_ports: Input port mappings
            output_ports: Output port mappings
            call_lines: Generated call method code lines

        Returns:
            Complete template context dict
        """
        base_context = self._build_template_context(
            group_definition, sorted_nodes, input_ports, output_ports
        )

        # Build output variables
        output_node_ids = [port['internalNodeId'] for port in output_ports]
        output_vars = []
        for node_id in output_node_ids:
            # Use the variable name from call method
            var_name = f"x_{node_id.replace('-', '_')}"
            output_vars.append(var_name)

        return {
            **base_context,
            'internal_specs': internal_specs,
            'call_lines': call_lines,
            'output_vars': output_vars,
            'init_params': []  # No custom init params for Phase 1-2
        }

    def _extract_init_params(
        self,
        group_definition: Dict[str, Any],
        instance_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract initialization parameters for group block instance.

        Args:
            group_definition: The group definition
            instance_config: Per-instance config overrides

        Returns:
            Dict of init parameters (empty for Phase 1-2)
        """
        # For Phase 1-2, group blocks don't have instance-level init params
        # All config is baked into the class definition
        return {}
