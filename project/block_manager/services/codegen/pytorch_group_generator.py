"""
PyTorch Group Block Generator
Generates PyTorch nn.Module code for group block definitions.
"""

from typing import Dict, List, Any, Optional
from .group_block_generator import GroupBlockGenerator
from ..nodes.base import Framework, LayerCodeSpec
from ..nodes.registry import get_node_definition
from .base import get_node_type, get_node_config


class PyTorchGroupBlockGenerator(GroupBlockGenerator):
    """PyTorch-specific group block code generation"""

    def __init__(self):
        super().__init__(Framework.PYTORCH)

    def generate_group_block_spec(
        self,
        group_definition: Dict[str, Any],
        node_id: str,
        instance_config: Optional[Dict[str, Any]] = None,
        input_shape: Optional[Any] = None
    ) -> LayerCodeSpec:
        """
        Generate LayerCodeSpec for PyTorch group block instance.

        Args:
            group_definition: The GroupBlockDefinition dict
            node_id: The node ID of the group instance in main graph
            instance_config: Optional per-instance config overrides
            input_shape: Optional input shape for the group block

        Returns:
            LayerCodeSpec for this group instance
        """
        class_name = self._sanitize_class_name(group_definition['name'])
        sanitized_id = node_id.replace('-', '_')
        layer_var_name = f"{sanitized_id}_{class_name}"

        # Extract init params from internal structure with shape information
        init_params = self._extract_init_params(group_definition, instance_config, input_shape)

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
        group_definition: Dict[str, Any],
        input_shape: Optional[Any] = None
    ) -> str:
        """
        Generate PyTorch group block class using template.

        Args:
            group_definition: The GroupBlockDefinition dict
            input_shape: Optional representative input shape for the group block

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

        # Generate LayerCodeSpecs for internal nodes with shape inference
        internal_specs = self._generate_internal_node_specs(
            sorted_nodes,
            internal_edges=internal_edges,
            input_shape=input_shape
        )

        # Build edge map
        edge_map = self._build_edge_map(internal_edges)

        # Generate forward pass lines
        forward_lines = self._generate_forward_pass(
            sorted_nodes,
            internal_specs,
            edge_map,
            input_ports,
            output_ports
        )

        # Build template context
        context = self._build_pytorch_template_context(
            group_definition,
            sorted_nodes,
            internal_specs,
            input_ports,
            output_ports,
            forward_lines
        )

        # Render template
        template_path = 'pytorch/layers/group_block.py.jinja2'
        return self.template_manager.render(template_path, context)

    def _generate_internal_node_specs(
        self,
        internal_nodes: List[Dict[str, Any]],
        internal_edges: Optional[List[Dict[str, Any]]] = None,
        input_shape: Optional[Any] = None
    ) -> List[LayerCodeSpec]:
        """
        Generate code specs for each internal node with shape inference.

        Args:
            internal_nodes: Sorted list of internal nodes
            internal_edges: List of internal edges for shape propagation
            input_shape: Input shape to the group block

        Returns:
            List of LayerCodeSpec for processable internal nodes
        """
        from ..nodes.rules.shape import TensorShape

        specs = []

        # Build edge map for shape inference
        edge_map = {}
        if internal_edges:
            from collections import defaultdict
            edge_map_builder = defaultdict(list)
            for edge in internal_edges:
                target = edge.get('target')
                source = edge.get('source')
                if target and source:
                    edge_map_builder[target].append(source)
            edge_map = dict(edge_map_builder)

        # Track output shapes of each node
        node_output_shapes = {}

        # Initialize input nodes with the group's input shape
        for node in internal_nodes:
            node_type = get_node_type(node)
            if node_type == 'input':
                node_output_shapes[node['id']] = input_shape

        for node in internal_nodes:
            node_type = get_node_type(node)

            # Skip special nodes
            if node_type in ('input', 'output', 'dataloader'):
                continue

            node_id = node['id']
            config = get_node_config(node)

            # Check if this is a nested group
            if node_type == 'group':
                # For Phase 1-2, we can defer nested groups
                # TODO: Implement nested group support
                continue

            # Determine input shape from incoming connections
            computed_input_shape = None
            incoming = edge_map.get(node_id, [])
            if incoming:
                if len(incoming) == 1:
                    computed_input_shape = node_output_shapes.get(incoming[0])
                else:
                    # Multiple inputs - use first for now
                    for src in incoming:
                        if src in node_output_shapes:
                            computed_input_shape = node_output_shapes[src]
                            break

            # Get node definition from registry
            node_def = get_node_definition(node_type, Framework.PYTORCH)
            if node_def:
                # Compute output shape
                computed_output_shape = None
                try:
                    if hasattr(node_def, 'compute_output_shape'):
                        computed_output_shape = node_def.compute_output_shape(computed_input_shape, config)
                except Exception:
                    pass

                # Generate spec with shape information
                spec = node_def.get_pytorch_code_spec(
                    node_id=node_id,
                    config=config,
                    input_shape=computed_input_shape,
                    output_shape=computed_output_shape
                )
                specs.append(spec)

                # Store output shape for downstream nodes
                if computed_output_shape:
                    node_output_shapes[node_id] = computed_output_shape

        return specs

    def _generate_forward_pass(
        self,
        sorted_nodes: List[Dict[str, Any]],
        internal_specs: List[LayerCodeSpec],
        edge_map: Dict[str, List[str]],
        input_ports: List[Dict],
        output_ports: List[Dict]
    ) -> List[str]:
        """
        Generate forward pass lines for internal graph.

        Args:
            sorted_nodes: Topologically sorted internal nodes
            internal_specs: LayerCodeSpecs for internal nodes
            edge_map: Map of node_id -> [incoming_node_ids]
            input_ports: Input port mappings
            output_ports: Output port mappings

        Returns:
            List of forward pass code lines
        """
        forward_lines = []
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
                    port_info = input_node_map[node_id]
                    # For now, use simple variable naming
                    if len(input_ports) == 1:
                        var_map[node_id] = 'x'
                    else:
                        # Multi-input: use port labels or indices
                        port_label = port_info.get('externalPortLabel', f'input_{len(var_map)}')
                        var_name = port_label.lower().replace(' ', '_')
                        var_map[node_id] = var_name
                continue

            # Skip output and dataloader nodes (they don't produce code)
            if node_type in ('output', 'dataloader'):
                continue

            # Get the spec for this node
            spec = spec_map.get(node_id)
            if not spec:
                # Node not in specs (might be nested group or unsupported)
                continue

            # Determine input variable
            incoming = edge_map.get(node_id, [])
            if not incoming:
                input_var = 'x'
            elif len(incoming) == 1:
                input_var = var_map.get(incoming[0], 'x')
            else:
                # Multiple inputs (add, concat nodes)
                input_vars = [var_map.get(src, 'x') for src in incoming]
                input_var = f"[{', '.join(input_vars)}]"

            # Generate output variable
            output_var = f"x_{node_id.replace('-', '_')}"

            # Generate layer call
            if node_type in ('add', 'concat'):
                # Special handling for merge nodes
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
                # Standard layer call
                forward_lines.append(
                    f"{output_var} = self.{spec.layer_variable_name}({input_var})"
                )

            var_map[node_id] = output_var

        return forward_lines

    def _build_pytorch_template_context(
        self,
        group_definition: Dict[str, Any],
        sorted_nodes: List[Dict[str, Any]],
        internal_specs: List[LayerCodeSpec],
        input_ports: List[Dict],
        output_ports: List[Dict],
        forward_lines: List[str]
    ) -> Dict[str, Any]:
        """
        Build comprehensive template context for PyTorch group block.

        Args:
            group_definition: The group definition
            sorted_nodes: Sorted internal nodes
            internal_specs: LayerCodeSpecs for internal nodes
            input_ports: Input port mappings
            output_ports: Output port mappings
            forward_lines: Generated forward pass code lines

        Returns:
            Complete template context dict
        """
        base_context = self._build_template_context(
            group_definition, sorted_nodes, input_ports, output_ports
        )

        # Build input parameter signature
        if len(input_ports) == 1:
            input_params = 'x'
        else:
            # Multi-input: generate parameter names from port labels
            param_names = []
            for port in input_ports:
                label = port.get('externalPortLabel', f'input_{len(param_names)}')
                param_name = label.lower().replace(' ', '_')
                param_names.append(param_name)
            input_params = ', '.join(param_names)

        # Build output variables
        output_node_ids = [port['internalNodeId'] for port in output_ports]
        output_vars = []
        for node_id in output_node_ids:
            # Use the variable name from forward pass
            var_name = f"x_{node_id.replace('-', '_')}"
            output_vars.append(var_name)

        return {
            **base_context,
            'internal_specs': internal_specs,
            'forward_lines': forward_lines,
            'input_params': input_params,
            'output_vars': output_vars,
            'init_params': []  # No custom init params for Phase 1-2
        }

    def _extract_init_params(
        self,
        group_definition: Dict[str, Any],
        instance_config: Optional[Dict[str, Any]],
        input_shape: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Extract initialization parameters for group block instance.

        Args:
            group_definition: The group definition
            instance_config: Per-instance config overrides
            input_shape: Optional input shape for the group block

        Returns:
            Dict of init parameters (empty for Phase 1-2)
        """
        # For Phase 1-2, group blocks don't have instance-level init params
        # All config is baked into the class definition
        return {}
