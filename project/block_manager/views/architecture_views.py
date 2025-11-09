from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from block_manager.models import Project, ModelArchitecture, Block, Connection
from block_manager.serializers import (
    SaveArchitectureSerializer,
    ModelArchitectureSerializer,
)


@api_view(['POST'])
def save_architecture(request, project_id):
    """
    Save architecture for a project
    Accepts nodes and edges from frontend canvas
    """
    project = get_object_or_404(Project, pk=project_id)
    serializer = SaveArchitectureSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(
            {'success': False, 'error': serializer.errors},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    nodes = serializer.validated_data['nodes']
    edges = serializer.validated_data['edges']
    
    # Get or create architecture
    architecture, created = ModelArchitecture.objects.get_or_create(project=project)
    
    # Clear existing blocks and connections
    architecture.blocks.all().delete()
    architecture.connections.all().delete()
    
    # Create blocks from nodes
    node_id_to_block = {}
    for node in nodes:
        node_id = node.get('id')
        node_data = node.get('data', {})
        position = node.get('position', {'x': 0, 'y': 0})
        
        block = Block.objects.create(
            architecture=architecture,
            node_id=node_id,
            block_type=node_data.get('blockType', 'unknown'),
            position_x=position.get('x', 0),
            position_y=position.get('y', 0),
            config=node_data.get('config', {}),
            input_shape=node_data.get('inputShape'),
            output_shape=node_data.get('outputShape'),
        )
        node_id_to_block[node_id] = block
    
    # Create connections from edges
    for edge in edges:
        edge_id = edge.get('id')
        source_id = edge.get('source')
        target_id = edge.get('target')
        # Ensure handles are never None, always use empty string as default
        source_handle = edge.get('sourceHandle') or ''
        target_handle = edge.get('targetHandle') or ''
        
        if source_id in node_id_to_block and target_id in node_id_to_block:
            Connection.objects.create(
                architecture=architecture,
                edge_id=edge_id,
                source_block=node_id_to_block[source_id],
                target_block=node_id_to_block[target_id],
                source_handle=source_handle,
                target_handle=target_handle,
            )
    
    # Update canvas state
    architecture.canvas_state = {
        'nodes': nodes,
        'edges': edges,
    }
    architecture.save()
    
    # Update project timestamp
    project.save()
    
    return Response({
        'success': True,
        'architecture_id': str(architecture.id),
        'validation': {
            'is_valid': architecture.is_valid,
            'errors': architecture.validation_errors,
        }
    })


@api_view(['GET'])
def load_architecture(request, project_id):
    """
    Load architecture for a project
    Returns nodes and edges for frontend canvas
    """
    project = get_object_or_404(Project, pk=project_id)
    
    try:
        architecture = project.architecture
    except ModelArchitecture.DoesNotExist:
        return Response({
            'nodes': [],
            'edges': [],
        })
    
    serializer = ModelArchitectureSerializer(architecture)
    
    # Return canvas state if available, otherwise reconstruct from blocks
    if architecture.canvas_state:
        return Response(architecture.canvas_state)
    
    # Reconstruct from database
    nodes = []
    for block in architecture.blocks.all():
        nodes.append({
            'id': block.node_id,
            'type': block.block_type,
            'position': {
                'x': block.position_x,
                'y': block.position_y,
            },
            'data': {
                'blockType': block.block_type,
                'config': block.config,
                'inputShape': block.input_shape,
                'outputShape': block.output_shape,
            }
        })
    
    edges = []
    for conn in architecture.connections.all():
        edges.append({
            'id': conn.edge_id,
            'source': conn.source_block.node_id,
            'target': conn.target_block.node_id,
            'sourceHandle': conn.source_handle,
            'targetHandle': conn.target_handle,
        })
    
    return Response({
        'nodes': nodes,
        'edges': edges,
    })


@api_view(['GET'])
def get_node_definitions(request):
    """
    Get available node definitions for a specific framework
    Returns node metadata and configuration schemas
    """
    from block_manager.services.nodes.specs.registry import list_node_specs
    from block_manager.services.nodes.specs.serialization import spec_to_dict
    from block_manager.services.nodes.specs import Framework
    
    # Get framework from query params, default to PyTorch
    framework_param = request.query_params.get('framework', 'pytorch').lower()
    
    try:
        framework = Framework(framework_param)
    except ValueError:
        return Response(
            {'success': False, 'error': f'Invalid framework: {framework_param}. Must be pytorch or tensorflow'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Get all node specs for the framework
    node_specs = list_node_specs(framework)
    
    # Serialize to dict format
    definitions_data = []
    for node_spec in node_specs:
        try:
            definitions_data.append(spec_to_dict(node_spec))
        except Exception as e:
            # Skip nodes that fail to serialize
            print(f"Error serializing node {node_spec.type}: {e}")
            continue
    
    return Response({
        'success': True,
        'framework': framework.value,
        'definitions': definitions_data,
        'count': len(definitions_data)
    })


@api_view(['GET'])
def get_node_definition(request, node_type):
    """
    Get a specific node definition by type
    """
    from block_manager.services.nodes.specs.registry import get_node_spec
    from block_manager.services.nodes.specs.serialization import spec_to_dict
    from block_manager.services.nodes.specs import Framework
    
    # Get framework from query params, default to PyTorch
    framework_param = request.query_params.get('framework', 'pytorch').lower()
    
    try:
        framework = Framework(framework_param)
    except ValueError:
        return Response(
            {'success': False, 'error': f'Invalid framework: {framework_param}'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Get the node spec
    node_spec = get_node_spec(node_type, framework)
    
    if not node_spec:
        return Response(
            {'success': False, 'error': f'Node type "{node_type}" not found for framework {framework.value}'},
            status=status.HTTP_404_NOT_FOUND
        )
    
    try:
        return Response({
            'success': True,
            'definition': spec_to_dict(node_spec)
        })
    except Exception as e:
        return Response(
            {'success': False, 'error': f'Error serializing node definition: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def render_node_code(request):
    """
    Render code for a node given its type, framework, and config.
    
    Request body:
    {
        "node_type": "conv2d",
        "framework": "pytorch",
        "config": {"out_channels": 64, "kernel_size": 3, ...},
        "metadata": {"node_id": "node_1", ...}  # optional
    }
    
    Returns:
    {
        "success": true,
        "code": "nn.Conv2d(3, 64, 3, ...)",
        "spec_hash": "abc123...",
        "node_type": "conv2d"
    }
    """
    from block_manager.services.nodes.specs.registry import get_node_spec
    from block_manager.services.nodes.specs.serialization import compute_spec_hash
    from block_manager.services.nodes.templates.renderer import render_node_template
    from block_manager.services.nodes.specs import Framework
    
    # Validate request data
    node_type = request.data.get('node_type')
    framework_param = request.data.get('framework', 'pytorch').lower()
    config = request.data.get('config', {})
    metadata = request.data.get('metadata', {})
    
    if not node_type:
        return Response(
            {'success': False, 'error': 'node_type is required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        framework = Framework(framework_param)
    except ValueError:
        return Response(
            {'success': False, 'error': f'Invalid framework: {framework_param}'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Get the node spec
    node_spec = get_node_spec(node_type, framework)
    
    if not node_spec:
        return Response(
            {'success': False, 'error': f'Node type "{node_type}" not found for framework {framework.value}'},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Render the template
    try:
        rendered = render_node_template(node_spec, config, metadata)
        
        # Compute hash from serialized spec
        from block_manager.services.nodes.specs.serialization import spec_to_dict
        spec_dict = spec_to_dict(node_spec)
        spec_hash = compute_spec_hash(spec_dict)
        
        return Response({
            'success': True,
            'code': rendered.code,
            'node_type': node_type,
            'framework': framework.value,
            'spec_hash': spec_hash,
            'context': rendered.context,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()  # Log to console for debugging
        return Response(
            {'success': False, 'error': f'Error rendering node code: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

