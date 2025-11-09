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
        source_handle = edge.get('sourceHandle', '')
        target_handle = edge.get('targetHandle', '')
        
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
