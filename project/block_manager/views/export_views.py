from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from block_manager.serializers import ExportRequestSerializer
from block_manager.services.tensorflow_codegen import generate_tensorflow_code


@api_view(['POST'])
def export_model(request):
    """
    Export model code
    This endpoint matches the frontend API contract: /api/export
    """
    # Validate incoming data
    nodes = request.data.get('nodes', [])
    edges = request.data.get('edges', [])
    export_format = request.data.get('format', 'pytorch')
    project_name = request.data.get('projectName', 'GeneratedModel')
    
    if not nodes:
        return Response(
            {'error': 'No nodes provided'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        if export_format == 'pytorch':
            code = generate_pytorch_stub(nodes, edges, project_name)
        elif export_format == 'tensorflow':
            # Use the actual TensorFlow code generator
            generated = generate_tensorflow_code(nodes, edges, project_name)
            code = generated.get('model', '')
            # Optionally return all generated files
            return Response({
                'code': code,
                'additionalFiles': {
                    'train.py': generated.get('train', ''),
                    'dataset.py': generated.get('dataset', ''),
                    'config.py': generated.get('config', '')
                }
            })
        else:
            return Response(
                {'error': f'Unsupported export format: {export_format}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        return Response({
            'code': code,
        })
    
    except Exception as e:
        # Pass detailed error messages to frontend
        return Response(
            {
                'error': f'Code generation failed: {str(e)}',
                'details': str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


def generate_pytorch_stub(nodes, edges, project_name='GeneratedModel'):
    """Generate a stub PyTorch model"""
    return """import torch
import torch.nn as nn

class GeneratedModel(nn.Module):
    def __init__(self):
        super(GeneratedModel, self).__init__()
        # TODO: Layers will be generated based on your architecture
        pass
    
    def forward(self, x):
        # TODO: Forward pass will be generated based on your architecture
        return x

# Model generated from VisionForge
# Full code generation coming soon!
"""


def generate_tensorflow_stub(nodes, edges):
    """Generate a stub TensorFlow model"""
    return """import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    # TODO: Model will be generated based on your architecture
    model = keras.Sequential([
        # Layers will be added here
    ])
    return model

# Model generated from VisionForge
# Full code generation coming soon!
"""

