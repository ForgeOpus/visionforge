from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.request import Request
from rest_framework.permissions import AllowAny
from django.http import HttpResponse
from django.conf import settings
from django_ratelimit.decorators import ratelimit
import logging

from block_manager.serializers import ExportRequestSerializer
from block_manager.services.tensorflow_codegen import generate_tensorflow_code
from block_manager.services.enhanced_pytorch_codegen import generate_pytorch_code
from authentication.middleware import require_authentication

import zipfile
import io

logger = logging.getLogger(__name__)


@api_view(['POST'])
@permission_classes([AllowAny])
@require_authentication
@ratelimit(key='user_or_ip', rate='5/m', method='POST', block=True)
def export_model(request: Request) -> Response:
    """
    Export model code with professional class-based structure.

    Generates multiple files (model, train, dataset, config) for both frameworks.
    Returns a zip file containing all generated files.

    This endpoint matches the frontend API contract: /api/export
    """
    # Validate incoming data
    nodes = request.data.get('nodes', [])
    edges = request.data.get('edges', [])
    export_format = request.data.get('format', 'pytorch')
    project_name = request.data.get('projectName', 'GeneratedModel')
    group_definitions = request.data.get('groupDefinitions', [])

    if not nodes:
        return Response(
            {'error': 'No nodes provided'},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        # Generate code based on framework
        shape_errors = []
        if export_format == 'pytorch':
            generated, shape_errors = generate_pytorch_code(nodes, edges, project_name, group_definitions)
        elif export_format == 'tensorflow':
            generated, shape_errors = generate_tensorflow_code(nodes, edges, project_name, group_definitions)
        else:
            return Response(
                {'error': f'Unsupported export format: {export_format}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Check if there are shape errors that should prevent export
        if shape_errors:
            # Format shape errors with comprehensive details for frontend
            formatted_errors = []
            for error in shape_errors:
                error_dict = {
                    'type': 'error',  # Mark as error type for frontend
                    'message': str(error)
                }

                # Extract additional context from specific error types
                # These attributes come from our custom exception classes
                if hasattr(error, 'node_id'):
                    error_dict['nodeId'] = error.node_id
                if hasattr(error, 'node_type'):
                    error_dict['nodeType'] = error.node_type
                if hasattr(error, 'block_name'):
                    error_dict['blockName'] = error.block_name
                if hasattr(error, 'layer_name'):
                    error_dict['layerName'] = error.layer_name
                if hasattr(error, 'expected'):
                    error_dict['expected'] = error.expected
                if hasattr(error, 'actual'):
                    error_dict['actual'] = error.actual
                if hasattr(error, 'suggestion'):
                    error_dict['suggestion'] = error.suggestion
                if hasattr(error, 'reason'):
                    error_dict['reason'] = error.reason
                if hasattr(error, 'upstream_node_id'):
                    error_dict['upstreamNodeId'] = error.upstream_node_id
                if hasattr(error, 'missing_keys'):
                    error_dict['missingKeys'] = error.missing_keys
                if hasattr(error, 'framework'):
                    error_dict['framework'] = error.framework

                formatted_errors.append(error_dict)

            # Return validation-style error response that frontend expects
            return Response(
                {
                    'error': 'Code generation errors detected',
                    'validationErrors': formatted_errors,
                    'details': 'Please fix the errors in your architecture before exporting.',
                    'errorCount': len(formatted_errors)
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        # Create a zip file with all generated files
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add model file
            zip_file.writestr('model.py', generated.get('model', ''))
            # Add training script
            zip_file.writestr('train.py', generated.get('train', ''))
            # Add dataset file
            zip_file.writestr('dataset.py', generated.get('dataset', ''))
            # Add config file
            zip_file.writestr('config.py', generated.get('config', ''))

            # Add README
            readme_content = f"""# {project_name} - Generated by VisionForge

## Framework: {export_format.upper()}

This project was automatically generated from your visual architecture.

## Files:
- `model.py`: Contains the model architecture with separate layer classes
- `train.py`: Training script with best practices
- `dataset.py`: Dataset class template for loading your data
- `config.py`: Configuration file with all hyperparameters

## Usage:

1. Install dependencies:
   ```bash
   {"pip install torch torchvision" if export_format == 'pytorch' else "pip install tensorflow"}
   ```

2. Replace the dataset loading logic in `dataset.py` with your actual data.

3. Train the model:
   ```bash
   python train.py
   ```

## Model Architecture:
Each layer is implemented as a separate class for clarity and reusability.
The main model class combines all layers with proper type hints and documentation.

Generated with VisionForge
"""
            zip_file.writestr('README.md', readme_content)

        # Prepare response with zip file
        zip_buffer.seek(0)

        # Track first export milestone for user analytics
        if hasattr(request, 'firebase_user') and request.firebase_user:
            request.firebase_user.mark_first_export()

        # Return as JSON response with base64 encoded zip for frontend compatibility
        import base64
        zip_base64 = base64.b64encode(zip_buffer.getvalue()).decode('utf-8')

        return Response({
            'success': True,
            'framework': export_format,
            'projectName': project_name,
            'files': {
                'model.py': generated.get('model', ''),
                'train.py': generated.get('train', ''),
                'dataset.py': generated.get('dataset', ''),
                'config.py': generated.get('config', '')
            },
            'zip': zip_base64,  # Base64 encoded zip file
            'filename': f'{project_name}_{export_format}.zip'
        })

    except Exception as e:
        import traceback
        logger.error(f"Error in export_model: {str(e)}", exc_info=True)
        response = {
            'error': 'Code generation failed'
        }
        return Response(response, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

