from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import traceback

from block_manager.serializers import SaveArchitectureSerializer
from block_manager.services.validation import validate_architecture
from block_manager.services.inference import infer_dimensions


@api_view(['POST'])
def validate_model(request):
    """
    Validate model architecture
    This endpoint matches the frontend API contract: /api/validate
    """
    try:
        serializer = SaveArchitectureSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(
                {
                    'isValid': False,
                    'errors': [{'message': 'Invalid request format', 'type': 'error'}],
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        nodes = serializer.validated_data['nodes']
        edges = serializer.validated_data['edges']
        
        # Use validation service
        result = validate_architecture(nodes, edges)
        
        # Add dimension inference
        inferred_shapes = infer_dimensions(nodes, edges)
        if inferred_shapes:
            result['inferred_shapes'] = inferred_shapes
        
        return Response(result)
        
    except Exception as e:
        # Log the error for debugging
        print(f"Validation error: {str(e)}")
        traceback.print_exc()
        
        return Response(
            {
                'isValid': False,
                'errors': [{'message': f'Server error: {str(e)}', 'type': 'error'}],
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
