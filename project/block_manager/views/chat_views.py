from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status


@api_view(['POST'])
def chat_message(request):
    """
    Handle chat messages
    This endpoint matches the frontend API contract: /api/chat
    """
    message = request.data.get('message', '')
    history = request.data.get('history', [])
    
    if not message:
        return Response(
            {'error': 'No message provided'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # TODO: Implement actual AI chat integration
    # For now, return a stub response
    
    response_text = f"I received your message: '{message}'. AI chat integration coming soon!"
    
    return Response({
        'response': response_text,
    })


@api_view(['POST'])
def get_suggestions(request):
    """
    Get model architecture suggestions
    This endpoint matches the frontend API contract: /api/suggestions
    """
    nodes = request.data.get('nodes', [])
    edges = request.data.get('edges', [])
    
    # TODO: Implement actual suggestion logic
    # For now, return basic suggestions
    
    suggestions = [
        "Consider adding a Dropout layer to prevent overfitting",
        "Add Batch Normalization after convolutional layers",
        "Use ReLU activation for faster convergence",
    ]
    
    return Response({
        'suggestions': suggestions,
    })
