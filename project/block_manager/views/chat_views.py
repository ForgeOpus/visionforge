from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import logging

from block_manager.services.gemini_service import GeminiChatService

logger = logging.getLogger(__name__)


@api_view(['POST'])
def chat_message(request):
    """
    Handle chat messages with Gemini AI integration.

    Endpoint: POST /api/chat

    Request body:
    {
        "message": str,
        "history": [{"role": "user"|"assistant", "content": str}],
        "modificationMode": bool,
        "workflowState": {"nodes": [...], "edges": [...]}
    }

    Response:
    {
        "response": str,
        "modifications": [{"action": str, "details": {...}, "explanation": str}] | null
    }
    """
    message = request.data.get('message', '')
    history = request.data.get('history', [])
    modification_mode = request.data.get('modificationMode', False)
    workflow_state = request.data.get('workflowState', None)

    if not message:
        return Response(
            {'error': 'No message provided'},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        # Initialize Gemini service
        gemini_service = GeminiChatService()

        # Get chat response
        result = gemini_service.chat(
            message=message,
            history=history,
            modification_mode=modification_mode,
            workflow_state=workflow_state
        )

        return Response(result)

    except ValueError as e:
        # API key not configured
        logger.error(f"Gemini API key error: {e}")
        return Response(
            {
                'error': 'Gemini API key is not configured. Please set GEMINI_API_KEY environment variable.',
                'response': 'Sorry, the AI chat service is not properly configured. Please contact the administrator.'
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    except Exception as e:
        # Other errors
        logger.error(f"Error in chat_message: {e}", exc_info=True)
        return Response(
            {
                'error': str(e),
                'response': 'An error occurred while processing your message. Please try again.'
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def get_suggestions(request):
    """
    Get model architecture suggestions based on current workflow.

    Endpoint: POST /api/suggestions

    Request body:
    {
        "nodes": [...],
        "edges": [...]
    }

    Response:
    {
        "suggestions": [str]
    }
    """
    nodes = request.data.get('nodes', [])
    edges = request.data.get('edges', [])

    if not nodes:
        return Response({
            'suggestions': ['Start by adding an Input node to define your model input.']
        })

    try:
        # Initialize Gemini service
        gemini_service = GeminiChatService()

        # Get suggestions
        workflow_state = {
            'nodes': nodes,
            'edges': edges
        }

        suggestions = gemini_service.generate_suggestions(workflow_state)

        return Response({
            'suggestions': suggestions,
        })

    except ValueError as e:
        # API key not configured - return basic suggestions
        logger.warning(f"Gemini API key not configured for suggestions: {e}")
        return Response({
            'suggestions': [
                "Configure GEMINI_API_KEY environment variable to get AI-powered suggestions",
                "Consider adding normalization layers (BatchNorm2D) after convolutional layers",
                "Add dropout layers to prevent overfitting",
                "Ensure your architecture has proper input and output nodes"
            ]
        })

    except Exception as e:
        logger.error(f"Error in get_suggestions: {e}", exc_info=True)
        return Response({
            'suggestions': [
                "Error generating suggestions. Please check your workflow configuration."
            ]
        })
