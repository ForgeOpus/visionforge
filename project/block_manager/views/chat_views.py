from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import logging

from block_manager.services.ai_service_factory import AIServiceFactory

logger = logging.getLogger(__name__)


@api_view(['POST'])
def chat_message(request):
    """
    Handle chat messages with Gemini AI integration.

    Endpoint: POST /api/chat

    Request body (JSON or FormData):
    - JSON format:
    {
        "message": str,
        "history": [{"role": "user"|"assistant", "content": str}],
        "modificationMode": bool,
        "workflowState": {"nodes": [...], "edges": [...]}
    }
    - FormData format (when file is included):
    {
        "file": File,
        "message": str,
        "history": JSON string,
        "modificationMode": str ("true"/"false"),
        "workflowState": JSON string
    }

    Response:
    {
        "response": str,
        "modifications": [{"action": str, "details": {...}, "explanation": str}] | null
    }
    """
    import json as json_lib

    # Check if request has file upload (FormData)
    uploaded_file = request.FILES.get('file', None)

    if uploaded_file:
        # Parse FormData parameters
        message = request.POST.get('message', '')
        try:
            history = json_lib.loads(request.POST.get('history', '[]'))
        except:
            history = []

        try:
            modification_mode = request.POST.get('modificationMode', 'false').lower() == 'true'
        except:
            modification_mode = False

        try:
            workflow_state = json_lib.loads(request.POST.get('workflowState', 'null'))
        except:
            workflow_state = None
    else:
        # Parse JSON body
        message = request.data.get('message', '')
        history = request.data.get('history', [])
        modification_mode = request.data.get('modificationMode', False)
        workflow_state = request.data.get('workflowState', None)

    if not message and not uploaded_file:
        return Response(
            {'error': 'No message or file provided'},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Get API key from request header
    api_key = request.headers.get('X-Gemini-Api-Key')
    if not api_key:
        return Response(
            {
                'error': 'API key required',
                'response': 'Please provide your Gemini API key to use the AI assistant.'
            },
            status=status.HTTP_401_UNAUTHORIZED
        )

    try:
        # Initialize AI service with user-provided API key
        ai_service = AIServiceFactory.create_service(api_key=api_key)
        provider_name = AIServiceFactory.get_provider_name()

        # Handle file upload if present
        file_content = None
        if uploaded_file:
            logger.info(f"Processing file with {provider_name}: {uploaded_file.name}")

            # For Gemini, upload file to Gemini API
            if provider_name == 'Gemini':
                file_content = ai_service.upload_file_to_gemini(uploaded_file)
                if not file_content:
                    return Response(
                        {
                            'error': f'Failed to upload file to {provider_name}',
                            'response': 'Sorry, I could not process the uploaded file. Please try again.'
                        },
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
            # For Claude, read file content directly
            elif provider_name == 'Claude':
                file_content = ai_service._read_file_content(uploaded_file)
                if file_content.get('type') == 'text' and 'Error' in file_content.get('text', ''):
                    return Response(
                        {
                            'error': f'Failed to process file with {provider_name}',
                            'response': file_content.get('text', 'Could not process file.')
                        },
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )

        # Get chat response
        # Note: For Gemini, file_content is the gemini_file object
        # For Claude, file_content is the formatted file dict
        result = ai_service.chat(
            message=message,
            history=history,
            modification_mode=modification_mode,
            workflow_state=workflow_state,
            **({'gemini_file': file_content} if provider_name == 'Gemini' else {'file_content': file_content})
        )

        return Response(result)

    except ValueError as e:
        # API key not configured or invalid provider
        logger.error(f"AI service configuration error: {e}")
        error_message = str(e)
        return Response(
            {
                'error': error_message,
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

    # Get API key from request header
    api_key = request.headers.get('X-Gemini-Api-Key')
    if not api_key:
        return Response({
            'suggestions': [
                'Please provide your Gemini API key to get AI-powered suggestions.',
                'Click the chat button to configure your API key.'
            ]
        })

    try:
        # Initialize AI service with user-provided API key
        ai_service = AIServiceFactory.create_service(api_key=api_key)
        provider_name = AIServiceFactory.get_provider_name()

        # Get suggestions
        workflow_state = {
            'nodes': nodes,
            'edges': edges
        }

        suggestions = ai_service.generate_suggestions(workflow_state)

        return Response({
            'suggestions': suggestions,
        })

    except ValueError as e:
        # API key not configured - return basic suggestions
        logger.warning(f"AI service not configured for suggestions: {e}")
        return Response({
            'suggestions': [
                "Configure AI_PROVIDER and corresponding API key to get AI-powered suggestions",
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
