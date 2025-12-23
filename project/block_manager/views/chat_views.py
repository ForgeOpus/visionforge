from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import logging
from django_ratelimit.decorators import ratelimit

from block_manager.services.ai_service_factory import AIServiceFactory
from block_manager.services.universal_ai_factory import UniversalAIFactory
from block_manager.services.api_key_detector import APIKeyDetector

logger = logging.getLogger(__name__)


@api_view(['POST'])
@ratelimit(key='user_or_ip', rate='20/m', method='POST', block=True)
def chat_message(request):
    """
    Handle chat messages with AI integration supporting both BYOK and server-side keys.

    Endpoint: POST /api/chat

    Request headers (PROD mode only):
        - X-OpenRouter-Api-Key: User's OpenRouter API key (unified access to all models)
        - X-Selected-Model: Selected model (e.g., 'gpt-5.2', 'claude-opus-4.5', 'gemini-3-flash')

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
    from django.conf import settings

    # Extract API key and model from headers (supports any provider in PROD mode)
    # X-API-Key accepts keys from: OpenRouter, Google AI, OpenAI, or Anthropic
    api_key = request.headers.get('X-API-Key') or request.headers.get('X-OpenRouter-Api-Key')  # Backward compatibility
    selected_model = request.headers.get('X-Selected-Model')  # Frontend model name (e.g., 'gpt-5.2', 'claude-opus-4.5', 'gemini-3-flash')

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

    try:
        # Initialize Universal AI service with API key and model (auto-detects provider)
        ai_service = UniversalAIFactory.detect_and_create_service(
            api_key=api_key,
            model=selected_model
        )
        provider_name = UniversalAIFactory.get_provider_name(api_key)

        # Log file upload if present
        if uploaded_file:
            logger.info(f"Processing file with {provider_name}: {uploaded_file.name}")

        # Get chat response (OpenRouter handles file uploads via base64 encoding)
        result = ai_service.chat(
            message=message,
            history=history,
            modification_mode=modification_mode,
            workflow_state=workflow_state,
            uploaded_file=uploaded_file
        )

        return Response(result)

    except ValueError as e:
        # API key not configured or invalid provider
        logger.error(f"AI service configuration error: {e}")
        error_message = str(e)

        # Check if this is a PROD mode (user key required) error
        requires_user_key = AIServiceFactory.requires_user_api_key()

        if requires_user_key and ('required' in error_message.lower() or 'provide' in error_message.lower()):
            # PROD mode - user needs to provide API key
            return Response(
                {
                    'error': error_message,
                    'response': 'Please provide your API key to use the AI assistant.'
                },
                status=status.HTTP_401_UNAUTHORIZED
            )
        else:
            # DEV mode - server configuration issue
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
@ratelimit(key='user_or_ip', rate='15/m', method='POST', block=True)
def get_suggestions(request):
    """
    Get model architecture suggestions based on current workflow.

    Endpoint: POST /api/suggestions

    Request headers (PROD mode only):
        - X-OpenRouter-Api-Key: User's OpenRouter API key (unified access to all models)
        - X-Selected-Model: Selected model (e.g., 'gpt-5.2', 'claude-opus-4.5', 'gemini-3-flash')

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
    # Extract API key and model from headers (supports any provider in PROD mode)
    api_key = request.headers.get('X-API-Key') or request.headers.get('X-OpenRouter-Api-Key')  # Backward compatibility
    selected_model = request.headers.get('X-Selected-Model')  # Frontend model name (e.g., 'gpt-5.2', 'claude-opus-4.5', 'gemini-3-flash')

    nodes = request.data.get('nodes', [])
    edges = request.data.get('edges', [])

    if not nodes:
        return Response({
            'suggestions': ['Start by adding an Input node to define your model input.']
        })

    try:
        # Initialize Universal AI service with API key and model (auto-detects provider)
        ai_service = UniversalAIFactory.detect_and_create_service(
            api_key=api_key,
            model=selected_model
        )
        provider_name = UniversalAIFactory.get_provider_name(api_key)

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


@api_view(['GET'])
def get_environment_info(request):
    """
    Get environment configuration information.

    Endpoint: GET /api/environment

    Response:
    {
        'environment': 'PROD' | 'DEV' | 'LOCAL',
        'isProduction': boolean,
        'requiresApiKey': boolean,
        'provider': 'Gemini' | 'Claude'
    }
    """
    from django.conf import settings

    environment = AIServiceFactory.get_environment_mode()
    is_production = getattr(settings, 'IS_PRODUCTION', True)
    requires_api_key = getattr(settings, 'REQUIRES_USER_API_KEY', True)
    provider = AIServiceFactory.get_provider_name()

    return Response({
        'environment': environment,
        'isProduction': is_production,
        'requiresApiKey': requires_api_key,
        'provider': provider
    })


@api_view(['POST'])
def validate_api_key(request):
    """
    Validate an API key and detect its provider.

    Endpoint: POST /api/validate-key

    Request body:
    {
        "apiKey": str
    }

    Response:
    {
        "valid": boolean,
        "provider": str | null,
        "displayName": str | null,
        "availableModels": int,
        "models": [str],
        "isFreeTier": boolean,
        "message": str
    }
    """
    api_key = request.data.get('apiKey')

    if not api_key:
        return Response({
            'valid': False,
            'provider': None,
            'displayName': None,
            'availableModels': 0,
            'models': [],
            'isFreeTier': False,
            'message': 'No API key provided'
        }, status=status.HTTP_400_BAD_REQUEST)

    # Detect provider
    provider = APIKeyDetector.detect_provider(api_key)

    if not provider:
        return Response({
            'valid': False,
            'provider': None,
            'displayName': None,
            'availableModels': 0,
            'models': [],
            'isFreeTier': False,
            'message': 'Unknown API key format. Supported: OpenRouter (sk-or-v1-...), Google AI (AIza...), OpenAI (sk-proj-... or sk-...), Anthropic (sk-ant-api03-...)'
        })

    # Get provider info
    provider_info = APIKeyDetector.get_provider_info(provider)
    available_models = APIKeyDetector.get_available_models(provider)

    return Response({
        'valid': True,
        'provider': provider,
        'displayName': provider_info.display_name,
        'availableModels': provider_info.models_count,
        'models': available_models,
        'isFreeTier': provider_info.is_free_tier,
        'message': f'Valid {provider_info.display_name} API key detected'
    })


@api_view(['POST'])
def get_available_models_for_key(request):
    """
    Get list of models available for a specific API key.

    Endpoint: POST /api/available-models

    Request body:
    {
        "apiKey": str
    }

    Response:
    {
        "provider": str,
        "models": [str],
        "defaultModel": str
    }
    """
    api_key = request.data.get('apiKey')

    if not api_key:
        return Response({
            'error': 'No API key provided'
        }, status=status.HTTP_400_BAD_REQUEST)

    # Detect provider
    provider = APIKeyDetector.detect_provider(api_key)

    if not provider:
        return Response({
            'error': 'Unknown API key format'
        }, status=status.HTTP_400_BAD_REQUEST)

    # Get available models
    models = APIKeyDetector.get_available_models(provider)
    provider_info = APIKeyDetector.get_provider_info(provider)

    # Determine default model
    default_model = models[0] if models else None

    return Response({
        'provider': provider,
        'displayName': provider_info.display_name,
        'models': models,
        'defaultModel': default_model,
        'isFreeTier': provider_info.is_free_tier
    })
