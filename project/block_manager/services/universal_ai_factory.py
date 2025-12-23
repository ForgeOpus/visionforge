"""
Universal AI Service Factory - Auto-detect and create appropriate AI service.
Supports OpenRouter, Google AI, OpenAI, and Anthropic with automatic provider detection.
"""
import os
from typing import Optional, Union
from django.conf import settings

from block_manager.services.api_key_detector import APIKeyDetector
from block_manager.services.openrouter_service import OpenRouterService
from block_manager.services.gemini_service import GeminiChatService
from block_manager.services.openai_service import OpenAIChatService
from block_manager.services.claude_service import ClaudeChatService
from block_manager.services.model_config import get_model_identifier


class UniversalAIFactory:
    """
    Universal factory that auto-detects API key provider and creates the appropriate service.

    Features:
    - Automatic provider detection from API key format
    - Support for multiple providers (OpenRouter, Google, OpenAI, Anthropic)
    - Backward compatibility with existing code
    - Smart model validation
    """

    @staticmethod
    def requires_user_api_key() -> bool:
        """
        Check if user-provided API keys are required.

        Returns:
            True if ENVIRONMENT is PROD or missing (BYOK mode)
            False if ENVIRONMENT is DEV/LOCAL (server keys mode)
        """
        return getattr(settings, 'REQUIRES_USER_API_KEY', True)

    @staticmethod
    def detect_and_create_service(
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider_hint: Optional[str] = None,
        **kwargs
    ) -> Union[OpenRouterService, GeminiChatService, OpenAIChatService, ClaudeChatService]:
        """
        Automatically detect provider from API key and create appropriate service.

        Args:
            api_key: User-provided API key (any provider)
            model: Frontend model name (e.g., 'gpt-5.2', 'claude-opus-4.5', 'gemini-3-flash')
            provider_hint: Optional provider name if user explicitly selected one
            **kwargs: Additional legacy parameters

        Returns:
            Appropriate AI service instance (OpenRouter, Gemini, OpenAI, or Claude)

        Raises:
            ValueError: If API key is invalid or provider cannot be determined
        """
        requires_user_key = UniversalAIFactory.requires_user_api_key()

        # Handle server-side keys (DEV mode)
        if not requires_user_key:
            return UniversalAIFactory._create_server_side_service(model)

        # Require API key in PROD mode
        if not api_key:
            raise ValueError(
                "API key is required. Please provide an API key from any supported provider: "
                "OpenRouter, Google AI, OpenAI, or Anthropic."
            )

        # Detect provider from API key
        detected_provider = APIKeyDetector.detect_provider(api_key)

        if not detected_provider:
            raise ValueError(
                "Unable to detect API provider from key format. "
                "Supported formats: OpenRouter (sk-or-v1-...), Google AI (AIza...), "
                "OpenAI (sk-proj-... or sk-...), Anthropic (sk-ant-api03-...)"
            )

        # Validate provider hint matches detection
        if provider_hint and provider_hint != detected_provider:
            provider_info = APIKeyDetector.get_provider_info(detected_provider)
            raise ValueError(
                f"API key appears to be for {provider_info.display_name}, "
                f"but {provider_hint} was specified. Please check your key."
            )

        # Map frontend model name to API identifier
        model_identifier = None
        if model:
            # For OpenRouter, use model config mapping
            if detected_provider == 'openrouter':
                try:
                    model_identifier = get_model_identifier(model)
                except ValueError as e:
                    # Only catch model config errors for OpenRouter, not validation errors
                    print(f"Warning: {e}. Using default model.")
            else:
                # For direct providers, validate model availability (strict validation)
                if not APIKeyDetector.is_model_available(model, detected_provider):
                    available = APIKeyDetector.get_available_models(detected_provider)
                    provider_info = APIKeyDetector.get_provider_info(detected_provider)
                    raise ValueError(
                        f"Model '{model}' is not available with {provider_info.display_name}. "
                        f"Available models: {', '.join(available)}"
                    )
                # Model name is valid, use it as-is for direct providers
                model_identifier = model

        # Create appropriate service based on detected provider
        return UniversalAIFactory._create_service_for_provider(
            provider=detected_provider,
            api_key=api_key,
            model=model_identifier
        )

    @staticmethod
    def _create_server_side_service(model: Optional[str]) -> Union[OpenRouterService, GeminiChatService, OpenAIChatService, ClaudeChatService]:
        """
        Create AI service using server-side keys (DEV mode).

        Args:
            model: Frontend model name

        Returns:
            AI service using server-side configuration

        Raises:
            ValueError: If no server-side keys are configured
        """
        # Map model to OpenRouter identifier if provided
        model_identifier = None
        if model:
            try:
                model_identifier = get_model_identifier(model)
            except ValueError:
                pass

        # Try OpenRouter first (unified access)
        if os.getenv('OPENROUTER_API_KEY'):
            return OpenRouterService(model=model_identifier)

        # Fall back to direct providers
        if os.getenv('GEMINI_API_KEY'):
            return GeminiChatService()

        if os.getenv('OPENAI_API_KEY'):
            return OpenAIChatService()

        if os.getenv('ANTHROPIC_API_KEY'):
            return ClaudeChatService()

        raise ValueError(
            "No server-side API keys configured. "
            "Set OPENROUTER_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in environment."
        )

    @staticmethod
    def _create_service_for_provider(
        provider: str,
        api_key: str,
        model: Optional[str]
    ) -> Union[OpenRouterService, GeminiChatService, OpenAIChatService, ClaudeChatService]:
        """
        Create service instance for specific provider.

        Args:
            provider: Provider name ('openrouter', 'google', 'openai', 'anthropic')
            api_key: API key for the provider
            model: Model identifier (already mapped for OpenRouter)

        Returns:
            Appropriate service instance

        Raises:
            ValueError: If provider is unknown
        """
        if provider == 'openrouter':
            return OpenRouterService(api_key=api_key, model=model)

        elif provider == 'google':
            # GeminiChatService doesn't use model parameter in __init__
            # Model selection happens at chat() time
            return GeminiChatService(api_key=api_key)

        elif provider == 'openai':
            return OpenAIChatService(api_key=api_key)

        elif provider == 'anthropic':
            return ClaudeChatService(api_key=api_key)

        else:
            raise ValueError(f"Unknown provider: {provider}")

    @staticmethod
    def get_provider_name(api_key: Optional[str] = None) -> str:
        """
        Get the name of the AI provider being used.

        Args:
            api_key: Optional API key to detect provider from

        Returns:
            Provider display name
        """
        if api_key:
            provider = APIKeyDetector.detect_provider(api_key)
            if provider:
                provider_info = APIKeyDetector.get_provider_info(provider)
                return provider_info.display_name

        # Default to OpenRouter if no key or detection fails
        return 'Universal AI (Multi-Provider)'

    @staticmethod
    def get_environment_mode() -> str:
        """
        Get current environment mode.

        Returns:
            'PROD', 'DEV', 'LOCAL', etc.
        """
        return os.getenv('ENVIRONMENT', 'PROD')

    @staticmethod
    def validate_api_key(api_key: str, provider: Optional[str] = None) -> dict:
        """
        Validate an API key format.

        Args:
            api_key: API key to validate
            provider: Optional expected provider

        Returns:
            Validation result dictionary
        """
        return APIKeyDetector.validate_key_format(api_key, provider)

    @staticmethod
    def get_available_models_for_key(api_key: str) -> list:
        """
        Get list of models available for a given API key.

        Args:
            api_key: API key to check

        Returns:
            List of available model names
        """
        provider = APIKeyDetector.detect_provider(api_key)
        if not provider:
            return []

        return APIKeyDetector.get_available_models(provider)


# Backward compatibility alias
AIServiceFactory = UniversalAIFactory
