"""
AI Service Factory - OpenRouter unified access to all AI providers.
"""
import os
from typing import Optional
from django.conf import settings
from block_manager.services.openrouter_service import OpenRouterService
from block_manager.services.model_config import get_model_identifier


class AIServiceFactory:
    """Factory to create OpenRouter AI service with unified API access."""

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
    def create_service(
        openrouter_api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs  # Accept legacy parameters for backwards compatibility
    ) -> OpenRouterService:
        """
        Create and return OpenRouter AI service.

        OpenRouter provides unified access to Gemini, GPT, and Claude models
        through a single API key, simplifying integration.

        In PROD mode (or missing): uses user-provided API key (BYOK)
        In DEV mode: uses server-side API key from environment

        Args:
            openrouter_api_key: User-provided OpenRouter API key (PROD mode only)
            model: Frontend model name (e.g., 'gpt-5.2', 'claude-opus-4.5', 'gemini-3-flash')

        Returns:
            OpenRouterService configured with the specified model

        Raises:
            ValueError: If API key is missing in PROD mode
        """
        requires_user_key = AIServiceFactory.requires_user_api_key()

        # Map frontend model name to OpenRouter identifier if provided
        model_identifier = None
        if model:
            try:
                model_identifier = get_model_identifier(model)
            except ValueError as e:
                # If model mapping fails, log and continue with default
                print(f"Warning: {e}. Using default model.")

        if requires_user_key:
            # PROD mode: require user-provided OpenRouter key
            if not openrouter_api_key:
                raise ValueError("OpenRouter API key is required. Please provide your API key.")
            return OpenRouterService(api_key=openrouter_api_key, model=model_identifier)
        else:
            # DEV mode: use server-side OpenRouter key
            return OpenRouterService(model=model_identifier)

    @staticmethod
    def get_provider_name() -> str:
        """
        Get the name of the unified AI provider.

        Returns:
            'OpenRouter' (unified access to all AI models)
        """
        return 'OpenRouter'

    @staticmethod
    def get_environment_mode() -> str:
        """
        Get current environment mode.

        Returns:
            'PROD', 'DEV', 'LOCAL', etc.
        """
        return os.getenv('ENVIRONMENT', 'PROD')
