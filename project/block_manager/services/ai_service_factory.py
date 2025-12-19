"""
AI Service Factory - Provider selection for Gemini, Claude, or OpenAI with BYOK support.
"""
import os
from typing import Union, Optional
from django.conf import settings
from block_manager.services.gemini_service import GeminiChatService
from block_manager.services.claude_service import ClaudeChatService
from block_manager.services.openai_service import OpenAIChatService
from block_manager.services.model_config import get_model_identifier


class AIServiceFactory:
    """Factory to create the appropriate AI service based on configuration."""

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
        gemini_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        active_provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> Union[GeminiChatService, ClaudeChatService, OpenAIChatService]:
        """
        Create and return the configured AI service.

        In PROD mode (or missing): uses user-provided API keys (BYOK)
        In DEV mode: uses server-side API keys from environment

        Args:
            gemini_api_key: User-provided Gemini API key (PROD mode only)
            anthropic_api_key: User-provided Anthropic API key (PROD mode only)
            openai_api_key: User-provided OpenAI API key (PROD mode only)
            active_provider: User's chosen provider ('gemini', 'claude', or 'openai')
            model: Frontend model name (e.g., 'gpt-5', 'claude-opus-4.5')

        Returns:
            GeminiChatService, ClaudeChatService, or OpenAIChatService based on active_provider

        Raises:
            ValueError: If API keys are missing or provider is invalid
        """
        # Use active_provider if provided, otherwise fall back to AI_PROVIDER env var
        provider = (active_provider or os.getenv('AI_PROVIDER', 'gemini')).lower()
        requires_user_key = AIServiceFactory.requires_user_api_key()

        # Map frontend model name to API identifier if provided
        model_identifier = None
        if model:
            try:
                model_identifier = get_model_identifier(model)
            except ValueError as e:
                # If model mapping fails, log and continue with default
                print(f"Warning: {e}. Using default model.")

        if provider == 'gemini':
            if requires_user_key:
                # PROD mode: require user-provided key
                if not gemini_api_key:
                    raise ValueError("Gemini API key is required. Please provide your API key.")
                return GeminiChatService(api_key=gemini_api_key, model=model_identifier)
            else:
                # DEV mode: use server-side key
                return GeminiChatService(model=model_identifier)

        elif provider == 'claude':
            if requires_user_key:
                # PROD mode: require user-provided key
                if not anthropic_api_key:
                    raise ValueError("Anthropic API key is required. Please provide your API key.")
                return ClaudeChatService(api_key=anthropic_api_key, model=model_identifier)
            else:
                # DEV mode: use server-side key
                return ClaudeChatService(model=model_identifier)

        elif provider == 'openai':
            if requires_user_key:
                # PROD mode: require user-provided key
                if not openai_api_key:
                    raise ValueError("OpenAI API key is required. Please provide your API key.")
                return OpenAIChatService(api_key=openai_api_key, model=model_identifier)
            else:
                # DEV mode: use server-side key
                return OpenAIChatService(model=model_identifier)
        else:
            raise ValueError(
                f"Invalid provider: '{provider}'. Must be 'gemini', 'claude', or 'openai'."
            )

    @staticmethod
    def get_provider_name() -> str:
        """
        Get the name of the current AI provider.

        Returns:
            'Gemini', 'Claude', or 'OpenAI'
        """
        provider = os.getenv('AI_PROVIDER', 'gemini').lower()
        if provider == 'openai':
            return 'OpenAI'
        return provider.capitalize()

    @staticmethod
    def get_environment_mode() -> str:
        """
        Get current environment mode.

        Returns:
            'PROD', 'DEV', 'LOCAL', etc.
        """
        return os.getenv('ENVIRONMENT', 'PROD')
