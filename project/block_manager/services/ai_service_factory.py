"""
AI Service Factory - Provider selection for Gemini or Claude with BYOK support.
"""
import os
from typing import Union, Optional
from django.conf import settings
from block_manager.services.gemini_service import GeminiChatService
from block_manager.services.claude_service import ClaudeChatService


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
        anthropic_api_key: Optional[str] = None
    ) -> Union[GeminiChatService, ClaudeChatService]:
        """
        Create and return the configured AI service.

        In PROD mode (or missing): uses user-provided API keys (BYOK)
        In DEV mode: uses server-side API keys from environment

        Args:
            gemini_api_key: User-provided Gemini API key (PROD mode only)
            anthropic_api_key: User-provided Anthropic API key (PROD mode only)

        Returns:
            GeminiChatService or ClaudeChatService based on AI_PROVIDER

        Raises:
            ValueError: If API keys are missing or provider is invalid
        """
        provider = os.getenv('AI_PROVIDER', 'gemini').lower()
        requires_user_key = AIServiceFactory.requires_user_api_key()

        if provider == 'gemini':
            if requires_user_key:
                # PROD mode: require user-provided key
                if not gemini_api_key:
                    raise ValueError("Gemini API key is required. Please provide your API key.")
                return GeminiChatService(api_key=gemini_api_key)
            else:
                # DEV mode: use server-side key
                return GeminiChatService()

        elif provider == 'claude':
            if requires_user_key:
                # PROD mode: require user-provided key
                if not anthropic_api_key:
                    raise ValueError("Anthropic API key is required. Please provide your API key.")
                return ClaudeChatService(api_key=anthropic_api_key)
            else:
                # DEV mode: use server-side key
                return ClaudeChatService()
        else:
            raise ValueError(
                f"Invalid AI_PROVIDER: '{provider}'. Must be 'gemini' or 'claude'."
            )

    @staticmethod
    def get_provider_name() -> str:
        """
        Get the name of the current AI provider.

        Returns:
            'Gemini' or 'Claude'
        """
        provider = os.getenv('AI_PROVIDER', 'gemini').lower()
        return provider.capitalize()

    @staticmethod
    def get_environment_mode() -> str:
        """
        Get current environment mode.

        Returns:
            'PROD', 'DEV', 'LOCAL', etc.
        """
        return getattr(settings, 'ENVIRONMENT', 'PROD')
