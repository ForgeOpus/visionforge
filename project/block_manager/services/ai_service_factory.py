"""
AI Service Factory - Provider selection for Gemini or Claude.
"""
import os
from typing import Union
from block_manager.services.gemini_service import GeminiChatService
from block_manager.services.claude_service import ClaudeChatService


class AIServiceFactory:
    """Factory to create the appropriate AI service based on configuration."""

    @staticmethod
    def create_service() -> Union[GeminiChatService, ClaudeChatService]:
        """
        Create and return the configured AI service.

        Returns:
            GeminiChatService or ClaudeChatService based on AI_PROVIDER env variable

        Raises:
            ValueError: If AI_PROVIDER is invalid or required API key is missing
        """
        provider = os.getenv('AI_PROVIDER', 'gemini').lower()

        if provider == 'gemini':
            return GeminiChatService()
        elif provider == 'claude':
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
