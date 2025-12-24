"""
API Key Detection Utility - Auto-detect provider from API key format.
Supports OpenRouter, Google AI, OpenAI, and Anthropic.

References:
- OpenRouter: https://openrouter.ai/docs/api/reference/authentication
- Google AI: https://ai.google.dev/gemini-api/docs/api-key
- OpenAI: https://platform.openai.com/docs/api-reference/project-api-keys
- Anthropic: https://docs.claude.com/en/api/admin-api/apikeys/get-api-key
"""
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class ProviderInfo:
    """Information about a detected API provider."""
    name: str
    display_name: str
    key_prefix: str
    models_count: int
    is_free_tier: bool
    models: List[str]


class APIKeyDetector:
    """Detects which AI provider an API key belongs to based on its format."""

    # Provider configurations
    PROVIDERS = {
        'openrouter': ProviderInfo(
            name='openrouter',
            display_name='OpenRouter',
            key_prefix='sk-or-v1-',
            models_count=32,
            is_free_tier=True,  # Has 14 truly free models ($0/$0 with :free suffix)
            models=[
                # Flagship models via OpenRouter (10 models - all PAID)
                'gemini-3-flash', 'gemini-3-pro', 'gemini-2.5-flash', 'gemini-2.5-pro',
                'gpt-5.2', 'gpt-4o', 'gpt-4o-mini',
                'claude-opus-4.5', 'claude-sonnet-4.5', 'claude-haiku-4.5',
                # Truly FREE OpenRouter models (14 models - $0/$0 with :free suffix)
                'llama-4-maverick', 'llama-4-scout', 'llama-3.3-70b',
                'gemini-2.0-flash-exp', 'gemini-2.5-pro-exp',
                'mistral-small-3.1', 'devstral-2',
                'deepseek-chat-v3', 'deepseek-r1-zero',
                'nemotron-nano-8b', 'mimo-v2-flash', 'kat-coder-pro',
                'optimus-alpha', 'quasar-alpha',
                # Affordable PAID models (8 models)
                'llama-3.1-405b', 'llama-3.1-70b',
                'deepseek-v3', 'deepseek-coder-v2',
                'claude-3.5-sonnet', 'claude-3.5-haiku',
                'qwen-2.5-72b', 'mistral-large-2'
            ]
        ),
        'google': ProviderInfo(
            name='google',
            display_name='Google AI (Gemini)',
            key_prefix='AIza',
            models_count=4,
            is_free_tier=True,  # Free on direct Google AI API, paid on OpenRouter
            models=['gemini-3-flash', 'gemini-3-pro', 'gemini-2.5-flash', 'gemini-2.5-pro']
        ),
        'openai': ProviderInfo(
            name='openai',
            display_name='OpenAI',
            key_prefix='sk-proj-',  # Also sk-svcacct-, sk- (legacy)
            models_count=3,
            is_free_tier=False,
            models=['gpt-5.2', 'gpt-4o', 'gpt-4o-mini']
        ),
        'anthropic': ProviderInfo(
            name='anthropic',
            display_name='Anthropic (Claude)',
            key_prefix='sk-ant-api03-',
            models_count=3,
            is_free_tier=False,
            models=['claude-opus-4.5', 'claude-sonnet-4.5', 'claude-haiku-4.5']
        )
    }

    @staticmethod
    def detect_provider(api_key: str) -> Optional[str]:
        """
        Detect which provider an API key belongs to.

        Args:
            api_key: The API key to analyze

        Returns:
            Provider name ('openrouter', 'google', 'openai', 'anthropic') or None if unknown

        Examples:
            >>> detect_provider('sk-or-v1-abc123...')
            'openrouter'
            >>> detect_provider('AIzaSyAbc123...')
            'google'
            >>> detect_provider('sk-proj-abc123...')
            'openai'
            >>> detect_provider('sk-ant-api03-abc123...')
            'anthropic'
        """
        if not api_key or not isinstance(api_key, str):
            return None

        api_key = api_key.strip()

        # OpenRouter: sk-or-v1-{64 hex chars}
        if api_key.startswith('sk-or-v1-'):
            return 'openrouter'

        # Google AI: AIza{~35 chars} (39 total)
        if api_key.startswith('AIza') and len(api_key) == 39:
            return 'google'

        # Anthropic: sk-ant-api03-{48 chars}
        if api_key.startswith('sk-ant-api03-'):
            return 'anthropic'

        # OpenAI: sk-proj-, sk-svcacct-, or legacy sk-
        # Check for newer formats first
        if api_key.startswith('sk-proj-') or api_key.startswith('sk-svcacct-'):
            return 'openai'

        # Legacy OpenAI key: sk-{alphanumeric}
        # Be careful not to conflict with Anthropic (sk-ant-) or OpenRouter (sk-or-)
        if api_key.startswith('sk-') and not api_key.startswith('sk-ant-') and not api_key.startswith('sk-or-'):
            return 'openai'

        return None

    @staticmethod
    def get_provider_info(provider: str) -> Optional[ProviderInfo]:
        """
        Get detailed information about a provider.

        Args:
            provider: Provider name

        Returns:
            ProviderInfo object or None if provider not found
        """
        return APIKeyDetector.PROVIDERS.get(provider)

    @staticmethod
    def validate_key_format(api_key: str, provider: str) -> Dict[str, any]:
        """
        Validate that an API key matches the expected format for a provider.

        Args:
            api_key: The API key to validate
            provider: Expected provider name

        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'provider': str,
                'message': str,
                'suggestions': List[str]
            }
        """
        detected = APIKeyDetector.detect_provider(api_key)

        if not detected:
            return {
                'valid': False,
                'provider': None,
                'message': 'Unknown API key format',
                'suggestions': [
                    'Check for typos or extra spaces',
                    'Verify you copied the complete key',
                    'Ensure the key starts with the correct prefix'
                ]
            }

        if provider and detected != provider:
            provider_info = APIKeyDetector.get_provider_info(detected)
            expected_info = APIKeyDetector.get_provider_info(provider)

            return {
                'valid': False,
                'provider': detected,
                'message': f'This appears to be a {provider_info.display_name} key, but you selected {expected_info.display_name}',
                'suggestions': [
                    f'Switch to {provider_info.display_name} to use this key',
                    f'Or get a {expected_info.display_name} key instead'
                ]
            }

        provider_info = APIKeyDetector.get_provider_info(detected)

        return {
            'valid': True,
            'provider': detected,
            'message': f'Valid {provider_info.display_name} API key detected',
            'suggestions': []
        }

    @staticmethod
    def get_available_models(provider: str) -> List[str]:
        """
        Get list of models available for a provider.

        Args:
            provider: Provider name

        Returns:
            List of model names
        """
        provider_info = APIKeyDetector.get_provider_info(provider)
        return provider_info.models if provider_info else []

    @staticmethod
    def is_model_available(model: str, provider: str) -> bool:
        """
        Check if a model is available for a given provider.

        Args:
            model: Model name (e.g., 'gpt-5.2')
            provider: Provider name

        Returns:
            True if model is available, False otherwise
        """
        available_models = APIKeyDetector.get_available_models(provider)
        return model in available_models
