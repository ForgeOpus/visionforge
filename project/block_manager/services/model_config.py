"""
Model configuration mapping - maps frontend model names to provider-specific identifiers.
Supports both OpenRouter (unified) and direct provider APIs.
"""

# Gemini model mapping for OpenRouter (frontend name -> OpenRouter identifier)
# Latest models as of December 2025 - Available on free tier
# Reference: https://openrouter.ai/google
GEMINI_MODELS = {
    'gemini-3-flash': 'google/gemini-3-flash-preview',  # Newest - Dec 17, 2025 (free tier)
    'gemini-3-pro': 'google/gemini-3-pro-preview',  # Nov 18, 2025 - multimodal (free tier)
    'gemini-2.5-flash': 'google/gemini-2.5-flash',  # Stable - intelligent speed (free tier)
    'gemini-2.5-pro': 'google/gemini-2.5-pro',  # State-of-the-art thinking model (free tier)
}

# Gemini model mapping for direct Google AI API (frontend name -> Google API identifier)
# Reference: https://ai.google.dev/gemini-api/docs/models
GOOGLE_AI_MODELS = {
    'gemini-3-flash': 'gemini-3-flash-preview',  # Newest - Dec 2025
    'gemini-3-pro': 'gemini-3-pro-preview',  # Nov 2025 - best multimodal
    'gemini-2.5-flash': 'gemini-2.5-flash',  # Stable - fast & reliable
    'gemini-2.5-pro': 'gemini-2.5-pro',  # Advanced thinking model
}

# OpenAI model mapping (frontend name -> OpenRouter identifier)
# Latest models as of December 2025 - Pay-as-you-go
# Reference: https://openrouter.ai/openai
OPENAI_MODELS = {
    'gpt-5.2': 'openai/gpt-5.2',  # Newest flagship - Dec 11, 2025
    'gpt-4o': 'openai/gpt-4o',  # Stable GPT-4 omni model (production-ready)
    'gpt-4o-mini': 'openai/gpt-4o-mini',  # Fast and cost-effective
}

# Claude model mapping (frontend name -> OpenRouter identifier)
# Latest models as of December 2025 - Pay-as-you-go
# Reference: https://openrouter.ai/anthropic
CLAUDE_MODELS = {
    'claude-opus-4.5': 'anthropic/claude-opus-4.5',  # Newest - Nov 24, 2025 (best for coding)
    'claude-sonnet-4.5': 'anthropic/claude-sonnet-4.5',  # Sept 29, 2025 (balanced)
    'claude-haiku-4.5': 'anthropic/claude-haiku-4.5',  # Oct 15, 2025 (fast)
}

# Combined model mapping
MODEL_IDENTIFIERS = {
    **GEMINI_MODELS,
    **OPENAI_MODELS,
    **CLAUDE_MODELS,
}

def get_model_identifier(frontend_model: str) -> str:
    """
    Get the OpenRouter model identifier for a frontend model name.

    Args:
        frontend_model: Model name from frontend (e.g., 'gpt-5', 'claude-opus-4.5')

    Returns:
        OpenRouter model identifier (e.g., 'openai/gpt-5', 'anthropic/claude-opus-4.5')

    Raises:
        ValueError: If model name is not recognized
    """
    if frontend_model not in MODEL_IDENTIFIERS:
        raise ValueError(f"Unknown model: {frontend_model}")

    return MODEL_IDENTIFIERS[frontend_model]

def get_google_ai_model(frontend_model: str) -> str:
    """
    Get the Google AI API model identifier for a frontend model name.

    Args:
        frontend_model: Model name from frontend (e.g., 'gemini-3-flash', 'gemini-2.5-pro')

    Returns:
        Google AI API model identifier (e.g., 'gemini-3-flash-preview')

    Raises:
        ValueError: If model name is not recognized or not a Gemini model
    """
    if frontend_model not in GOOGLE_AI_MODELS:
        available = ', '.join(GOOGLE_AI_MODELS.keys())
        raise ValueError(
            f"Unknown Google AI model: {frontend_model}. "
            f"Available models: {available}"
        )

    return GOOGLE_AI_MODELS[frontend_model]
