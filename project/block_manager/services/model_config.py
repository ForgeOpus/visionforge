"""
Model configuration mapping - maps frontend model names to OpenRouter model identifiers.
OpenRouter provides unified access to all AI providers through a single API.
"""

# Gemini model mapping (frontend name -> OpenRouter identifier)
# Latest models as of December 2025 - Available on free tier
GEMINI_MODELS = {
    'gemini-3-flash': 'google/gemini-3-flash-preview',  # Newest - Dec 17, 2025 (free tier)
    'gemini-3-pro': 'google/gemini-3-pro-preview',  # Nov 18, 2025 - multimodal (free tier)
    'gemini-2.5-flash': 'google/gemini-2.5-flash-image',  # Stable - intelligent speed (free tier)
    'gemini-2.5-pro': 'google/gemini-2.5-flash-image',  # Using flash-image variant (free tier)
}

# OpenAI model mapping (frontend name -> OpenRouter identifier)
# Latest models as of December 2025 - Pay-as-you-go
OPENAI_MODELS = {
    'gpt-5.2': 'openai/gpt-5.2',  # Newest flagship - Dec 11, 2025
    'gpt-4o': 'openai/chatgpt-4o-latest',  # Latest GPT-4 omni model
    'gpt-4o-mini': 'openai/gpt-4o-mini',  # Fast and cost-effective
}

# Claude model mapping (frontend name -> OpenRouter identifier)
# Latest models as of December 2025 - Pay-as-you-go
CLAUDE_MODELS = {
    'claude-opus-4.5': 'anthropic/claude-opus-4.5',  # Newest - Nov 24, 2025
    'claude-sonnet-4.5': 'anthropic/claude-sonnet-4.5',  # Sept 29, 2025
    'claude-haiku-4.5': 'anthropic/claude-haiku-4.5',  # Oct 15, 2025
}

# Combined model mapping
MODEL_IDENTIFIERS = {
    **GEMINI_MODELS,
    **OPENAI_MODELS,
    **CLAUDE_MODELS,
}

def get_model_identifier(frontend_model: str) -> str:
    """
    Get the API model identifier for a frontend model name.

    Args:
        frontend_model: Model name from frontend (e.g., 'gpt-5', 'claude-opus-4.5')

    Returns:
        API model identifier (e.g., 'gpt-5', 'claude-opus-4-20250514')

    Raises:
        ValueError: If model name is not recognized
    """
    if frontend_model not in MODEL_IDENTIFIERS:
        raise ValueError(f"Unknown model: {frontend_model}")

    return MODEL_IDENTIFIERS[frontend_model]
