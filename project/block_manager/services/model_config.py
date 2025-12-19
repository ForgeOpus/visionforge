"""
Model configuration mapping - maps frontend model names to API model identifiers.
"""

# Gemini model mapping (frontend name -> API identifier)
GEMINI_MODELS = {
    'gemini-3-flash': 'gemini-3-flash',
    'gemini-2.5-flash': 'gemini-2.5-flash',
    'gemini-2.0-flash': 'gemini-2.0-flash',
}

# OpenAI model mapping (frontend name -> API identifier)
OPENAI_MODELS = {
    'gpt-5': 'gpt-5',  # Will map to actual identifier when available
    'gpt-4.1': 'gpt-4.1',  # Coding specialist model
    'gpt-4o': 'gpt-4o',
    'gpt-4o-mini': 'gpt-4o-mini',
}

# Claude model mapping (frontend name -> API identifier)
# Using dated model identifiers as per Anthropic API
CLAUDE_MODELS = {
    'claude-opus-4.5': 'claude-opus-4-20250514',  # Latest Opus 4.5
    'claude-sonnet-4.5': 'claude-sonnet-4-5-20250929',  # Sonnet 4.5 from Sept 2025
    'claude-haiku-4.5': 'claude-haiku-4-5',  # Haiku 4.5
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
