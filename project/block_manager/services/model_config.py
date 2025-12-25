"""
Model configuration mapping - maps frontend model names to provider-specific identifiers.
Supports both OpenRouter (unified) and direct provider APIs.
"""

# Gemini model mapping for OpenRouter (frontend name -> OpenRouter identifier)
# Latest models as of December 2025 - PAID on OpenRouter (Free on direct Google AI API)
# Reference: https://openrouter.ai/google
# NOTE: gemini-3-flash and gemini-2.5-flash are in FREE models list (verified working)
GEMINI_MODELS = {
    'gemini-3-pro': 'google/gemini-3-pro-preview-20251117',  # Nov 18, 2025 - multimodal ($2/M input)
    'gemini-2.5-pro': 'google/gemini-2.5-pro',  # Advanced thinking ($1.25/M input)
}

# Gemini model mapping for direct Google AI API (frontend name -> Google API identifier)
# Reference: https://ai.google.dev/gemini-api/docs/models
GOOGLE_AI_MODELS = {
    'gemini-3-flash': 'gemini-3-flash-preview',  # Newest - Dec 2025
    'gemini-3-pro': 'gemini-3-pro-preview',  # Nov 2025 - best multimodal
    'gemini-2.5-flash': 'gemini-2.5-flash',  # Stable - fast & reliable
    'gemini-2.5-pro': 'gemini-2.5-pro',  # Advanced thinking model
}

# Claude model mapping for direct Anthropic API (frontend name -> Anthropic API identifier)
# Reference: https://github.com/anthropics/anthropic-sdk-python
# Note: Anthropic uses HYPHENS (4-5) not periods (4.5)
ANTHROPIC_AI_MODELS = {
    'claude-opus-4.5': 'claude-opus-4-5',  # Best for coding - Nov 2025
    'claude-sonnet-4.5': 'claude-sonnet-4-5',  # Balanced performance - Sept 2025
    'claude-haiku-4.5': 'claude-haiku-4-5',  # Fast, near-frontier - Oct 2025
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
    'claude-opus-4.5': 'anthropic/claude-4.5-opus-20251124',  # Newest - Nov 24, 2025 (best for coding)
    'claude-sonnet-4.5': 'anthropic/claude-4.5-sonnet-20250929',  # Sept 29, 2025 (balanced)
    'claude-haiku-4.5': 'anthropic/claude-4.5-haiku-20251001',  # Oct 15, 2025 (fast)
}

# OpenRouter-specific FREE models (from official free models collection)
# Reference: https://openrouter.ai/collections/free-models
# NOTE: Most free models don't use :free suffix - use exact identifiers from OpenRouter
# VERIFIED WORKING - Only includes models that successfully respond (404 errors removed)
OPENROUTER_FREE_MODELS = {
    # Meta Llama FREE models (VERIFIED WORKING)
    'llama-3.3-70b': 'meta-llama/llama-3.3-70b-instruct',  # 70B capable - FREE
    'llama-3.1-70b': 'meta-llama/llama-3.1-70b-instruct',  # 70B - FREE
    'llama-3.1-8b': 'meta-llama/llama-3.1-8b-instruct',  # 8B efficient - FREE

    # Google Gemini FREE models (VERIFIED WORKING)
    'gemini-2.0-flash': 'google/gemini-2.0-flash-001',  # Gemini 2.0 Flash - FREE
    'gemini-3-flash': 'google/gemini-3-flash-preview-20251217',  # Gemini 3 Flash - FREE
    'gemini-2.5-flash': 'google/gemini-2.5-flash',  # Gemini 2.5 Flash - FREE

    # Mistral FREE models (VERIFIED WORKING)
    'mistral-nemo': 'mistralai/mistral-nemo',  # Mistral Nemo - FREE

    # DeepSeek FREE models (VERIFIED WORKING)
    'deepseek-chat-v3': 'deepseek/deepseek-chat-v3-0324',  # Chat V3 - FREE
    'deepseek-chat-v3.1': 'deepseek/deepseek-chat-v3.1',  # Chat V3.1 - FREE
    'deepseek-v3.2': 'deepseek/deepseek-v3.2-20251201',  # DeepSeek V3.2 - FREE

    # Other FREE models (VERIFIED WORKING)
    'nemotron-nano-30b': 'nvidia/nemotron-3-nano-30b-a3b',  # NVIDIA 30B - FREE
}

# OpenRouter-specific PAID models (affordable, not flagships)
# Reference: https://openrouter.ai/pricing
OPENROUTER_PAID_MODELS = {
    # Affordable Llama models (paid versions)
    'llama-3.1-405b': 'meta-llama/llama-3.1-405b-instruct',  # $3.50/M - powerful 405B

    # Affordable DeepSeek models (paid versions - not on free list)
    'deepseek-v3': 'deepseek/deepseek-chat',  # $0.30/M - latest flagship (DeepSeek V3)

    # Older Claude (still very good, cheaper than 4.5)
    'claude-3.5-sonnet': 'anthropic/claude-3.5-sonnet',  # $3/M - very capable
    'claude-3.5-haiku': 'anthropic/claude-3.5-haiku',  # $0.80/M - cheapest Claude

    # Alternative affordable providers
    'qwen-2.5-72b': 'qwen/qwen-2.5-72b-instruct',  # $0.12/M - Chinese model, affordable
    'mistral-large-3': 'mistralai/mistral-large-2512',  # Mistral Large 3 2512 - latest flagship
}

# Combined model mapping (for OpenRouter)
MODEL_IDENTIFIERS = {
    **GEMINI_MODELS,
    **OPENAI_MODELS,
    **CLAUDE_MODELS,
    **OPENROUTER_FREE_MODELS,
    **OPENROUTER_PAID_MODELS,
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

def get_anthropic_ai_model(frontend_model: str) -> str:
    """
    Get the Anthropic API model identifier for a frontend model name.

    Args:
        frontend_model: Model name from frontend (e.g., 'claude-opus-4.5', 'claude-sonnet-4.5')

    Returns:
        Anthropic API model identifier (e.g., 'claude-opus-4-5')

    Raises:
        ValueError: If model name is not recognized or not a Claude model
    """
    if frontend_model not in ANTHROPIC_AI_MODELS:
        available = ', '.join(ANTHROPIC_AI_MODELS.keys())
        raise ValueError(
            f"Unknown Anthropic model: {frontend_model}. "
            f"Available models: {available}"
        )

    return ANTHROPIC_AI_MODELS[frontend_model]
