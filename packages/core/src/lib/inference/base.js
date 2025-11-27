/**
 * Base inference client interface
 *
 * This defines the contract that all inference implementations must follow.
 * - Local version: Calls local Python server (reads API keys from .env)
 * - Web version: Calls cloud API (sends API keys in headers)
 */
/**
 * Abstract base class for inference clients
 *
 * Implementations:
 * - LocalInferenceClient: Communicates with local Python server
 * - ApiInferenceClient: Communicates with cloud backend (requires API key)
 */
export class BaseInferenceClient {
}
