/**
 * Base inference client interface
 *
 * This defines the contract that all inference implementations must follow.
 * - Local version: Calls local Python server (reads API keys from .env)
 * - Web version: Calls cloud API (sends API keys in headers)
 */

import type {
  ChatOptions,
  ChatResponse,
  ValidationOptions,
  ValidationResponse,
  ExportOptions,
  ExportResponse,
  NodeDefinitionsResponse,
} from './types'

/**
 * Abstract base class for inference clients
 *
 * Implementations:
 * - LocalInferenceClient: Communicates with local Python server
 * - ApiInferenceClient: Communicates with cloud backend (requires API key)
 */
export abstract class BaseInferenceClient {
  /**
   * Send a chat message to the AI assistant
   *
   * @param options - Chat options including message, history, and workflow state
   * @returns AI response with optional node modifications
   */
  abstract chat(options: ChatOptions): Promise<ChatResponse>

  /**
   * Validate model architecture
   *
   * @param options - Validation options with nodes and edges
   * @returns Validation result with errors and warnings
   */
  abstract validateModel(options: ValidationOptions): Promise<ValidationResponse>

  /**
   * Export model to production code
   *
   * @param options - Export options including framework and project name
   * @returns Generated code files
   */
  abstract exportModel(options: ExportOptions): Promise<ExportResponse>

  /**
   * Get available node definitions for a framework
   *
   * @param framework - Target framework (pytorch or tensorflow)
   * @returns List of available node types
   */
  abstract getNodeDefinitions(
    framework: 'pytorch' | 'tensorflow'
  ): Promise<NodeDefinitionsResponse>
}

/**
 * Factory function type for creating inference clients
 */
export type InferenceClientFactory = () => BaseInferenceClient

/**
 * Inference provider interface for dependency injection
 */
export interface InferenceProvider {
  client: BaseInferenceClient
}
