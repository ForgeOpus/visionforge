/**
 * Local Inference Client
 *
 * Communicates with the local Python FastAPI server.
 * API keys are read from .env file by the server - never exposed to frontend.
 */

import {
  BaseInferenceClient,
  ChatOptions,
  ChatResponse,
  ValidationOptions,
  ValidationResponse,
  ExportOptions,
  ExportResponse,
  NodeDefinitionsResponse,
} from '@visionforge/core/inference'

/**
 * Local implementation of inference client
 * Talks to FastAPI server running on localhost
 */
export class LocalInferenceClient extends BaseInferenceClient {
  private apiUrl: string

  constructor(apiUrl: string = 'http://localhost:8000') {
    super()
    this.apiUrl = apiUrl
  }

  /**
   * Send chat message to local AI service
   * API keys are handled server-side from .env file
   */
  async chat(options: ChatOptions): Promise<ChatResponse> {
    try {
      const formData = new FormData()

      formData.append('message', options.message)
      formData.append('history', JSON.stringify(options.history || []))
      formData.append('modificationMode', String(options.modificationMode || false))
      formData.append('workflowState', JSON.stringify(options.workflowState || null))

      if (options.file) {
        formData.append('file', options.file)
      }

      const response = await fetch(`${this.apiUrl}/api/chat`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Chat request failed')
      }

      const data = await response.json()
      return {
        response: data.response,
        modifications: data.modifications,
      }
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(`Failed to connect to local server: ${error.message}`)
      }
      throw new Error('Failed to connect to local server')
    }
  }

  /**
   * Validate model architecture
   */
  async validateModel(options: ValidationOptions): Promise<ValidationResponse> {
    try {
      const response = await fetch(`${this.apiUrl}/api/validate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(options),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Validation failed')
      }

      return await response.json()
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(`Validation failed: ${error.message}`)
      }
      throw new Error('Validation failed')
    }
  }

  /**
   * Export model to code
   */
  async exportModel(options: ExportOptions): Promise<ExportResponse> {
    try {
      const response = await fetch(`${this.apiUrl}/api/export`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(options),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Export failed')
      }

      return await response.json()
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(`Export failed: ${error.message}`)
      }
      throw new Error('Export failed')
    }
  }

  /**
   * Get available node definitions
   */
  async getNodeDefinitions(
    framework: 'pytorch' | 'tensorflow'
  ): Promise<NodeDefinitionsResponse> {
    try {
      const response = await fetch(
        `${this.apiUrl}/api/node-definitions?framework=${framework}`
      )

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to fetch node definitions')
      }

      return await response.json()
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(`Failed to fetch node definitions: ${error.message}`)
      }
      throw new Error('Failed to fetch node definitions')
    }
  }

  /**
   * Health check - verify server is running
   */
  async healthCheck(): Promise<{ status: string; ai_enabled: boolean }> {
    try {
      const response = await fetch(`${this.apiUrl}/api/health`)
      if (!response.ok) {
        throw new Error('Server not responding')
      }
      return await response.json()
    } catch (error) {
      throw new Error('Cannot connect to local server. Is it running?')
    }
  }
}

/**
 * Create a local inference client instance
 */
export function createLocalClient(apiUrl?: string): LocalInferenceClient {
  return new LocalInferenceClient(apiUrl)
}

/**
 * Default client instance
 */
export const localClient = new LocalInferenceClient()
