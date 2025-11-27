/**
 * API client for local FastAPI server
 *
 * This wraps the LocalInferenceClient and provides additional
 * utilities for working with the VisionForge local server.
 */

import { localClient } from './inference'
import type {
  ChatOptions,
  ValidationOptions,
  ExportOptions,
} from '@visionforge/core/inference'

/**
 * API client singleton
 */
export const api = {
  /**
   * Health check - verify server is running
   */
  async healthCheck() {
    try {
      const health = await localClient.healthCheck()
      return {
        success: true,
        data: health,
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Server not available',
      }
    }
  },

  /**
   * Chat with AI assistant
   */
  async chat(options: ChatOptions) {
    try {
      const response = await localClient.chat(options)
      return {
        success: true,
        data: response,
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Chat failed',
      }
    }
  },

  /**
   * Validate model architecture
   */
  async validateModel(options: ValidationOptions) {
    try {
      const response = await localClient.validateModel(options)
      return {
        success: true,
        data: response,
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Validation failed',
      }
    }
  },

  /**
   * Export model to code
   */
  async exportModel(options: ExportOptions) {
    try {
      const response = await localClient.exportModel(options)
      return {
        success: true,
        data: response,
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Export failed',
      }
    }
  },

  /**
   * Get node definitions for a framework
   */
  async getNodeDefinitions(framework: 'pytorch' | 'tensorflow' = 'pytorch') {
    try {
      const response = await localClient.getNodeDefinitions(framework)
      return {
        success: true,
        data: response,
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to fetch node definitions',
      }
    }
  },

  /**
   * Direct access to inference client (for advanced use)
   */
  client: localClient,
}

/**
 * Helper for downloading exported files
   */
export function downloadExportedFiles(
  files: Record<string, string>,
  projectName: string,
  framework: string
) {
  Object.entries(files).forEach(([filename, content]) => {
    const blob = new Blob([content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${projectName}_${framework}_${filename}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  })
}

/**
 * Helper for downloading zip file
 */
export function downloadZip(base64Zip: string, filename: string) {
  const binaryString = atob(base64Zip)
  const bytes = new Uint8Array(binaryString.length)
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i)
  }
  const blob = new Blob([bytes], { type: 'application/zip' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export default api

/**
 * Render code for a specific node
 */
export async function renderNodeCode(
  nodeType: string,
  framework: 'pytorch' | 'tensorflow',
  config: Record<string, any>,
  metadata?: Record<string, any>
): Promise<any> {
  // TODO: Implement when backend is ready
  return Promise.resolve({ 
    success: false, 
    error: "Feature not yet implemented",
    data: null 
  });
}
