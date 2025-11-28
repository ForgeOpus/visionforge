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

  /**
   * Project Management APIs
   */

  /**
   * Create a new project
   */
  async createProject(data: {
    name: string
    description?: string
    framework?: 'pytorch' | 'tensorflow'
    nodes?: any[]
    edges?: any[]
  }) {
    try {
      const apiUrl = (localClient as any).apiUrl || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}/api/projects`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })
      if (!response.ok) throw new Error('Failed to create project')
      const result = await response.json()
      return { success: true, data: result }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to create project',
      }
    }
  },

  /**
   * Get all projects
   */
  async getProjects() {
    try {
      const apiUrl = (localClient as any).apiUrl || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}/api/projects`)
      if (!response.ok) throw new Error('Failed to fetch projects')
      const result = await response.json()
      return { success: true, data: result }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to fetch projects',
      }
    }
  },

  /**
   * Get a specific project
   */
  async getProject(projectId: number) {
    try {
      const apiUrl = (localClient as any).apiUrl || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}/api/projects/${projectId}`)
      if (!response.ok) throw new Error('Failed to fetch project')
      const result = await response.json()
      return { success: true, data: result }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to fetch project',
      }
    }
  },

  /**
   * Update a project
   */
  async updateProject(
    projectId: number,
    data: {
      name?: string
      description?: string
      framework?: 'pytorch' | 'tensorflow'
    }
  ) {
    try {
      const apiUrl = (localClient as any).apiUrl || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}/api/projects/${projectId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })
      if (!response.ok) throw new Error('Failed to update project')
      const result = await response.json()
      return { success: true, data: result }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to update project',
      }
    }
  },

  /**
   * Delete a project
   */
  async deleteProject(projectId: number) {
    try {
      const apiUrl = (localClient as any).apiUrl || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}/api/projects/${projectId}`, {
        method: 'DELETE',
      })
      if (!response.ok) throw new Error('Failed to delete project')
      return { success: true, data: null }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to delete project',
      }
    }
  },

  /**
   * Save workflow to a project
   */
  async saveWorkflow(projectId: number, data: { nodes: any[]; edges: any[] }) {
    try {
      const apiUrl = (localClient as any).apiUrl || 'http://localhost:8000'
      const response = await fetch(
        `${apiUrl}/api/projects/${projectId}/workflow`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        }
      )
      if (!response.ok) throw new Error('Failed to save workflow')
      const result = await response.json()
      return { success: true, data: result }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to save workflow',
      }
    }
  },
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
 * Convenience wrapper for chat - maintains backward compatibility
 * Accepts individual parameters and converts to ChatOptions
 */
export async function sendChatMessage(
  message: string,
  history?: Array<{ role: 'user' | 'assistant'; content: string }>,
  modificationMode?: boolean,
  workflowState?: { nodes: any[]; edges: any[] } | null,
  file?: File,
  apiKey?: string
) {
  // Note: apiKey is not used in local mode - server reads from .env
  return api.chat({
    message,
    history,
    modificationMode,
    workflowState,
    file,
  })
}

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
