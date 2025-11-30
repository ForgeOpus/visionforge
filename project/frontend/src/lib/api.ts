/**
 * API Service for VisionForge
 * Handles all backend communication
 */

import type { NodeSpec, NodeDefinitionsResponse, RenderCodeResponse } from './nodeSpec.types'
import { getCsrfToken } from './apiUtils'

// API configuration
// For single-service deployment: use relative path (same origin, no CORS)
// For development: use full URL to separate backend server
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

// Type for API key headers
interface ApiKeyHeaders {
  geminiApiKey?: string | null
  anthropicApiKey?: string | null
}

/**
 * Get API key headers for requests
 */
function getApiKeyHeaders(keys?: ApiKeyHeaders): Record<string, string> {
  const headers: Record<string, string> = {}

  if (keys?.geminiApiKey) {
    headers['X-Gemini-Api-Key'] = keys.geminiApiKey
  }

  if (keys?.anthropicApiKey) {
    headers['X-Anthropic-Api-Key'] = keys.anthropicApiKey
  }

  return headers
}

/**
 * Generic fetch wrapper with error handling and API key support
 */
async function apiFetch<T>(
  endpoint: string,
  options: RequestInit = {},
  apiKeys?: ApiKeyHeaders
): Promise<ApiResponse<T>> {
  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...getApiKeyHeaders(apiKeys),
    }

    // Add CSRF token for unsafe methods (POST, PUT, DELETE, PATCH)
    const isUnsafeMethod = options.method &&
      !['GET', 'HEAD', 'OPTIONS', 'TRACE'].includes(options.method.toUpperCase())

    if (isUnsafeMethod) {
      const csrfToken = getCsrfToken()
      if (csrfToken) {
        headers['X-CSRFToken'] = csrfToken
      } else {
        console.warn('CSRF token not found. Cookie:', document.cookie)
      }
    }

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        ...headers,
        ...options.headers,
      },
      ...options,
    })

    // Handle non-JSON responses (like 404 HTML pages)
    let data: any
    try {
      data = await response.json()
    } catch (jsonError) {
      if (!response.ok) {
        return {
          success: false,
          error: `Server error (${response.status}): ${response.statusText}`,
        }
      }
      throw jsonError
    }

    if (!response.ok) {
      return {
        success: false,
        error: data.error || data.message || 'An error occurred',
      }
    }

    return {
      success: true,
      data,
    }
  } catch (error) {
    console.error('API Error:', error)
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Network error occurred',
    }
  }
}

/**
 * Validate model architecture
 * Sends the model JSON structure to backend for validation
 */
export async function validateModel(modelData: {
  nodes: any[]
  edges: any[]
}): Promise<ApiResponse<{
  isValid: boolean
  errors?: string[]
  warnings?: string[]
}>> {
  return apiFetch('/validate', {
    method: 'POST',
    body: JSON.stringify(modelData),
  })
}

/**
 * Send chat message to AI assistant with workflow context
 */
export async function sendChatMessage(
  message: string,
  history?: any[],
  modificationMode?: boolean,
  workflowState?: { nodes: any[], edges: any[] },
  file?: File,
  apiKeys?: ApiKeyHeaders
): Promise<ApiResponse<{
  response: string
  modifications?: any[]
}>> {
  // If there's a file, use FormData
  if (file) {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('message', message)
    formData.append('history', JSON.stringify(history || []))
    formData.append('modificationMode', String(modificationMode || false))
    formData.append('workflowState', JSON.stringify(workflowState || null))

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: getApiKeyHeaders(apiKeys),
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        return {
          success: false,
          error: data.error || data.message || 'An error occurred',
        }
      }

      return {
        success: true,
        data,
      }
    } catch (error) {
      console.error('API Error:', error)
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error occurred',
      }
    }
  }

  // No file - use regular JSON
  return apiFetch('/chat', {
    method: 'POST',
    body: JSON.stringify({
      message,
      history: history || [],
      modificationMode: modificationMode || false,
      workflowState: workflowState || null
    }),
  }, apiKeys)
}

/**
 * Export model configuration with professional multi-file code generation
 */
export async function exportModel(modelData: {
  nodes: any[]
  edges: any[]
  format: 'pytorch' | 'tensorflow'
  projectName: string
}): Promise<ApiResponse<{
  success: boolean
  framework: string
  projectName: string
  files: {
    'model.py': string
    'train.py': string
    'dataset.py': string
    'config.py': string
  }
  zip: string  // Base64 encoded zip file
  filename: string
}>> {
  return apiFetch('/export', {
    method: 'POST',
    body: JSON.stringify(modelData),
  })
}

/**
 * Get model suggestions based on current architecture
 */
export async function getModelSuggestions(
  modelData: {
    nodes: any[]
    edges: any[]
  },
  apiKeys?: ApiKeyHeaders
): Promise<ApiResponse<{
  suggestions: string[]
}>> {
  return apiFetch('/suggestions', {
    method: 'POST',
    body: JSON.stringify(modelData),
  }, apiKeys)
}

/**
 * Get all available node definitions for a framework
 */
export async function getNodeDefinitions(
  framework: 'pytorch' | 'tensorflow' = 'pytorch'
): Promise<ApiResponse<NodeDefinitionsResponse>> {
  return apiFetch(`/node-definitions?framework=${framework}`, {
    method: 'GET',
  })
}

/**
 * Get a specific node definition
 */
export async function getNodeDefinition(
  nodeType: string, 
  framework: 'pytorch' | 'tensorflow' = 'pytorch'
): Promise<ApiResponse<{
  success: boolean
  definition: NodeSpec
}>> {
  return apiFetch(`/node-definitions/${nodeType}?framework=${framework}`, {
    method: 'GET',
  })
}

/**
 * Render node code from spec and config
 */
export async function renderNodeCode(
  nodeType: string,
  framework: 'pytorch' | 'tensorflow',
  config: Record<string, any>,
  metadata?: Record<string, any>
): Promise<ApiResponse<RenderCodeResponse>> {
  return apiFetch('/render-node-code', {
    method: 'POST',
    body: JSON.stringify({
      node_type: nodeType,
      framework,
      config,
      metadata: metadata || {},
    }),
  })
}

/**
 * Get environment configuration from backend
 */
export async function getEnvironmentInfo(): Promise<ApiResponse<{
  environment: string
  isProduction: boolean
  requiresApiKey: boolean
  provider: string
}>> {
  return apiFetch('/environment', {
    method: 'GET',
  })
}

export default {
  validateModel,
  sendChatMessage,
  exportModel,
  getModelSuggestions,
  getNodeDefinitions,
  getNodeDefinition,
  renderNodeCode,
  getEnvironmentInfo,
}
