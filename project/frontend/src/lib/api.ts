/**
 * API Service for VisionForge
 * Handles all backend communication
 */

// API configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'

interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<ApiResponse<T>> {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
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
  file?: File
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
  })
}

/**
 * Export model configuration
 */
export async function exportModel(modelData: {
  nodes: any[]
  edges: any[]
  format: 'pytorch' | 'tensorflow' | 'onnx'
}): Promise<ApiResponse<{
  code: string
}>> {
  return apiFetch('/export', {
    method: 'POST',
    body: JSON.stringify(modelData),
  })
}

/**
 * Get model suggestions based on current architecture
 */
export async function getModelSuggestions(modelData: {
  nodes: any[]
  edges: any[]
}): Promise<ApiResponse<{
  suggestions: string[]
}>> {
  return apiFetch('/suggestions', {
    method: 'POST',
    body: JSON.stringify(modelData),
  })
}

/**
 * Get all available node definitions for a framework
 */
export async function getNodeDefinitions(framework: 'pytorch' | 'tensorflow' = 'pytorch'): Promise<ApiResponse<{
  [nodeType: string]: any
}>> {
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
  metadata: any
  config_schema: any
}>> {
  return apiFetch(`/node-definitions/${nodeType}?framework=${framework}`, {
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
}
