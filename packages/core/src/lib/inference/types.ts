/**
 * Shared types for inference providers
 */

import type { ValidationError as CoreValidationError } from '../types'

export interface ChatOptions {
  message: string
  history?: ChatMessage[]
  workflowState?: WorkflowState | null
  modificationMode?: boolean
  file?: File
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface WorkflowState {
  nodes: any[]
  edges: any[]
}

export interface ChatResponse {
  response: string
  modifications?: NodeModification[]
}

export interface NodeModification {
  type: 'add' | 'update' | 'delete'
  nodeId?: string
  node?: any
  edge?: any
}

export interface ValidationOptions {
  nodes: any[]
  edges: any[]
}

export interface ValidationResponse {
  isValid: boolean
  errors?: CoreValidationError[]
  warnings?: ValidationWarning[]
}

export interface ValidationWarning {
  nodeId?: string
  message: string
  type: 'performance' | 'best_practice'
}

export interface ExportOptions {
  nodes: any[]
  edges: any[]
  format: 'pytorch' | 'tensorflow'
  projectName: string
}

export interface ExportResponse {
  success: boolean
  framework: string
  projectName: string
  files: {
    [filename: string]: string
  }
  zip?: string  // Base64 encoded zip file
  filename?: string
}

export interface NodeDefinitionsResponse {
  definitions: NodeDefinition[]
  framework: 'pytorch' | 'tensorflow'
}

export interface NodeDefinition {
  type: string
  label: string
  category: string
  description?: string
  config?: Record<string, any>
  ports?: {
    inputs?: PortDefinition[]
    outputs?: PortDefinition[]
  }
}

export interface PortDefinition {
  id: string
  label: string
  type: string
}
