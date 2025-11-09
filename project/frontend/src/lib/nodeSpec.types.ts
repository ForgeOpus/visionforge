/**
 * TypeScript types for NodeSpec API responses
 * Mirrors the Python NodeSpec structure from backend
 */

export type Framework = 'pytorch' | 'tensorflow'

export interface ConfigOption {
  value: string
  label: string
}

export interface ConfigField {
  name: string
  label: string
  field_type: 'text' | 'number' | 'boolean' | 'select'
  default?: any
  required?: boolean
  min?: number
  max?: number
  options?: ConfigOption[]
  description?: string
}

export interface NodeTemplate {
  name: string
  engine: 'jinja2'
  content: string
}

export interface NodeSpec {
  type: string
  label: string
  category: 'input' | 'basic' | 'advanced' | 'merge' | 'output' | 'utility'
  color: string
  icon: string
  description: string
  framework: Framework
  config_schema: ConfigField[]
  template: NodeTemplate
  allows_multiple_inputs?: boolean
}

export interface NodeDefinitionsResponse {
  success: boolean
  framework: Framework
  definitions: NodeSpec[]
  count: number
}

export interface RenderCodeRequest {
  node_type: string
  framework: Framework
  config: Record<string, any>
  metadata?: Record<string, any>
}

export interface RenderCodeResponse {
  success: boolean
  code: string
  spec_hash: string
  node_type: string
  framework: Framework
  context: Record<string, any>
}
