export type BlockType =
  | 'input'
  | 'linear'
  | 'conv2d'
  | 'dropout'
  | 'batchnorm'
  | 'relu'
  | 'flatten'
  | 'maxpool'
  | 'attention'
  | 'concat'
  | 'softmax'

export type BlockCategory = 'input' | 'basic' | 'advanced' | 'merge'

export interface TensorShape {
  dims: (number | string)[]
  description?: string
}

export interface BlockConfig {
  [key: string]: number | string | boolean | number[]
}

export interface BlockData extends Record<string, unknown> {
  blockType: BlockType
  label: string
  config: BlockConfig
  inputShape?: TensorShape
  outputShape?: TensorShape
  category: BlockCategory
}

export interface BlockDefinition {
  type: BlockType
  label: string
  category: BlockCategory
  color: string
  icon: string
  description: string
  configSchema: ConfigField[]
  computeOutputShape: (inputShape: TensorShape | undefined, config: BlockConfig) => TensorShape | undefined
}

export interface ConfigField {
  name: string
  label: string
  type: 'number' | 'select' | 'boolean' | 'tuple' | 'text'
  required?: boolean
  default?: number | string | boolean | number[]
  min?: number
  max?: number
  options?: { value: string | number; label: string }[]
  description?: string
  placeholder?: string
}

export interface Project {
  id: string
  name: string
  description: string
  framework: 'pytorch' | 'tensorflow'
  nodes: any[]
  edges: any[]
  createdAt: number
  updatedAt: number
}

export interface ValidationError {
  nodeId?: string
  edgeId?: string
  message: string
  type: 'error' | 'warning'
}
