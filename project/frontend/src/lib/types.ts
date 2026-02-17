import type { Node, Edge } from '@xyflow/react'
import type { PortSemantic } from './nodes/ports'

export type BlockType =
  | 'input'
  | 'dataloader'
  | 'groundtruth'
  | 'output'
  | 'loss'
  | 'empty'
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
  | 'add'
  | 'custom'
  | 'group'

export type BlockCategory = 'input' | 'output' | 'basic' | 'activation' | 'advanced' | 'merge' | 'utility'

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
  type: 'number' | 'select' | 'boolean' | 'tuple' | 'text' | 'file'
  required?: boolean
  default?: number | string | boolean | number[]
  min?: number
  max?: number
  options?: { value: string | number; label: string }[]
  description?: string
  placeholder?: string
  accept?: string  // For file inputs, e.g., ".csv,.txt"
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
  blockName?: string  // For shape inference errors
  layerName?: string  // For shape inference errors
}

// Group Block Types

export interface PortMapping {
  internalNodeId: string      // ID of internal node
  internalPortId: string      // Port ID on internal node
  externalPortId: string      // Generated external port ID
  externalPortLabel: string   // User-provided display label
  type: 'input' | 'output'
  semantic: PortSemantic      // Preserve semantic type
}

export interface GroupBlockDefinition {
  id: string
  name: string
  description: string
  category: BlockCategory
  color: string

  // Internal structure (snapshot at creation)
  internalNodes: Node<BlockData>[]
  internalEdges: Edge[]
  portMappings: PortMapping[]

  createdAt: number
  updatedAt: number
}

export interface GroupBlockData extends BlockData {
  blockType: 'group'
  groupDefinitionId: string
  
  // Explicit expansion state
  isExpanded: boolean
  
  // Reference to container when expanded
  expandedContainerId?: string

  // Instance-specific configuration overrides for internal nodes
  instanceConfigOverrides?: Record<string, BlockConfig>

  // For repeated blocks
  repetitionMetadata?: {
    sequenceId: string
    index: number
    totalCount: number
  }
}
