import type {
  ShapePattern,
  ShapeFlags,
  ShapeProvenance,
  NodeShapeStatus,
  DimensionValue,
} from '@visionforge/core/validation';

export type BlockType =
  | 'input'
  | 'dataloader'
  | 'output'
  | 'loss'
  | 'empty'
  | 'linear'
  | 'conv2d'
  | 'dropout'
  | 'batchnorm'
  | 'relu'
  | 'flatten'
  | 'maxpool2d'
  | 'attention'
  | 'concat'
  | 'softmax'
  | 'add'
  | 'custom'

export type BlockCategory = 'input' | 'output' | 'basic' | 'activation' | 'advanced' | 'merge' | 'utility'

/**
 * Tensor shape representation
 * Supports both numeric and symbolic dimensions for progressive resolution
 */
export interface TensorShape {
  /** Dimension values - can be numbers or symbolic strings (e.g., 'B', 'T', 'F') */
  dims: DimensionValue[];
  /** Human-readable description */
  description?: string;
  /** Shape pattern this tensor conforms to */
  pattern?: ShapePattern;
  /** Behavior flags (auto_flatten, inferred, etc.) */
  flags?: ShapeFlags;
  /** Provenance information for how shape was derived */
  provenance?: ShapeProvenance;
}

// Re-export validation types for convenience
export type { ShapePattern, ShapeFlags, ShapeProvenance, NodeShapeStatus, DimensionValue };

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
  /** Validation and shape inference status */
  shapeStatus?: NodeShapeStatus
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
}
