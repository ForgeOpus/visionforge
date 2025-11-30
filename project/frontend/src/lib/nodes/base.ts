/**
 * Abstract base class for node definitions providing default implementations
 * and shared utilities for all node types.
 */

import {
  INodeDefinition,
  NodeMetadata,
  BackendFramework,
  DimensionRequirement
} from './contracts'
import { TensorShape, BlockConfig, ConfigField, BlockType, ShapePattern } from '../types'
import { PortDefinition, DEFAULT_INPUT_PORT, DEFAULT_OUTPUT_PORT } from './ports'

/**
 * Abstract base class that all node definitions extend
 * Provides common functionality and enforces interface compliance
 */
export abstract class NodeDefinition implements INodeDefinition {
  abstract readonly metadata: NodeMetadata
  abstract readonly configSchema: ConfigField[]

  /**
   * Input shape pattern this node accepts
   * Override in subclasses to define specific shape requirements
   */
  readonly inputPattern?: ShapePattern

  /**
   * Output shape pattern this node produces
   * Override in subclasses to define output shape characteristics
   */
  readonly outputPattern?: ShapePattern

  /**
   * Get input ports for this node based on configuration
   * Override in subclasses for nodes with multiple or dynamic inputs
   */
  getInputPorts(config: BlockConfig): PortDefinition[] {
    return [DEFAULT_INPUT_PORT]
  }
  
  /**
   * Get output ports for this node based on configuration
   * Override in subclasses for nodes with multiple or dynamic outputs
   */
  getOutputPorts(config: BlockConfig): PortDefinition[] {
    return [DEFAULT_OUTPUT_PORT]
  }

  /**
   * Default implementation: pass through input shape
   * Override in subclasses for custom shape transformation
   */
  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    return inputShape
  }

  /**
   * Default implementation: allow connections from any source
   * Override in subclasses for specific validation rules
   */
  validateIncomingConnection(
    sourceNodeType: BlockType,
    sourceOutputShape: TensorShape | undefined,
    targetConfig: BlockConfig
  ): string | undefined {
    return undefined
  }

  /**
   * Default implementation: single input only
   * Override to return true for merge nodes
   */
  allowsMultipleInputs(): boolean {
    return false
  }

  /**
   * Default implementation: validate required fields from schema
   * Override to add custom validation logic
   */
  validateConfig(config: BlockConfig): string[] {
    const errors: string[] = []

    this.configSchema.forEach(field => {
      if (field.required && (config[field.name] === undefined || config[field.name] === '')) {
        errors.push(`${field.label} is required`)
      }

      // Validate numeric ranges
      if (field.type === 'number' && config[field.name] !== undefined) {
        const value = config[field.name] as number
        if (field.min !== undefined && value < field.min) {
          errors.push(`${field.label} must be at least ${field.min}`)
        }
        if (field.max !== undefined && value > field.max) {
          errors.push(`${field.label} must be at most ${field.max}`)
        }
      }
    })

    return errors
  }

  /**
   * Generate default configuration from schema
   */
  getDefaultConfig(): BlockConfig {
    const config: BlockConfig = {}

    this.configSchema.forEach(field => {
      if (field.default !== undefined) {
        config[field.name] = field.default
      }
    })

    return config
  }

  /**
   * Helper: Validate input tensor dimensions against requirements
   */
  protected validateDimensions(
    shape: TensorShape | undefined,
    requirement: DimensionRequirement
  ): string | undefined {
    if (!shape) {
      return 'Input shape is not defined'
    }

    const actualDims = shape.dims.length

    if (requirement.dims === 'any') {
      return undefined
    }

    if (typeof requirement.dims === 'number') {
      if (actualDims !== requirement.dims) {
        return `${this.metadata.label} requires ${requirement.dims}D input ${requirement.description}, got ${actualDims}D`
      }
    } else if (Array.isArray(requirement.dims)) {
      if (!requirement.dims.includes(actualDims)) {
        return `${this.metadata.label} requires ${requirement.dims.join(' or ')}D input ${requirement.description}, got ${actualDims}D`
      }
    }

    return undefined
  }

  /**
   * Helper: Compute 2D convolution output dimensions
   */
  protected computeConv2DOutput(
    inputHeight: number,
    inputWidth: number,
    kernelSize: number,
    stride: number,
    padding: number,
    dilation: number
  ): [number, number] {
    const effectiveKernel = dilation * (kernelSize - 1) + 1
    const outputHeight = Math.floor((inputHeight + 2 * padding - effectiveKernel) / stride + 1)
    const outputWidth = Math.floor((inputWidth + 2 * padding - effectiveKernel) / stride + 1)
    return [outputHeight, outputWidth]
  }

  /**
   * Helper: Compute 2D pooling output dimensions
   */
  protected computePool2DOutput(
    inputHeight: number,
    inputWidth: number,
    kernelSize: number,
    stride: number,
    padding: number
  ): [number, number] {
    const outputHeight = Math.floor((inputHeight + 2 * padding - kernelSize) / stride + 1)
    const outputWidth = Math.floor((inputWidth + 2 * padding - kernelSize) / stride + 1)
    return [outputHeight, outputWidth]
  }

  /**
   * Helper: Parse shape from string representation
   */
  protected parseShapeString(shapeStr: string): number[] | undefined {
    try {
      const parsed = JSON.parse(shapeStr)
      if (Array.isArray(parsed) && parsed.every(d => typeof d === 'number' && d > 0)) {
        return parsed
      }
    } catch {
      return undefined
    }
    return undefined
  }

  /**
   * Helper: Check if all dimensions in shapes match
   */
  protected shapesMatch(shape1: TensorShape, shape2: TensorShape): boolean {
    if (shape1.dims.length !== shape2.dims.length) {
      return false
    }

    return shape1.dims.every((dim, idx) => {
      const dim1 = typeof dim === 'number' ? dim : parseInt(String(dim), 10)
      const dim2 = typeof shape2.dims[idx] === 'number' 
        ? shape2.dims[idx] 
        : parseInt(String(shape2.dims[idx]), 10)
      return dim1 === dim2
    })
  }
}

/**
 * Base class for input/source nodes that don't receive connections
 */
export abstract class SourceNodeDefinition extends NodeDefinition {
  validateIncomingConnection(
    sourceNodeType: BlockType,
    sourceOutputShape: TensorShape | undefined,
    targetConfig: BlockConfig
  ): string | undefined {
    return `${this.metadata.label} blocks cannot receive connections (they are source nodes)`
  }
}

/**
 * Base class for output/terminal nodes that accept any input
 */
export abstract class TerminalNodeDefinition extends NodeDefinition {
  validateIncomingConnection(
    sourceNodeType: BlockType,
    sourceOutputShape: TensorShape | undefined,
    targetConfig: BlockConfig
  ): string | undefined {
    return undefined // Always valid
  }
}

/**
 * Base class for merge nodes that accept multiple inputs
 */
export abstract class MergeNodeDefinition extends NodeDefinition {
  allowsMultipleInputs(): boolean {
    return true
  }
}

/**
 * Base class for passthrough/utility nodes
 */
export abstract class PassthroughNodeDefinition extends NodeDefinition {
  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined {
    return inputShape
  }

  validateIncomingConnection(
    sourceNodeType: BlockType,
    sourceOutputShape: TensorShape | undefined,
    targetConfig: BlockConfig
  ): string | undefined {
    return undefined // Always valid
  }
}
