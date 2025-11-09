/**
 * Core interfaces and contracts for the modular node definition system.
 * These interfaces define the structure for extensible, framework-agnostic node definitions.
 */

import { TensorShape, BlockConfig, ConfigField, BlockType, BlockCategory } from '../types'
import { PortDefinition } from './ports'

/**
 * Supported backend frameworks for model building
 */
export enum BackendFramework {
  PyTorch = 'pytorch',
  TensorFlow = 'tensorflow'
}

/**
 * Metadata describing a node's visual and categorical properties
 */
export interface NodeMetadata {
  /** Unique identifier for this node type */
  type: BlockType
  /** Display label shown in UI */
  label: string
  /** Categorical grouping for palette organization */
  category: BlockCategory
  /** CSS color variable for visual theming */
  color: string
  /** Phosphor icon name for node representation */
  icon: string
  /** Human-readable description of node functionality */
  description: string
  /** Target framework this definition is optimized for */
  framework: BackendFramework
}

/**
 * Interface for shape computation logic
 */
export interface IShapeComputer {
  /**
   * Compute the output tensor shape given input shape and configuration
   * @param inputShape - Shape of incoming tensor (undefined for source nodes)
   * @param config - Node configuration parameters
   * @returns Output shape or undefined if cannot be determined
   */
  computeOutputShape(inputShape: TensorShape | undefined, config: BlockConfig): TensorShape | undefined
}

/**
 * Interface for connection validation logic
 */
export interface INodeValidator {
  /**
   * Validate whether this node can receive a connection from a source node
   * @param sourceNodeType - Type of the source node
   * @param sourceOutputShape - Output shape of the source node
   * @param targetConfig - Configuration of this target node
   * @returns Error message if invalid, undefined if valid
   */
  validateIncomingConnection(
    sourceNodeType: BlockType,
    sourceOutputShape: TensorShape | undefined,
    targetConfig: BlockConfig
  ): string | undefined

  /**
   * Check if this node type allows multiple input connections
   * @returns true if multiple inputs are supported (e.g., merge nodes)
   */
  allowsMultipleInputs(): boolean

  /**
   * Validate node configuration parameters
   * @param config - Configuration to validate
   * @returns Array of error messages, empty if valid
   */
  validateConfig(config: BlockConfig): string[]
}

/**
 * Complete interface for a node definition
 * Combines metadata, configuration schema, shape computation, and validation
 */
export interface INodeDefinition extends IShapeComputer, INodeValidator {
  /** Metadata describing this node */
  readonly metadata: NodeMetadata

  /** Configuration schema defining editable parameters */
  readonly configSchema: ConfigField[]

  /**
   * Get input ports for this node based on configuration
   * @param config - Node configuration
   * @returns Array of input port definitions
   */
  getInputPorts(config: BlockConfig): PortDefinition[]

  /**
   * Get output ports for this node based on configuration
   * @param config - Node configuration
   * @returns Array of output port definitions
   */
  getOutputPorts(config: BlockConfig): PortDefinition[]

  /**
   * Get default configuration values for this node
   * @returns Default configuration object
   */
  getDefaultConfig(): BlockConfig

  /**
   * Generate framework-specific code for this node
   * @param config - Node configuration
   * @param varName - Variable name to use in generated code
   * @returns Generated code string
   */
  generateCode?(config: BlockConfig, varName: string): string
}

/**
 * Constructor signature for node definition classes
 */
export interface NodeDefinitionConstructor {
  new(): INodeDefinition
}

/**
 * Registry map structure for organizing node definitions
 */
export interface NodeDefinitionRegistry {
  [framework: string]: {
    [nodeType: string]: INodeDefinition
  }
}

/**
 * Configuration for dimension validation
 */
export interface DimensionRequirement {
  /** Required number of dimensions, or 'any' for no restriction */
  dims: number | 'any' | number[]
  /** Human-readable description of the requirement */
  description: string
}
