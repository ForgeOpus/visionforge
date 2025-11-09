/**
 * Legacy Block Definitions Adapter
 * Provides backward compatibility with the old blockDefinitions structure
 * while using the new modular node definition system under the hood.
 * 
 * @deprecated This adapter maintains compatibility during migration.
 * New code should use the registry directly from '../nodes/registry'.
 */

import { getNodeDefinition, getAllNodeDefinitions, BackendFramework } from '../nodes/registry'
import { BlockDefinition, TensorShape, BlockConfig, BlockType } from '../types'

/**
 * Flag to track if deprecation warning has been shown
 */
let hasShownDeprecationWarning = false

/**
 * Show deprecation warning once
 */
function showDeprecationWarning(): void {
  if (!hasShownDeprecationWarning) {
    console.warn(
      '[VisionForge] blockDefinitions is deprecated. ' +
      'Please migrate to using the node registry: ' +
      'import { getNodeDefinition } from "./lib/nodes/registry"'
    )
    hasShownDeprecationWarning = true
  }
}

/**
 * Convert new node definition to legacy block definition format
 */
function convertToLegacyFormat(nodeDef: any): BlockDefinition {
  return {
    type: nodeDef.metadata.type,
    label: nodeDef.metadata.label,
    category: nodeDef.metadata.category,
    color: nodeDef.metadata.color,
    icon: nodeDef.metadata.icon,
    description: nodeDef.metadata.description,
    configSchema: nodeDef.configSchema,
    computeOutputShape: (inputShape: TensorShape | undefined, config: BlockConfig) => {
      return nodeDef.computeOutputShape(inputShape, config)
    }
  }
}

/**
 * Build legacy blockDefinitions object from registry
 * Uses PyTorch as the default framework for backward compatibility
 */
function buildLegacyBlockDefinitions(): Record<string, BlockDefinition> {
  showDeprecationWarning()
  
  const definitions: Record<string, BlockDefinition> = {}
  const allNodes = getAllNodeDefinitions(BackendFramework.PyTorch)

  allNodes.forEach(nodeDef => {
    definitions[nodeDef.metadata.type] = convertToLegacyFormat(nodeDef)
  })

  return definitions
}

/**
 * Legacy blockDefinitions export
 * @deprecated Use getNodeDefinition() from registry instead
 */
export const blockDefinitions: Record<string, BlockDefinition> = new Proxy({}, {
  get(target, prop) {
    if (typeof prop === 'string') {
      showDeprecationWarning()
      const nodeDef = getNodeDefinition(prop as BlockType, BackendFramework.PyTorch)
      if (nodeDef) {
        return convertToLegacyFormat(nodeDef)
      }
    }
    return undefined
  },
  ownKeys() {
    showDeprecationWarning()
    const allNodes = getAllNodeDefinitions(BackendFramework.PyTorch)
    return allNodes.map(node => node.metadata.type)
  },
  getOwnPropertyDescriptor(target, prop) {
    if (typeof prop === 'string') {
      const nodeDef = getNodeDefinition(prop as BlockType, BackendFramework.PyTorch)
      if (nodeDef) {
        return {
          enumerable: true,
          configurable: true,
          value: convertToLegacyFormat(nodeDef)
        }
      }
    }
    return undefined
  }
})

/**
 * Get block definition by type
 * @deprecated Use getNodeDefinition() from registry instead
 */
export function getBlockDefinition(type: string): BlockDefinition | undefined {
  showDeprecationWarning()
  const nodeDef = getNodeDefinition(type as BlockType, BackendFramework.PyTorch)
  return nodeDef ? convertToLegacyFormat(nodeDef) : undefined
}

/**
 * Get blocks by category
 * @deprecated Use getNodeDefinitionsByCategory() from registry instead
 */
export function getBlocksByCategory(category: string): BlockDefinition[] {
  showDeprecationWarning()
  const allNodes = getAllNodeDefinitions(BackendFramework.PyTorch)
  return allNodes
    .filter(node => node.metadata.category === category)
    .map(node => convertToLegacyFormat(node))
}

/**
 * Validate connection between blocks
 * @deprecated Use node.validateIncomingConnection() method instead
 */
export function validateBlockConnection(
  sourceBlockType: string,
  targetBlockType: string,
  sourceOutputShape?: TensorShape
): string | undefined {
  showDeprecationWarning()
  
  const targetNode = getNodeDefinition(targetBlockType as BlockType, BackendFramework.PyTorch)
  if (!targetNode) {
    return 'Unknown target block type'
  }

  return targetNode.validateIncomingConnection(
    sourceBlockType as BlockType,
    sourceOutputShape,
    {}
  )
}

/**
 * Check if block allows multiple inputs
 * @deprecated Use node.allowsMultipleInputs() method instead
 */
export function allowsMultipleInputs(blockType: string): boolean {
  showDeprecationWarning()
  
  const nodeDef = getNodeDefinition(blockType as BlockType, BackendFramework.PyTorch)
  if (!nodeDef) {
    return false
  }

  return nodeDef.allowsMultipleInputs()
}
