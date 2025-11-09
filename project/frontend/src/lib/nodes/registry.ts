/**
 * Node Definition Registry
 * Dynamically loads and manages node definitions for all supported frameworks
 */

import { INodeDefinition, BackendFramework, NodeDefinitionRegistry } from './contracts'
import { BlockType } from '../types'

// Import all PyTorch nodes
import * as PyTorchNodes from './definitions/pytorch'

// Import all TensorFlow nodes
import * as TensorFlowNodes from './definitions/tensorflow'

/**
 * Registry cache - stores instantiated node definitions
 */
const registryCache: NodeDefinitionRegistry = {
  [BackendFramework.PyTorch]: {},
  [BackendFramework.TensorFlow]: {}
}

/**
 * Category cache - memoizes category groupings for performance
 */
const categoryCache: Map<BackendFramework, Map<string, INodeDefinition[]>> = new Map()

/**
 * Initialization flag to ensure registry is built only once
 */
let isInitialized = false

/**
 * Initialize the registry by instantiating all node definition classes
 */
function initializeRegistry(): void {
  if (isInitialized) return

  // Register PyTorch nodes
  Object.values(PyTorchNodes).forEach((NodeClass: any) => {
    const instance = new NodeClass() as INodeDefinition
    registryCache[BackendFramework.PyTorch][instance.metadata.type] = instance
  })

  // Register TensorFlow nodes
  Object.values(TensorFlowNodes).forEach((NodeClass: any) => {
    const instance = new NodeClass() as INodeDefinition
    registryCache[BackendFramework.TensorFlow][instance.metadata.type] = instance
  })

  isInitialized = true
}

/**
 * Get a specific node definition by type and framework
 * @param type - The block type to retrieve
 * @param framework - The target framework (defaults to PyTorch)
 * @returns The node definition or undefined if not found
 */
export function getNodeDefinition(
  type: BlockType,
  framework: BackendFramework = BackendFramework.PyTorch
): INodeDefinition | undefined {
  initializeRegistry()
  return registryCache[framework]?.[type]
}

/**
 * Get all node definitions for a specific framework
 * @param framework - The target framework (defaults to PyTorch)
 * @returns Array of all node definitions for the framework
 */
export function getAllNodeDefinitions(
  framework: BackendFramework = BackendFramework.PyTorch
): INodeDefinition[] {
  initializeRegistry()
  return Object.values(registryCache[framework] || {})
}

/**
 * Get node definitions grouped by category
 * @param framework - The target framework (defaults to PyTorch)
 * @returns Map of category to node definitions
 */
export function getNodeDefinitionsByCategory(
  framework: BackendFramework = BackendFramework.PyTorch
): Map<string, INodeDefinition[]> {
  // Check cache first
  if (categoryCache.has(framework)) {
    return categoryCache.get(framework)!
  }

  // Build category map
  const allNodes = getAllNodeDefinitions(framework)
  const byCategory = new Map<string, INodeDefinition[]>()

  allNodes.forEach(node => {
    const category = node.metadata.category
    if (!byCategory.has(category)) {
      byCategory.set(category, [])
    }
    byCategory.get(category)!.push(node)
  })

  // Cache the result
  categoryCache.set(framework, byCategory)
  return byCategory
}

/**
 * Check if a node type exists in the registry
 * @param type - The block type to check
 * @param framework - The target framework (defaults to PyTorch)
 * @returns True if the node type exists
 */
export function hasNodeDefinition(
  type: BlockType,
  framework: BackendFramework = BackendFramework.PyTorch
): boolean {
  initializeRegistry()
  return !!registryCache[framework]?.[type]
}

/**
 * Get all available node types for a framework
 * @param framework - The target framework (defaults to PyTorch)
 * @returns Array of block types
 */
export function getAvailableNodeTypes(
  framework: BackendFramework = BackendFramework.PyTorch
): BlockType[] {
  initializeRegistry()
  return Object.keys(registryCache[framework] || {}) as BlockType[]
}

/**
 * Reset the registry (useful for testing)
 */
export function resetRegistry(): void {
  isInitialized = false
  categoryCache.clear()
  Object.keys(registryCache).forEach(framework => {
    registryCache[framework as BackendFramework] = {}
  })
}

// Export the framework enum for convenience
export { BackendFramework }
