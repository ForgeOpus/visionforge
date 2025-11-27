/**
 * Node Definition Registry
 * Dynamically loads and manages node definitions for all supported frameworks
 */
import { INodeDefinition, BackendFramework } from './contracts';
import { BlockType } from '../types';
/**
 * Get a specific node definition by type and framework
 * @param type - The block type to retrieve
 * @param framework - The target framework (defaults to PyTorch)
 * @returns The node definition or undefined if not found
 */
export declare function getNodeDefinition(type: BlockType, framework?: BackendFramework): INodeDefinition | undefined;
/**
 * Get all node definitions for a specific framework
 * @param framework - The target framework (defaults to PyTorch)
 * @returns Array of all node definitions for the framework
 */
export declare function getAllNodeDefinitions(framework?: BackendFramework): INodeDefinition[];
/**
 * Get node definitions grouped by category
 * @param framework - The target framework (defaults to PyTorch)
 * @returns Map of category to node definitions
 */
export declare function getNodeDefinitionsByCategory(framework?: BackendFramework): Map<string, INodeDefinition[]>;
/**
 * Check if a node type exists in the registry
 * @param type - The block type to check
 * @param framework - The target framework (defaults to PyTorch)
 * @returns True if the node type exists
 */
export declare function hasNodeDefinition(type: BlockType, framework?: BackendFramework): boolean;
/**
 * Get all available node types for a framework
 * @param framework - The target framework (defaults to PyTorch)
 * @returns Array of block types
 */
export declare function getAvailableNodeTypes(framework?: BackendFramework): BlockType[];
/**
 * Reset the registry (useful for testing)
 */
export declare function resetRegistry(): void;
export { BackendFramework };
//# sourceMappingURL=registry.d.ts.map