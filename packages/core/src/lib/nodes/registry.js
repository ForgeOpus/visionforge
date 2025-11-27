/**
 * Node Definition Registry
 * Dynamically loads and manages node definitions for all supported frameworks
 */
import { BackendFramework } from './contracts';
// Import all PyTorch nodes
import * as PyTorchNodes from './definitions/pytorch';
// Import all TensorFlow nodes
import * as TensorFlowNodes from './definitions/tensorflow';
/**
 * Registry cache - stores instantiated node definitions
 */
const registryCache = {
    [BackendFramework.PyTorch]: {},
    [BackendFramework.TensorFlow]: {}
};
/**
 * Category cache - memoizes category groupings for performance
 */
const categoryCache = new Map();
/**
 * Initialization flag to ensure registry is built only once
 */
let isInitialized = false;
/**
 * Initialize the registry by instantiating all node definition classes
 */
function initializeRegistry() {
    if (isInitialized)
        return;
    // Register PyTorch nodes
    Object.values(PyTorchNodes).forEach((NodeClass) => {
        const instance = new NodeClass();
        registryCache[BackendFramework.PyTorch][instance.metadata.type] = instance;
    });
    // Register TensorFlow nodes
    Object.values(TensorFlowNodes).forEach((NodeClass) => {
        const instance = new NodeClass();
        registryCache[BackendFramework.TensorFlow][instance.metadata.type] = instance;
    });
    isInitialized = true;
}
/**
 * Get a specific node definition by type and framework
 * @param type - The block type to retrieve
 * @param framework - The target framework (defaults to PyTorch)
 * @returns The node definition or undefined if not found
 */
export function getNodeDefinition(type, framework = BackendFramework.PyTorch) {
    initializeRegistry();
    return registryCache[framework]?.[type];
}
/**
 * Get all node definitions for a specific framework
 * @param framework - The target framework (defaults to PyTorch)
 * @returns Array of all node definitions for the framework
 */
export function getAllNodeDefinitions(framework = BackendFramework.PyTorch) {
    initializeRegistry();
    return Object.values(registryCache[framework] || {});
}
/**
 * Get node definitions grouped by category
 * @param framework - The target framework (defaults to PyTorch)
 * @returns Map of category to node definitions
 */
export function getNodeDefinitionsByCategory(framework = BackendFramework.PyTorch) {
    // Check cache first
    if (categoryCache.has(framework)) {
        return categoryCache.get(framework);
    }
    // Build category map
    const allNodes = getAllNodeDefinitions(framework);
    const byCategory = new Map();
    allNodes.forEach(node => {
        const category = node.metadata.category;
        if (!byCategory.has(category)) {
            byCategory.set(category, []);
        }
        byCategory.get(category).push(node);
    });
    // Cache the result
    categoryCache.set(framework, byCategory);
    return byCategory;
}
/**
 * Check if a node type exists in the registry
 * @param type - The block type to check
 * @param framework - The target framework (defaults to PyTorch)
 * @returns True if the node type exists
 */
export function hasNodeDefinition(type, framework = BackendFramework.PyTorch) {
    initializeRegistry();
    return !!registryCache[framework]?.[type];
}
/**
 * Get all available node types for a framework
 * @param framework - The target framework (defaults to PyTorch)
 * @returns Array of block types
 */
export function getAvailableNodeTypes(framework = BackendFramework.PyTorch) {
    initializeRegistry();
    return Object.keys(registryCache[framework] || {});
}
/**
 * Reset the registry (useful for testing)
 */
export function resetRegistry() {
    isInitialized = false;
    categoryCache.clear();
    Object.keys(registryCache).forEach(framework => {
        registryCache[framework] = {};
    });
}
// Export the framework enum for convenience
export { BackendFramework };
