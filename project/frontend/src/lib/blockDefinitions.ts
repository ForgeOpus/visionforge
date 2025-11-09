/**
 * Block Definitions - Legacy Compatibility Layer
 * 
 * This file maintains backward compatibility with the original blockDefinitions structure.
 * All definitions now come from the modular node definition system in ./nodes/
 * 
 * @deprecated This file re-exports from the legacy adapter for backward compatibility.
 * New code should import directly from './nodes/registry' instead.
 * 
 * Migration guide:
 * - OLD: import { getBlockDefinition } from './blockDefinitions'
 * - NEW: import { getNodeDefinition } from './nodes/registry'
 */

export {
  blockDefinitions,
  getBlockDefinition,
  getBlocksByCategory,
  validateBlockConnection,
  allowsMultipleInputs
} from './legacy/blockDefinitionsAdapter'

/**
 * For modern usage, prefer importing from the registry:
 * 
 * import { 
 *   getNodeDefinition, 
 *   getAllNodeDefinitions, 
 *   BackendFramework 
 * } from './nodes/registry'
 */
