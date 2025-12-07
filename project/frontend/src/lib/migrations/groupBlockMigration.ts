/**
 * Data migration utility for group blocks
 * Adds isExpanded property to existing group blocks
 */

import { Node } from '@xyflow/react'
import { BlockData, GroupBlockData } from '../types'

export interface MigrationResult {
  migratedCount: number
  errors: string[]
  warnings: string[]
}

/**
 * Migrate group blocks to add isExpanded property
 * @param nodes Array of nodes to migrate
 * @returns Migration result with count and any errors
 */
export function migrateGroupBlocks(nodes: Node<BlockData>[]): {
  migratedNodes: Node<BlockData>[]
  result: MigrationResult
} {
  const result: MigrationResult = {
    migratedCount: 0,
    errors: [],
    warnings: []
  }

  const migratedNodes = nodes.map(node => {
    // Only process group blocks
    if (node.data.blockType !== 'group') {
      return node
    }

    const groupData = node.data as GroupBlockData

    // Check if already migrated
    if ('isExpanded' in groupData) {
      result.warnings.push(`Node ${node.id} already has isExpanded property`)
      return node
    }

    // Add isExpanded property with default value false
    const migratedNode: Node<GroupBlockData> = {
      ...node,
      data: {
        ...groupData,
        isExpanded: false
      }
    }

    result.migratedCount++
    return migratedNode
  })

  return { migratedNodes, result }
}

/**
 * Validate all group blocks after migration
 * @param nodes Array of nodes to validate
 * @returns Validation result
 */
export function validateGroupBlocks(nodes: Node<BlockData>[]): {
  isValid: boolean
  errors: string[]
} {
  const errors: string[] = []

  nodes.forEach(node => {
    if (node.data.blockType === 'group') {
      const groupData = node.data as GroupBlockData

      // Check required properties
      if (!('isExpanded' in groupData)) {
        errors.push(`Node ${node.id}: Missing isExpanded property`)
      }

      if (!groupData.groupDefinitionId) {
        errors.push(`Node ${node.id}: Missing groupDefinitionId`)
      }

      // Validate isExpanded is boolean
      if (typeof groupData.isExpanded !== 'boolean') {
        errors.push(`Node ${node.id}: isExpanded must be boolean, got ${typeof groupData.isExpanded}`)
      }
    }
  })

  return {
    isValid: errors.length === 0,
    errors
  }
}

/**
 * Run migration and log results
 * @param nodes Array of nodes to migrate
 * @returns Migrated nodes
 */
export function runMigration(nodes: Node<BlockData>[]): Node<BlockData>[] {
  console.log('Starting group block migration...')

  const { migratedNodes, result } = migrateGroupBlocks(nodes)

  // Log migration results
  console.log(`Migration complete: ${result.migratedCount} node(s) migrated`)

  if (result.warnings.length > 0) {
    console.warn('Migration warnings:', result.warnings)
  }

  if (result.errors.length > 0) {
    console.error('Migration errors:', result.errors)
  }

  // Validate after migration
  const validation = validateGroupBlocks(migratedNodes)

  if (!validation.isValid) {
    console.error('Validation failed after migration:', validation.errors)
    throw new Error('Group block migration validation failed')
  }

  console.log('All group blocks validated successfully')

  return migratedNodes
}
