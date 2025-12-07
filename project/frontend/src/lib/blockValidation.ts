import { Node, Edge } from '@xyflow/react'
import { BlockData } from './types'

export interface ValidationResult {
  isValid: boolean
  errors: string[]
  warnings: string[]
}

/**
 * Validates that selected nodes form a connected subgraph
 * Requirements: 7.1
 */
export function validateConnectivity(
  selectedNodeIds: string[],
  edges: Edge[]
): ValidationResult {
  if (selectedNodeIds.length === 0) {
    return {
      isValid: false,
      errors: ['No nodes selected'],
      warnings: []
    }
  }

  if (selectedNodeIds.length === 1) {
    return {
      isValid: false,
      errors: ['Please select at least 2 nodes to create a block'],
      warnings: []
    }
  }

  // Build adjacency list for selected nodes
  const adjacency = new Map<string, Set<string>>()
  selectedNodeIds.forEach(id => adjacency.set(id, new Set()))

  // Add edges between selected nodes (undirected graph for connectivity check)
  edges.forEach(edge => {
    const sourceInSelection = selectedNodeIds.includes(edge.source)
    const targetInSelection = selectedNodeIds.includes(edge.target)

    if (sourceInSelection && targetInSelection) {
      adjacency.get(edge.source)?.add(edge.target)
      adjacency.get(edge.target)?.add(edge.source) // Treat as undirected
    }
  })

  // BFS to check connectivity
  const visited = new Set<string>()
  const queue: string[] = [selectedNodeIds[0]]
  visited.add(selectedNodeIds[0])

  while (queue.length > 0) {
    const current = queue.shift()!
    const neighbors = adjacency.get(current) || new Set()

    neighbors.forEach(neighbor => {
      if (!visited.has(neighbor)) {
        visited.add(neighbor)
        queue.push(neighbor)
      }
    })
  }

  // Check if all nodes were visited
  if (visited.size !== selectedNodeIds.length) {
    return {
      isValid: false,
      errors: ['Selected nodes must form a connected graph'],
      warnings: []
    }
  }

  return {
    isValid: true,
    errors: [],
    warnings: []
  }
}

/**
 * Detects cycles in the selected subgraph using DFS
 * Requirements: 7.2
 */
export function detectCycles(
  selectedNodeIds: string[],
  edges: Edge[]
): ValidationResult {
  if (selectedNodeIds.length === 0) {
    return {
      isValid: true,
      errors: [],
      warnings: []
    }
  }

  // Build adjacency list for directed graph
  const adjacency = new Map<string, string[]>()
  selectedNodeIds.forEach(id => adjacency.set(id, []))

  edges.forEach(edge => {
    const sourceInSelection = selectedNodeIds.includes(edge.source)
    const targetInSelection = selectedNodeIds.includes(edge.target)

    if (sourceInSelection && targetInSelection) {
      adjacency.get(edge.source)?.push(edge.target)
    }
  })

  // DFS with recursion stack to detect cycles
  const visited = new Set<string>()
  const recursionStack = new Set<string>()

  function hasCycleDFS(nodeId: string): boolean {
    visited.add(nodeId)
    recursionStack.add(nodeId)

    const neighbors = adjacency.get(nodeId) || []
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        if (hasCycleDFS(neighbor)) {
          return true
        }
      } else if (recursionStack.has(neighbor)) {
        // Back edge found - cycle detected
        return true
      }
    }

    recursionStack.delete(nodeId)
    return false
  }

  // Check for cycles starting from each unvisited node
  for (const nodeId of selectedNodeIds) {
    if (!visited.has(nodeId)) {
      if (hasCycleDFS(nodeId)) {
        return {
          isValid: false,
          errors: ['Selected layers contain circular dependencies'],
          warnings: []
        }
      }
    }
  }

  return {
    isValid: true,
    errors: [],
    warnings: []
  }
}

/**
 * Validates block name according to requirements
 * Requirements: 7.3, 7.4, 7.5
 */
export function validateBlockName(
  name: string,
  existingNames: string[]
): ValidationResult {
  const errors: string[] = []
  const warnings: string[] = []

  // Check if name is empty
  const trimmedName = name.trim()
  if (!trimmedName) {
    errors.push('Name is required')
    return { isValid: false, errors, warnings }
  }

  // Check length (Requirement 7.5)
  if (trimmedName.length > 50) {
    errors.push('Block name must be 50 characters or less')
  }

  // Check for invalid characters (Requirement 7.4)
  // Only allow letters, numbers, underscores, and hyphens
  const validNamePattern = /^[a-zA-Z0-9_-]+$/
  if (!validNamePattern.test(trimmedName)) {
    errors.push('Block name must contain only letters, numbers, underscores, and hyphens')
  }

  // Check for uniqueness (Requirement 7.3)
  if (existingNames.includes(trimmedName)) {
    errors.push('A block with this name already exists')
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings
  }
}

/**
 * Validates that at least one port is exposed
 * Requirements: 2.5
 */
export function validatePortSelection(selectedPortCount: number): ValidationResult {
  if (selectedPortCount === 0) {
    return {
      isValid: false,
      errors: ['At least one port must be exposed'],
      warnings: []
    }
  }

  return {
    isValid: true,
    errors: [],
    warnings: []
  }
}

/**
 * Comprehensive validation for block creation
 * Combines all validation checks
 */
export function validateBlockCreation(
  selectedNodeIds: string[],
  edges: Edge[],
  blockName: string,
  existingBlockNames: string[],
  selectedPortCount: number
): ValidationResult {
  const errors: string[] = []
  const warnings: string[] = []

  // Validate connectivity
  const connectivityResult = validateConnectivity(selectedNodeIds, edges)
  errors.push(...connectivityResult.errors)
  warnings.push(...connectivityResult.warnings)

  // Only check for cycles if nodes are connected
  if (connectivityResult.isValid) {
    const cycleResult = detectCycles(selectedNodeIds, edges)
    errors.push(...cycleResult.errors)
    warnings.push(...cycleResult.warnings)
  }

  // Validate block name
  const nameResult = validateBlockName(blockName, existingBlockNames)
  errors.push(...nameResult.errors)
  warnings.push(...nameResult.warnings)

  // Validate port selection
  const portResult = validatePortSelection(selectedPortCount)
  errors.push(...portResult.errors)
  warnings.push(...portResult.warnings)

  return {
    isValid: errors.length === 0,
    errors,
    warnings
  }
}
