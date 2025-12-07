import { describe, it, expect } from 'vitest'
import { Edge } from '@xyflow/react'
import {
  validateConnectivity,
  detectCycles,
  validateBlockName,
  validatePortSelection,
  validateBlockCreation
} from './blockValidation'

describe('Block Validation', () => {
  describe('validateConnectivity', () => {
    it('should reject empty selection', () => {
      const result = validateConnectivity([], [])
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('No nodes selected')
    })

    it('should reject single node selection', () => {
      const result = validateConnectivity(['node1'], [])
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Please select at least 2 nodes to create a block')
    })

    it('should accept connected nodes', () => {
      const edges: Edge[] = [
        { id: 'e1', source: 'node1', target: 'node2' }
      ]
      const result = validateConnectivity(['node1', 'node2'], edges)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    it('should reject disconnected nodes', () => {
      const edges: Edge[] = [
        { id: 'e1', source: 'node1', target: 'node2' }
      ]
      const result = validateConnectivity(['node1', 'node2', 'node3'], edges)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Selected nodes must form a connected graph')
    })

    it('should accept complex connected graph', () => {
      const edges: Edge[] = [
        { id: 'e1', source: 'node1', target: 'node2' },
        { id: 'e2', source: 'node2', target: 'node3' },
        { id: 'e3', source: 'node3', target: 'node4' }
      ]
      const result = validateConnectivity(['node1', 'node2', 'node3', 'node4'], edges)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    it('should handle branching connected graph', () => {
      const edges: Edge[] = [
        { id: 'e1', source: 'node1', target: 'node2' },
        { id: 'e2', source: 'node1', target: 'node3' },
        { id: 'e3', source: 'node2', target: 'node4' },
        { id: 'e4', source: 'node3', target: 'node4' }
      ]
      const result = validateConnectivity(['node1', 'node2', 'node3', 'node4'], edges)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })
  })

  describe('detectCycles', () => {
    it('should accept empty selection', () => {
      const result = detectCycles([], [])
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    it('should accept acyclic graph', () => {
      const edges: Edge[] = [
        { id: 'e1', source: 'node1', target: 'node2' },
        { id: 'e2', source: 'node2', target: 'node3' }
      ]
      const result = detectCycles(['node1', 'node2', 'node3'], edges)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    it('should detect simple cycle', () => {
      const edges: Edge[] = [
        { id: 'e1', source: 'node1', target: 'node2' },
        { id: 'e2', source: 'node2', target: 'node1' }
      ]
      const result = detectCycles(['node1', 'node2'], edges)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Selected layers contain circular dependencies')
    })

    it('should detect complex cycle', () => {
      const edges: Edge[] = [
        { id: 'e1', source: 'node1', target: 'node2' },
        { id: 'e2', source: 'node2', target: 'node3' },
        { id: 'e3', source: 'node3', target: 'node1' }
      ]
      const result = detectCycles(['node1', 'node2', 'node3'], edges)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Selected layers contain circular dependencies')
    })

    it('should accept DAG with multiple paths', () => {
      const edges: Edge[] = [
        { id: 'e1', source: 'node1', target: 'node2' },
        { id: 'e2', source: 'node1', target: 'node3' },
        { id: 'e3', source: 'node2', target: 'node4' },
        { id: 'e4', source: 'node3', target: 'node4' }
      ]
      const result = detectCycles(['node1', 'node2', 'node3', 'node4'], edges)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })
  })

  describe('validateBlockName', () => {
    it('should reject empty name', () => {
      const result = validateBlockName('', [])
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Name is required')
    })

    it('should reject whitespace-only name', () => {
      const result = validateBlockName('   ', [])
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Name is required')
    })

    it('should reject name longer than 50 characters', () => {
      const longName = 'a'.repeat(51)
      const result = validateBlockName(longName, [])
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Block name must be 50 characters or less')
    })

    it('should accept name with exactly 50 characters', () => {
      const name = 'a'.repeat(50)
      const result = validateBlockName(name, [])
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    it('should reject name with invalid characters', () => {
      const result = validateBlockName('my block!', [])
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Block name must contain only letters, numbers, underscores, and hyphens')
    })

    it('should accept name with letters, numbers, underscores, and hyphens', () => {
      const result = validateBlockName('my_block-123', [])
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    it('should reject duplicate name', () => {
      const existingNames = ['existing_block', 'another_block']
      const result = validateBlockName('existing_block', existingNames)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('A block with this name already exists')
    })

    it('should accept unique name', () => {
      const existingNames = ['existing_block', 'another_block']
      const result = validateBlockName('new_block', existingNames)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    it('should reject name with spaces', () => {
      const result = validateBlockName('my block', [])
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Block name must contain only letters, numbers, underscores, and hyphens')
    })

    it('should reject name with special characters', () => {
      const result = validateBlockName('my@block', [])
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Block name must contain only letters, numbers, underscores, and hyphens')
    })
  })

  describe('validatePortSelection', () => {
    it('should reject zero ports', () => {
      const result = validatePortSelection(0)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('At least one port must be exposed')
    })

    it('should accept one or more ports', () => {
      const result = validatePortSelection(1)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    it('should accept multiple ports', () => {
      const result = validatePortSelection(5)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })
  })

  describe('validateBlockCreation', () => {
    it('should combine all validation errors', () => {
      const edges: Edge[] = []
      const result = validateBlockCreation(
        ['node1'], // Only one node - invalid
        edges,
        '', // Empty name - invalid
        [],
        0 // No ports - invalid
      )
      expect(result.isValid).toBe(false)
      expect(result.errors.length).toBeGreaterThan(0)
    })

    it('should pass with valid inputs', () => {
      const edges: Edge[] = [
        { id: 'e1', source: 'node1', target: 'node2' }
      ]
      const result = validateBlockCreation(
        ['node1', 'node2'],
        edges,
        'valid_block_name',
        [],
        2
      )
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    it('should detect cycles and report error', () => {
      const edges: Edge[] = [
        { id: 'e1', source: 'node1', target: 'node2' },
        { id: 'e2', source: 'node2', target: 'node1' }
      ]
      const result = validateBlockCreation(
        ['node1', 'node2'],
        edges,
        'valid_name',
        [],
        1
      )
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Selected layers contain circular dependencies')
    })

    it('should detect disconnected nodes', () => {
      const edges: Edge[] = []
      const result = validateBlockCreation(
        ['node1', 'node2', 'node3'],
        edges,
        'valid_name',
        [],
        1
      )
      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('Selected nodes must form a connected graph')
    })
  })
})
