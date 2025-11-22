import { Node, Edge } from '@xyflow/react'
import { BlockData, Project } from './types'

const DESIGNS_STORAGE_KEY = 'visionforge_local_designs'

export interface LocalDesign {
  id: string
  name: string
  description: string
  framework: 'pytorch' | 'tensorflow'
  nodes: Node<BlockData>[]
  edges: Edge[]
  createdAt: number
  updatedAt: number
}

/**
 * Get all locally stored designs
 */
export function getLocalDesigns(): LocalDesign[] {
  try {
    const stored = localStorage.getItem(DESIGNS_STORAGE_KEY)
    if (!stored) return []
    return JSON.parse(stored)
  } catch (error) {
    console.error('Failed to load local designs:', error)
    return []
  }
}

/**
 * Save a design to local storage
 */
export function saveDesignToLocal(design: LocalDesign): void {
  try {
    const designs = getLocalDesigns()
    const existingIndex = designs.findIndex(d => d.id === design.id)

    if (existingIndex >= 0) {
      designs[existingIndex] = {
        ...design,
        updatedAt: Date.now()
      }
    } else {
      designs.push({
        ...design,
        createdAt: Date.now(),
        updatedAt: Date.now()
      })
    }

    localStorage.setItem(DESIGNS_STORAGE_KEY, JSON.stringify(designs))
  } catch (error) {
    console.error('Failed to save design to local storage:', error)
    throw new Error('Failed to save design. Storage may be full.')
  }
}

/**
 * Load a design from local storage by ID
 */
export function loadDesignFromLocal(id: string): LocalDesign | null {
  const designs = getLocalDesigns()
  return designs.find(d => d.id === id) || null
}

/**
 * Delete a design from local storage
 */
export function deleteLocalDesign(id: string): void {
  try {
    const designs = getLocalDesigns()
    const filtered = designs.filter(d => d.id !== id)
    localStorage.setItem(DESIGNS_STORAGE_KEY, JSON.stringify(filtered))
  } catch (error) {
    console.error('Failed to delete local design:', error)
    throw new Error('Failed to delete design')
  }
}

/**
 * Generate a unique ID for a new design
 */
export function generateDesignId(): string {
  return `local-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
}

/**
 * Convert LocalDesign to Project format
 */
export function localDesignToProject(design: LocalDesign): Project {
  return {
    id: design.id,
    name: design.name,
    description: design.description,
    framework: design.framework,
    nodes: design.nodes,
    edges: design.edges,
    createdAt: design.createdAt,
    updatedAt: design.updatedAt,
  }
}

/**
 * Convert Project to LocalDesign format
 */
export function projectToLocalDesign(project: Project): LocalDesign {
  return {
    id: project.id,
    name: project.name,
    description: project.description,
    framework: project.framework,
    nodes: project.nodes,
    edges: project.edges,
    createdAt: project.createdAt,
    updatedAt: project.updatedAt,
  }
}

/**
 * Check available storage space (approximate)
 */
export function checkStorageSpace(): { used: number; available: boolean } {
  try {
    const stored = localStorage.getItem(DESIGNS_STORAGE_KEY) || ''
    const usedBytes = new Blob([stored]).size

    // Try to estimate if we have space (localStorage typically has 5-10MB limit)
    const testKey = '__storage_test__'
    const testData = 'x'.repeat(1024 * 100) // 100KB test

    try {
      localStorage.setItem(testKey, testData)
      localStorage.removeItem(testKey)
      return { used: usedBytes, available: true }
    } catch {
      return { used: usedBytes, available: false }
    }
  } catch (error) {
    return { used: 0, available: false }
  }
}
