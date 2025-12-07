/**
 * Helper functions for group block expansion and collapse operations
 * These functions provide robust ID mapping and edge rewiring with explicit error handling
 */

import { Node, Edge } from '@xyflow/react'
import { BlockData, GroupBlockDefinition, PortMapping } from './types'
import { toast } from 'sonner'

/**
 * Result of ID mapping operation with bidirectional lookup
 */
export class IdMappingResult {
  constructor(
    public toExpanded: Map<string, string>,
    public toOriginal: Map<string, string>
  ) {}

  /**
   * Get expanded ID for an original ID
   * @throws Error if mapping not found
   */
  getExpandedId(originalId: string): string {
    const expandedId = this.toExpanded.get(originalId)
    if (!expandedId) {
      throw new Error(
        `ID mapping failed: No expanded ID found for original ID "${originalId}". ` +
        `Available mappings: ${Array.from(this.toExpanded.keys()).join(', ')}`
      )
    }
    return expandedId
  }

  /**
   * Get original ID for an expanded ID
   * @throws Error if mapping not found
   */
  getOriginalId(expandedId: string): string {
    const originalId = this.toOriginal.get(expandedId)
    if (!originalId) {
      throw new Error(
        `ID mapping failed: No original ID found for expanded ID "${expandedId}". ` +
        `Available mappings: ${Array.from(this.toOriginal.keys()).join(', ')}`
      )
    }
    return originalId
  }
}

/**
 * Result of edge rewiring operation
 */
export interface EdgeRewiringResult {
  rewiredEdges: Edge[]
  failedEdges: Array<{
    edge: Edge
    reason: string
  }>
  validate(): void
}

/**
 * Create bidirectional ID mapping for internal nodes
 */
export function createIdMapping(
  internalNodes: Node<BlockData>[],
  suffix: string
): IdMappingResult {
  const toExpanded = new Map<string, string>()
  const toOriginal = new Map<string, string>()
  
  const timestamp = Date.now()
  internalNodes.forEach((node, index) => {
    const expandedId = `${node.id}-${suffix}-${timestamp}-${index}`
    toExpanded.set(node.id, expandedId)
    toOriginal.set(expandedId, node.id)
  })
  
  return new IdMappingResult(toExpanded, toOriginal)
}

/**
 * Rewire edges for expansion operation
 * Converts edges connected to group node to connect to internal nodes
 */
export function rewireEdgesForExpansion(
  edges: Edge[],
  groupNodeId: string,
  groupDef: GroupBlockDefinition,
  idMapping: IdMappingResult
): EdgeRewiringResult {
  const rewiredEdges: Edge[] = []
  const failedEdges: Array<{ edge: Edge; reason: string }> = []
  
  const timestamp = Date.now()
  let edgeCounter = 0
  
  edges.forEach(edge => {
    // Incoming edge to group node
    if (edge.target === groupNodeId) {
      const targetHandle = edge.targetHandle || 'default'
      const mapping = groupDef.portMappings.find(m =>
        m.type === 'input' && m.externalPortId === targetHandle
      )
      
      if (!mapping) {
        failedEdges.push({
          edge,
          reason: `No input port mapping found for handle "${targetHandle}"`
        })
        return
      }
      
      try {
        const newTargetId = idMapping.getExpandedId(mapping.internalNodeId)
        rewiredEdges.push({
          ...edge,
          id: `${edge.id}-rewired-${timestamp}-${edgeCounter++}`,
          target: newTargetId,
          targetHandle: mapping.internalPortId
        })
      } catch (error) {
        failedEdges.push({
          edge,
          reason: (error as Error).message
        })
      }
    }
    // Outgoing edge from group node
    else if (edge.source === groupNodeId) {
      const sourceHandle = edge.sourceHandle || 'default'
      const mapping = groupDef.portMappings.find(m =>
        m.type === 'output' && m.externalPortId === sourceHandle
      )
      
      if (!mapping) {
        failedEdges.push({
          edge,
          reason: `No output port mapping found for handle "${sourceHandle}"`
        })
        return
      }
      
      try {
        const newSourceId = idMapping.getExpandedId(mapping.internalNodeId)
        rewiredEdges.push({
          ...edge,
          id: `${edge.id}-rewired-${timestamp}-${edgeCounter++}`,
          source: newSourceId,
          sourceHandle: mapping.internalPortId
        })
      } catch (error) {
        failedEdges.push({
          edge,
          reason: (error as Error).message
        })
      }
    }
    // External edge (not connected to group)
    else {
      rewiredEdges.push(edge)
    }
  })
  
  return {
    rewiredEdges,
    failedEdges,
    validate() {
      if (this.failedEdges.length > 0) {
        const errors = this.failedEdges.map(f =>
          `  - Edge ${f.edge.id}: ${f.reason}`
        ).join('\n')
        throw new Error(`Edge rewiring failed:\n${errors}`)
      }
    }
  }
}

/**
 * Rewire edges for collapse operation
 * Converts edges connected to internal nodes to connect to group node
 */
export function rewireEdgesForCollapse(
  edges: Edge[],
  expandedNodeIds: Set<string>,
  groupNodeId: string,
  groupDef: GroupBlockDefinition,
  idMapping: IdMappingResult
): EdgeRewiringResult {
  const rewiredEdges: Edge[] = []
  const failedEdges: Array<{ edge: Edge; reason: string }> = []
  
  const timestamp = Date.now()
  let edgeCounter = 0
  
  edges.forEach(edge => {
    const sourceIsExpanded = expandedNodeIds.has(edge.source)
    const targetIsExpanded = expandedNodeIds.has(edge.target)
    
    // Internal edge (both nodes expanded) - remove it
    if (sourceIsExpanded && targetIsExpanded) {
      return
    }
    
    // Incoming edge to expanded node
    if (targetIsExpanded && !sourceIsExpanded) {
      try {
        const originalTargetId = idMapping.getOriginalId(edge.target)
        const targetHandle = edge.targetHandle || 'default'
        
        const mapping = groupDef.portMappings.find(m =>
          m.type === 'input' &&
          m.internalNodeId === originalTargetId &&
          m.internalPortId === targetHandle
        )
        
        if (!mapping) {
          failedEdges.push({
            edge,
            reason: `No input port mapping found for internal node "${originalTargetId}" port "${targetHandle}"`
          })
          return
        }
        
        rewiredEdges.push({
          ...edge,
          id: `${edge.id}-collapsed-${timestamp}-${edgeCounter++}`,
          target: groupNodeId,
          targetHandle: mapping.externalPortId
        })
      } catch (error) {
        failedEdges.push({
          edge,
          reason: (error as Error).message
        })
      }
    }
    // Outgoing edge from expanded node
    else if (sourceIsExpanded && !targetIsExpanded) {
      try {
        const originalSourceId = idMapping.getOriginalId(edge.source)
        const sourceHandle = edge.sourceHandle || 'default'
        
        const mapping = groupDef.portMappings.find(m =>
          m.type === 'output' &&
          m.internalNodeId === originalSourceId &&
          m.internalPortId === sourceHandle
        )
        
        if (!mapping) {
          failedEdges.push({
            edge,
            reason: `No output port mapping found for internal node "${originalSourceId}" port "${sourceHandle}"`
          })
          return
        }
        
        rewiredEdges.push({
          ...edge,
          id: `${edge.id}-collapsed-${timestamp}-${edgeCounter++}`,
          source: groupNodeId,
          sourceHandle: mapping.externalPortId
        })
      } catch (error) {
        failedEdges.push({
          edge,
          reason: (error as Error).message
        })
      }
    }
    // External edge (not connected to expanded nodes)
    else {
      rewiredEdges.push(edge)
    }
  })
  
  return {
    rewiredEdges,
    failedEdges,
    validate() {
      if (this.failedEdges.length > 0) {
        const errors = this.failedEdges.map(f =>
          `  - Edge ${f.edge.id}: ${f.reason}`
        ).join('\n')
        throw new Error(`Edge rewiring failed:\n${errors}`)
      }
    }
  }
}

/**
 * Create container node for expanded group block
 */
export function createContainerNode(
  groupNodeId: string,
  groupDef: GroupBlockDefinition,
  expandedNodes: Node[],
  overrides: Record<string, any>
): Node {
  // Calculate bounding box
  const minX = Math.min(...expandedNodes.map(n => n.position.x))
  const minY = Math.min(...expandedNodes.map(n => n.position.y))
  const maxX = Math.max(...expandedNodes.map(n => n.position.x + (n.width || 280)))
  const maxY = Math.max(...expandedNodes.map(n => n.position.y + (n.height || 150)))
  
  const padding = 30
  const containerWidth = maxX - minX + (2 * padding)
  const containerHeight = maxY - minY + (2 * padding)
  
  return {
    id: `${groupNodeId}-container`,
    type: 'expandedGroupContainer',
    position: {
      x: minX - padding,
      y: minY - padding
    },
    data: {
      _expandedFrom: groupNodeId,
      _groupDefinitionId: groupDef.id,
      groupName: groupDef.name,
      groupColor: groupDef.color,
      _instanceConfigOverrides: overrides
    },
    style: {
      width: containerWidth,
      height: containerHeight,
      zIndex: -1
    },
    selectable: false,
    draggable: false
  } as Node
}
