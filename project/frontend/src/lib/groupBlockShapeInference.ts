/**
 * Shape Inference for Group Blocks
 * Handles shape propagation through internal block structures
 */

import { Node, Edge } from '@xyflow/react'
import { BlockData, TensorShape, GroupBlockDefinition, PortMapping, BlockType } from './types'
import { getNodeDefinition, BackendFramework } from './nodes/registry'

export interface ShapeInferenceResult {
  inputShapes: Map<string, TensorShape>  // externalPortId -> shape
  outputShapes: Map<string, TensorShape> // externalPortId -> shape
  errors: string[]
}

/**
 * Compute shapes for a group block given its external input connections
 * @param groupDef - The group block definition
 * @param externalInputShapes - Map of external port IDs to their input shapes
 * @returns Shape inference result with output shapes and any errors
 */
export function computeGroupBlockShapes(
  groupDef: GroupBlockDefinition,
  externalInputShapes: Map<string, TensorShape>
): ShapeInferenceResult {
  const errors: string[] = []
  const inputShapes = new Map<string, TensorShape>()
  const outputShapes = new Map<string, TensorShape>()

  // Step 1: Map external input shapes to internal nodes
  const internalNodeInputShapes = new Map<string, TensorShape>()
  
  groupDef.portMappings
    .filter(m => m.type === 'input')
    .forEach(mapping => {
      const externalShape = externalInputShapes.get(mapping.externalPortId)
      if (externalShape) {
        inputShapes.set(mapping.externalPortId, externalShape)
        // Store the shape as input to the internal node
        internalNodeInputShapes.set(mapping.internalNodeId, externalShape)
      }
    })

  // Step 2: Traverse internal graph and propagate shapes
  const nodeMap = new Map(groupDef.internalNodes.map(n => [n.id, n]))
  const visited = new Set<string>()
  const processingStack = new Set<string>() // Track nodes currently being processed
  const nodeOutputShapes = new Map<string, TensorShape>()

  const processInternalNode = (nodeId: string): void => {
    // Check for cycles
    if (processingStack.has(nodeId)) {
      errors.push(`Cycle detected in group block ${groupDef.name} at node ${nodeId}`)
      return
    }

    // Already processed
    if (visited.has(nodeId)) return

    visited.add(nodeId)
    processingStack.add(nodeId) // Mark as currently processing

    const node = nodeMap.get(nodeId)
    if (!node) return

    // Get node definition
    const nodeDef = getNodeDefinition(node.data.blockType as BlockType, BackendFramework.PyTorch)
    if (!nodeDef) {
      errors.push(`Unknown node type: ${node.data.blockType} in block ${groupDef.name}`)
      return
    }

    // Find incoming edges to this node
    const incomingEdges = groupDef.internalEdges.filter(e => e.target === nodeId)

    // Process dependencies first
    incomingEdges.forEach(edge => {
      processInternalNode(edge.source)
    })

    // Compute output shape for this node
    try {
      // Check if this node receives external input directly
      const externalInputShape = internalNodeInputShapes.get(nodeId)
      
      if (node.data.blockType === 'input') {
        // Input nodes: pass external input as inputShape parameter
        const outputShape = nodeDef.computeOutputShape(externalInputShape, node.data.config)
        if (outputShape) {
          nodeOutputShapes.set(nodeId, outputShape)
        }
      } else if (externalInputShape && incomingEdges.length === 0) {
        // Non-input node receiving external input directly (no internal incoming edges)
        const outputShape = nodeDef.computeOutputShape(externalInputShape, node.data.config)
        if (outputShape) {
          nodeOutputShapes.set(nodeId, outputShape)
        } else {
          errors.push(`Failed to compute output shape for ${node.data.label} in block ${groupDef.name}`)
        }
      } else if (node.data.blockType === 'loss') {
        // Loss nodes: handle multiple semantic inputs (predictions, labels, etc.)
        const lossNodeDef = nodeDef as any
        if (lossNodeDef?.getInputPorts) {
          const requiredPorts = lossNodeDef.getInputPorts(node.data.config)

          // Gather shapes for each required port (handle-aware)
          const portShapes = new Map<string, TensorShape>()

          incomingEdges.forEach(edge => {
            const sourceShape = nodeOutputShapes.get(edge.source)
            const targetHandle = edge.targetHandle || 'default'

            if (sourceShape) {
              portShapes.set(targetHandle, sourceShape)
            }
          })

          // Validate all required ports have shapes
          const allPortsConnected = requiredPorts.every((p: any) => portShapes.has(p.id))

          if (allPortsConnected) {
            // Use first port's shape as primary input for computeOutputShape
            const primaryShape = portShapes.get(requiredPorts[0].id)
            if (primaryShape) {
              const outputShape = nodeDef.computeOutputShape(primaryShape, node.data.config)
              if (outputShape) {
                nodeOutputShapes.set(nodeId, outputShape)
              } else {
                errors.push(`Failed to compute output shape for loss node in block ${groupDef.name}`)
              }
            }
          } else {
            const missingPorts = requiredPorts.filter((p: any) => !portShapes.has(p.id))
            errors.push(`Loss node in block ${groupDef.name} missing connections to: ${missingPorts.map((p: any) => p.label).join(', ')}`)
          }
        } else {
          // Fallback: treat as regular multi-input node
          const inputShapesList: TensorShape[] = incomingEdges
            .map(edge => nodeOutputShapes.get(edge.source))
            .filter((shape): shape is TensorShape => shape !== undefined)

          if (inputShapesList.length > 0) {
            const outputShape = nodeDef.computeOutputShape(inputShapesList[0], node.data.config)
            if (outputShape) {
              nodeOutputShapes.set(nodeId, outputShape)
            }
          }
        }
      } else if (node.data.blockType === 'concat' || node.data.blockType === 'add') {
        // Multi-input nodes: gather all input shapes
        const inputShapesList: TensorShape[] = []

        for (const edge of incomingEdges) {
          const sourceShape = nodeOutputShapes.get(edge.source)
          if (sourceShape) {
            inputShapesList.push(sourceShape)
          }
        }

        if (inputShapesList.length === incomingEdges.length && inputShapesList.length > 0) {
          // All inputs have shapes
          const nodeDefAny = nodeDef as any
          if (typeof nodeDefAny.computeMultiInputShape === 'function') {
            const outputShape = nodeDefAny.computeMultiInputShape(inputShapesList, node.data.config)
            if (outputShape) {
              nodeOutputShapes.set(nodeId, outputShape)
            } else {
              errors.push(`Failed to compute output shape for ${node.data.blockType} node in block ${groupDef.name}`)
            }
          } else {
            // Fallback: use first input shape
            const outputShape = nodeDef.computeOutputShape(inputShapesList[0], node.data.config)
            if (outputShape) {
              nodeOutputShapes.set(nodeId, outputShape)
            }
          }
        } else if (incomingEdges.length > 0) {
          errors.push(`Missing input shapes for ${node.data.blockType} node in block ${groupDef.name}`)
        }
      } else {
        // Regular nodes: use first incoming edge's shape
        if (incomingEdges.length > 0) {
          const sourceNode = nodeMap.get(incomingEdges[0].source)
          const sourceShape = nodeOutputShapes.get(incomingEdges[0].source)

          if (sourceShape) {
            const outputShape = nodeDef.computeOutputShape(sourceShape, node.data.config)
            if (outputShape) {
              nodeOutputShapes.set(nodeId, outputShape)
            } else {
              errors.push(`Failed to compute output shape for ${node.data.label} in block ${groupDef.name}`)
            }
          }
        }
      }
    } catch (error) {
      errors.push(`Error computing shape for ${node.data.label} in block ${groupDef.name}: ${error}`)
    }

    // Done processing this node
    processingStack.delete(nodeId)
  }

  // Process all internal nodes
  groupDef.internalNodes.forEach(node => {
    processInternalNode(node.id)
  })

  // Step 3: Map internal output shapes to external ports
  groupDef.portMappings
    .filter(m => m.type === 'output')
    .forEach(mapping => {
      const internalShape = nodeOutputShapes.get(mapping.internalNodeId)
      if (internalShape) {
        outputShapes.set(mapping.externalPortId, internalShape)
      } else {
        errors.push(`No output shape computed for port ${mapping.externalPortLabel} in block ${groupDef.name}`)
      }
    })

  return {
    inputShapes,
    outputShapes,
    errors
  }
}

/**
 * Validate shape compatibility within a group block
 * @param groupDef - The group block definition
 * @returns Array of validation error messages
 */
export function validateGroupBlockShapes(groupDef: GroupBlockDefinition): string[] {
  const errors: string[] = []
  const nodeMap = new Map(groupDef.internalNodes.map(n => [n.id, n]))

  // Check each internal edge for shape compatibility
  groupDef.internalEdges.forEach(edge => {
    const sourceNode = nodeMap.get(edge.source)
    const targetNode = nodeMap.get(edge.target)

    if (!sourceNode || !targetNode) {
      errors.push(`Invalid edge in block ${groupDef.name}: missing source or target node`)
      return
    }

    const sourceNodeDef = getNodeDefinition(sourceNode.data.blockType as BlockType, BackendFramework.PyTorch)
    const targetNodeDef = getNodeDefinition(targetNode.data.blockType as BlockType, BackendFramework.PyTorch)

    if (!sourceNodeDef || !targetNodeDef) {
      return
    }

    // Validate connection if both nodes have shapes
    if (sourceNode.data.outputShape && targetNode.data.inputShape) {
      const validationError = targetNodeDef.validateIncomingConnection(
        sourceNode.data.blockType as BlockType,
        sourceNode.data.outputShape,
        targetNode.data.config
      )

      if (validationError) {
        errors.push(`Shape mismatch in block ${groupDef.name}: ${validationError}`)
      }
    }
  })

  return errors
}
