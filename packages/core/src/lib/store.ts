import { create } from 'zustand'
import { Node, Edge, Connection } from '@xyflow/react'
import { BlockData, Project, ValidationError, TensorShape, BlockType } from './types'
import { getNodeDefinition, BackendFramework } from './nodes/registry'
import { arePortsCompatible } from './nodes/ports'
import {
  NodeValidationState,
} from './validation'
import { hasSymbolicDims } from './validation/matchers'

interface HistoryState {
  nodes: Node<BlockData>[]
  edges: Edge[]
}

interface ModelBuilderState {
  nodes: Node<BlockData>[]
  edges: Edge[]
  selectedNodeId: string | null
  selectedEdgeId: string | null
  recentlyUsedNodes: BlockType[]
  validationErrors: ValidationError[]
  currentProject: Project | null

  // History for undo/redo
  past: HistoryState[]
  future: HistoryState[]

  setNodes: (nodes: Node<BlockData>[]) => void
  setEdges: (edges: Edge[]) => void
  addNode: (node: Node<BlockData>) => void
  updateNode: (id: string, data: Partial<BlockData>) => void
  removeNode: (id: string) => void
  duplicateNode: (id: string) => void
  addEdge: (edge: Edge) => void
  removeEdge: (id: string) => void
  setSelectedNodeId: (id: string | null) => void
  setSelectedEdgeId: (id: string | null) => void
  trackRecentlyUsedNode: (nodeType: BlockType) => void

  validateConnection: (connection: Connection) => boolean
  validateArchitecture: () => ValidationError[]
  inferDimensions: () => void

  undo: () => void
  redo: () => void
  canUndo: () => boolean
  canRedo: () => boolean

  createProject: (name: string, description: string, framework: 'pytorch' | 'tensorflow') => void
  saveProject: () => void
  loadProject: (project: Project) => void
  updateProjectInfo: (name: string, description: string) => void

  reset: () => void
}

const MAX_HISTORY = 10

// Helper to save current state to history
const saveHistory = (state: ModelBuilderState) => {
  const currentState: HistoryState = {
    nodes: JSON.parse(JSON.stringify(state.nodes)),
    edges: JSON.parse(JSON.stringify(state.edges))
  }
  
  const newPast = [...state.past, currentState].slice(-MAX_HISTORY)
  
  return {
    past: newPast,
    future: [] // Clear future on new action
  }
}

export const useModelBuilderStore = create<ModelBuilderState>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNodeId: null,
  selectedEdgeId: null,
  recentlyUsedNodes: [],
  validationErrors: [],
  currentProject: null,
  past: [],
  future: [],

  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),

  addNode: (node) => {
    const state = get()
    const historyUpdate = saveHistory(state)

    // Track recently used node
    get().trackRecentlyUsedNode(node.data.blockType as BlockType)

    // Add node to canvas (project will be created on save)
    set((state) => ({
      nodes: [...state.nodes, node],
      ...historyUpdate
    }))
  },

  updateNode: (id, data) => {
    const state = get()
    const historyUpdate = saveHistory(state)
    
    set((state) => ({
      nodes: state.nodes.map((node) =>
        node.id === id ? { ...node, data: { ...node.data, ...data } } : node
      ),
      ...historyUpdate
    }))
    
    get().inferDimensions()
  },

  removeNode: (id) => {
    const state = get()
    const historyUpdate = saveHistory(state)
    
    set((state) => ({
      nodes: state.nodes.filter((node) => node.id !== id),
      edges: state.edges.filter((edge) => edge.source !== id && edge.target !== id),
      selectedNodeId: state.selectedNodeId === id ? null : state.selectedNodeId,
      ...historyUpdate
    }))
  },

  addEdge: (edge) => {
    const state = get()
    const historyUpdate = saveHistory(state)
    
    set((state) => ({
      edges: [...state.edges, edge],
      ...historyUpdate
    }))
    
    const { nodes, edges } = get()
    const targetNode = nodes.find((n) => n.id === edge.target)
    const sourceNode = nodes.find((n) => n.id === edge.source)
    
    if (targetNode && sourceNode?.data.outputShape) {
      const targetNodeDef = getNodeDefinition(
        targetNode.data.blockType as BlockType,
        BackendFramework.PyTorch
      )
      const sourceShape = sourceNode.data.outputShape
      
      if (targetNode.data.blockType === 'linear' && sourceShape.dims.length !== 2) {
        const updatedNodes = nodes.map((node) => {
          if (node.id === targetNode.id && sourceShape.dims.length > 2) {
            return {
              ...node,
              data: {
                ...node.data,
                config: {
                  ...node.data.config
                }
              }
            }
          }
          return node
        })
        set({ nodes: updatedNodes })
      }
      
      if (targetNode.data.blockType === 'conv2d' && !targetNode.data.config.out_channels) {
        const updatedNodes = nodes.map((node) => {
          if (node.id === targetNode.id) {
            const inferredChannels = sourceShape.dims.length >= 2 ? sourceShape.dims[1] : 64
            return {
              ...node,
              data: {
                ...node.data,
                config: {
                  ...node.data.config,
                  out_channels: inferredChannels
                }
              }
            }
          }
          return node
        })
        set({ nodes: updatedNodes })
      }
      
      if (!targetNode.data.inputShape) {
        const updatedNodes = nodes.map((node) => {
          if (node.id === targetNode.id) {
            return {
              ...node,
              data: {
                ...node.data,
                inputShape: sourceShape
              }
            }
          }
          return node
        })
        set({ nodes: updatedNodes })
      }
    }
    
    setTimeout(() => get().inferDimensions(), 0)
  },

  removeEdge: (id) => {
    const state = get()
    const historyUpdate = saveHistory(state)
    
    set((state) => ({
      edges: state.edges.filter((edge) => edge.id !== id),
      ...historyUpdate
    }))
  },

  setSelectedNodeId: (id) => set({ selectedNodeId: id, selectedEdgeId: null }),
  setSelectedEdgeId: (id) => set({ selectedEdgeId: id, selectedNodeId: null }),

  trackRecentlyUsedNode: (nodeType) => {
    const { recentlyUsedNodes } = get()
    const filtered = recentlyUsedNodes.filter(t => t !== nodeType)
    const updated = [nodeType, ...filtered].slice(0, 5) // Keep last 5
    set({ recentlyUsedNodes: updated })
  },

  duplicateNode: (id) => {
    const state = get()
    const historyUpdate = saveHistory(state)

    const nodeToDuplicate = state.nodes.find(n => n.id === id)
    if (!nodeToDuplicate) return

    const newNode: Node<BlockData> = {
      ...nodeToDuplicate,
      id: `${nodeToDuplicate.data.blockType}-${Date.now()}`,
      position: {
        x: nodeToDuplicate.position.x + 50,
        y: nodeToDuplicate.position.y + 50
      },
      data: {
        ...nodeToDuplicate.data,
        config: { ...nodeToDuplicate.data.config }
      }
    }

    set((state) => ({
      nodes: [...state.nodes, newNode],
      ...historyUpdate
    }))
  },

  validateConnection: (connection) => {
    const { nodes, edges } = get()
    
    const targetNode = nodes.find((n) => n.id === connection.target)
    if (!targetNode) return false
    
    const sourceNode = nodes.find((n) => n.id === connection.source)
    if (!sourceNode) return false
    
    // Get node definitions
    const targetNodeDef = getNodeDefinition(
      targetNode.data.blockType as BlockType,
      BackendFramework.PyTorch
    )
    const sourceNodeDef = getNodeDefinition(
      sourceNode.data.blockType as BlockType,
      BackendFramework.PyTorch
    )
    
    if (!targetNodeDef || !sourceNodeDef) return false
    
    // === NEW: Validate source handle exists ===
    const sourceHandleId = connection.sourceHandle || 'default'
    const sourcePorts = sourceNodeDef.getOutputPorts(sourceNode.data.config)
    const sourcePort = sourcePorts.find(p => p.id === sourceHandleId)
    
    if (!sourcePort) {
      console.error(`Source handle ${sourceHandleId} not found on ${sourceNode.data.blockType}`)
      return false
    }
    
    // === NEW: Validate target handle exists ===
    const targetHandleId = connection.targetHandle || 'default'
    const targetPorts = targetNodeDef.getInputPorts(targetNode.data.config)
    const targetPort = targetPorts.find(p => p.id === targetHandleId)
    
    if (!targetPort) {
      console.error(`Target handle ${targetHandleId} not found on ${targetNode.data.blockType}`)
      return false
    }
    
    // === NEW: Check if target handle already has a connection ===
    // Allow multiple connections to the same handle for merge nodes (concat, add)
    const isMergeNode = targetNode.data.blockType === 'concat' || targetNode.data.blockType === 'add'
    
    if (!isMergeNode) {
      const handleOccupied = edges.some(e => 
        e.target === connection.target && 
        (e.targetHandle || 'default') === targetHandleId
      )
      
      if (handleOccupied) {
        console.warn(`Target handle ${targetHandleId} already connected`)
        return false
      }
    }
    
    // === NEW: Semantic validation - check port compatibility ===
    if (!arePortsCompatible(sourcePort, targetPort)) {
      console.error(`Port semantic mismatch: ${sourcePort.semantic} -> ${targetPort.semantic}`)
      return false
    }
    
    // === NEW: Real-time loss node input count validation ===
    if (targetNode.data.blockType === 'loss') {
      const requiredPorts = targetPorts
      const existingConnections = edges.filter(e => e.target === connection.target)
      
      // Count how many connections exist after this one would be added
      const totalConnectionsAfter = existingConnections.length + 1
      
      if (totalConnectionsAfter > requiredPorts.length) {
        const lossType = targetNode.data.config.loss_type || 'cross_entropy'
        console.error(
          `Loss function "${lossType}" only accepts ${requiredPorts.length} inputs ` +
          `(${requiredPorts.map(p => p.label).join(', ')}). Cannot add more.`
        )
        return false
      }
    }
    
    // Check if target allows multiple inputs (for backwards compatibility)
    const allowsMultiple = targetNode.data.blockType === 'concat' || targetNode.data.blockType === 'add' || targetNode.data.blockType === 'loss'
    if (!allowsMultiple) {
      const hasExistingInput = edges.some((e) => e.target === connection.target)
      if (hasExistingInput) return false
    }
    
    // Use the node definition validation method
    const validationError = targetNodeDef.validateIncomingConnection(
      sourceNode.data.blockType as BlockType,
      sourceNode.data.outputShape,
      targetNode.data.config
    )
    
    if (validationError) {
      // Could show toast here with the error message if desired
      console.warn('Connection validation failed:', validationError)
      return false
    }
    
    // Special validation for add blocks - all inputs must have same shape
    if (targetNode.data.blockType === 'add') {
      const incomingEdges = edges.filter((e) => e.target === connection.target)
      if (incomingEdges.length > 0) {
        const firstSourceNode = nodes.find((n) => n.id === incomingEdges[0].source)
        if (firstSourceNode?.data.outputShape && sourceNode.data.outputShape) {
          const firstShape = firstSourceNode.data.outputShape
          const sourceShape = sourceNode.data.outputShape
          if (firstShape.dims.length !== sourceShape.dims.length) {
            return false
          }
        }
      }
    }
    
    return true
  },

  validateArchitecture: () => {
    const { nodes, edges } = get()
    const errors: ValidationError[] = []
    
    const inputNodes = nodes.filter((n) => n.data.blockType === 'input')
    if (inputNodes.length === 0) {
      errors.push({
        message: 'Architecture must have at least one Input block',
        type: 'error'
      })
    }
    
    nodes.forEach((node) => {
      const hasInput = edges.some((e) => e.target === node.id)
      const hasOutput = edges.some((e) => e.source === node.id)
      
      if (!hasInput && node.data.blockType !== 'input') {
        errors.push({
          nodeId: node.id,
          message: `Block "${node.data.label}" has no input connection`,
          type: 'warning'
        })
      }
      
      if (!hasOutput) {
        errors.push({
          nodeId: node.id,
          message: `Block "${node.data.label}" has no output connection`,
          type: 'warning'
        })
      }
      
      const nodeDef = getNodeDefinition(node.data.blockType as BlockType, BackendFramework.PyTorch)
      if (nodeDef) {
        const requiredFields = nodeDef.configSchema.filter((f) => f.required)
        requiredFields.forEach((field) => {
          if (!node.data.config[field.name]) {
            errors.push({
              nodeId: node.id,
              message: `Block "${node.data.label}" missing required parameter: ${field.label}`,
              type: 'error'
            })
          }
        })
      }
      
      // Special validation for loss nodes - check input count matches loss type
      if (node.data.blockType === 'loss') {
        const lossNodeDef = nodeDef as any
        if (lossNodeDef?.getInputPorts) {
          const requiredPorts = lossNodeDef.getInputPorts(node.data.config)
          const incomingEdges = edges.filter((e) => e.target === node.id)
          
          // Check total connection count
          if (incomingEdges.length !== requiredPorts.length) {
            errors.push({
              nodeId: node.id,
              message: `Loss function "${node.data.config.loss_type || 'cross_entropy'}" requires ${requiredPorts.length} inputs (${requiredPorts.map((p: any) => p.label).join(', ')}), but has ${incomingEdges.length}`,
              type: 'error'
            })
          } else {
            // Check that all required ports are filled (handle-aware)
            const connectedHandles = new Set(
              incomingEdges.map(e => e.targetHandle || 'default')
            )
            
            const missingPorts = requiredPorts.filter(
              (p: any) => !connectedHandles.has(p.id)
            )
            
            if (missingPorts.length > 0) {
              errors.push({
                nodeId: node.id,
                message: `Loss node missing connections to: ${missingPorts.map((p: any) => p.label).join(', ')}`,
                type: 'error'
              })
            }
          }
        }
      }
    })
    
    set({ validationErrors: errors })
    return errors
  },

  inferDimensions: () => {
    const { nodes, edges } = get()
    const updatedNodes = [...nodes]
    
    const nodeMap = new Map(updatedNodes.map((n) => [n.id, n]))
    
    const getIncomingEdges = (nodeId: string) => edges.filter((e) => e.target === nodeId)
    const visited = new Set<string>()
    
    const processNode = (nodeId: string): void => {
      if (visited.has(nodeId)) return
      visited.add(nodeId)
      
      const node = nodeMap.get(nodeId)
      if (!node) return
      
      const incomingEdges = getIncomingEdges(nodeId)
      
      // Try new registry first, fall back to legacy adapter
      let nodeDef = getNodeDefinition(node.data.blockType, BackendFramework.PyTorch)
      
      if (node.data.blockType === 'input') {
        if (nodeDef) {
          // Use new registry method
          const outputShape = nodeDef.computeOutputShape(undefined, node.data.config)
          node.data.outputShape = outputShape

          // Set shape status for input nodes
          node.data.shapeStatus = {
            state: outputShape ? NodeValidationState.VALID : NodeValidationState.UNCONFIGURED,
            inputShapes: [],
            outputShape: outputShape as any,
            timestamp: Date.now()
          }
        }
      } else {
        if (incomingEdges.length > 0) {
          // Special handling for merge nodes (concat, add) with multiple inputs
          if ((node.data.blockType === 'concat' || node.data.blockType === 'add') && incomingEdges.length > 1) {
            // Gather all input shapes
            const inputShapes: TensorShape[] = []
            for (const edge of incomingEdges) {
              const sourceNode = nodeMap.get(edge.source)
              if (sourceNode?.data.outputShape) {
                inputShapes.push(sourceNode.data.outputShape)
              }
            }
            
            // Only compute if all inputs have shapes
            if (inputShapes.length === incomingEdges.length && nodeDef) {
              // Set first input as inputShape for consistency
              node.data.inputShape = inputShapes[0]
              
              // Use computeMultiInputShape if available (for concat/add nodes)
              const nodeDefAny = nodeDef as any
              let outputShape: TensorShape | undefined
              if (typeof nodeDefAny.computeMultiInputShape === 'function') {
                outputShape = nodeDefAny.computeMultiInputShape(inputShapes, node.data.config)
                node.data.outputShape = outputShape
              } else {
                // Fallback to regular computation
                outputShape = nodeDef.computeOutputShape(node.data.inputShape, node.data.config)
                node.data.outputShape = outputShape
              }

              // Set shape status for merge nodes
              const state = outputShape
                ? (hasSymbolicDims(outputShape) ? NodeValidationState.NEGOTIATING : NodeValidationState.VALID)
                : NodeValidationState.ERROR
              node.data.shapeStatus = {
                state,
                inputShapes: inputShapes as any,
                outputShape: outputShape as any,
                timestamp: Date.now()
              }
            }
          } else {
            // Regular nodes or merge nodes with single input
            const sourceNode = nodeMap.get(incomingEdges[0].source)

            if (sourceNode?.data.outputShape) {
              node.data.inputShape = sourceNode.data.outputShape

              if (nodeDef) {
                // Use new registry method
                const outputShape = nodeDef.computeOutputShape(node.data.inputShape, node.data.config)
                node.data.outputShape = outputShape

                // Set shape status for regular nodes
                let state: NodeValidationState
                if (!outputShape) {
                  state = NodeValidationState.ERROR
                } else if (hasSymbolicDims(outputShape)) {
                  state = NodeValidationState.NEGOTIATING
                } else {
                  state = NodeValidationState.VALID
                }
                node.data.shapeStatus = {
                  state,
                  inputShapes: [sourceNode.data.outputShape as any],
                  outputShape: outputShape as any,
                  timestamp: Date.now()
                }
              }
            } else {
              // No input shape yet
              node.data.shapeStatus = {
                state: NodeValidationState.AWAITING_INPUT,
                inputShapes: [],
                timestamp: Date.now()
              }
            }
          }
        }
      }
      
      const outgoingEdges = edges.filter((e) => e.source === nodeId)
      outgoingEdges.forEach((e) => processNode(e.target))
    }
    
    const inputNodes = updatedNodes.filter((n) => n.data.blockType === 'input')
    inputNodes.forEach((node) => processNode(node.id))
    
    // Process standalone nodes that haven't been visited yet
    // These are nodes not connected to any input, but may still compute shapes
    const standaloneNodes = updatedNodes.filter((n) => 
      !visited.has(n.id) && 
      n.data.blockType !== 'input' && 
      n.data.blockType !== 'dataloader' &&
      n.data.blockType !== 'loss'
    )

    standaloneNodes.forEach((node) => {
      const nodeDef = getNodeDefinition(node.data.blockType, BackendFramework.PyTorch)
      if (nodeDef) {
        // Attempt to compute output shape even without input connection
        // This works for nodes that can infer shape from config alone
        const outputShape = nodeDef.computeOutputShape(node.data.inputShape, node.data.config)
        if (outputShape) {
          node.data.outputShape = outputShape
        }
      }
    })
    
    set({ nodes: updatedNodes })
  },

  createProject: (name, description, framework) => {
    const project: Project = {
      id: Date.now().toString(),
      name,
      description,
      framework,
      nodes: [],
      edges: [],
      createdAt: Date.now(),
      updatedAt: Date.now()
    }

    set({
      currentProject: project,
      nodes: [],
      edges: [],
      selectedNodeId: null,
      validationErrors: []
    })
  },

  saveProject: () => {
    const { currentProject, nodes, edges } = get()
    if (!currentProject) return
    
    const updatedProject = {
      ...currentProject,
      nodes,
      edges,
      updatedAt: Date.now()
    }
    
    set({ currentProject: updatedProject })
  },

  loadProject: (project) => {
    set({
      currentProject: project,
      nodes: project.nodes,
      edges: project.edges,
      selectedNodeId: null,
      validationErrors: []
    })
  },

  updateProjectInfo: (name, description) => {
    set((state) => ({
      currentProject: state.currentProject
        ? { ...state.currentProject, name, description, updatedAt: Date.now() }
        : null
    }))
  },

  undo: () => {
    const { past, nodes, edges } = get()
    if (past.length === 0) return
    
    const previous = past[past.length - 1]
    const newPast = past.slice(0, past.length - 1)
    
    set((state) => ({
      past: newPast,
      future: [...state.future, { nodes, edges }].slice(-MAX_HISTORY),
      nodes: previous.nodes,
      edges: previous.edges
    }))
    
    get().inferDimensions()
  },

  redo: () => {
    const { future, nodes, edges } = get()
    if (future.length === 0) return
    
    const next = future[future.length - 1]
    const newFuture = future.slice(0, future.length - 1)
    
    set((state) => ({
      future: newFuture,
      past: [...state.past, { nodes, edges }].slice(-MAX_HISTORY),
      nodes: next.nodes,
      edges: next.edges
    }))
    
    get().inferDimensions()
  },

  canUndo: () => get().past.length > 0,
  
  canRedo: () => get().future.length > 0,

  reset: () => {
    set({
      nodes: [],
      edges: [],
      selectedNodeId: null,
      validationErrors: [],
      currentProject: null,
      past: [],
      future: []
    })
  }
}))
