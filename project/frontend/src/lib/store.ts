import { create } from 'zustand'
import { Node, Edge, Connection } from '@xyflow/react'
import { BlockData, Project, ValidationError, TensorShape, BlockType, GroupBlockDefinition, PortMapping, GroupBlockData } from './types'
import { getNodeDefinition, BackendFramework } from './nodes/registry'
import { arePortsCompatible } from './nodes/ports'
import { computeGroupBlockShapes, validateGroupBlockShapes } from './groupBlockShapeInference'

interface HistoryState {
  nodes: Node<BlockData>[]
  edges: Edge[]
  groupDefinitions: Map<string, GroupBlockDefinition>
}

interface ModelBuilderState {
  nodes: Node<BlockData>[]
  edges: Edge[]
  selectedNodeId: string | null
  selectedEdgeId: string | null
  recentlyUsedNodes: BlockType[]
  validationErrors: ValidationError[]
  currentProject: Project | null
  groupDefinitions: Map<string, GroupBlockDefinition>

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

  // Group block actions
  createGroupBlock: (selectedNodeIds: string[], config: {
    name: string
    description: string
    category: string
    color: string
    portMappings: PortMapping[]
  }) => string
  toggleGroupExpansion: (nodeId: string) => void
  repeatGroupBlock: (nodeId: string, count: number, spacingX: number, spacingY?: number) => string[]
  ungroupBlock: (nodeId: string) => void
  loadGroupDefinitions: (definitions: GroupBlockDefinition[]) => void
  renameGroupDefinition: (definitionId: string, newName: string) => void
  deleteGroupDefinition: (definitionId: string, cascade: boolean) => void
  duplicateGroupDefinition: (definitionId: string) => string

  reset: () => void
}

const MAX_HISTORY = 10

// Helper to save current state to history
const saveHistory = (state: ModelBuilderState) => {
  const currentState: HistoryState = {
    nodes: JSON.parse(JSON.stringify(state.nodes)),
    edges: JSON.parse(JSON.stringify(state.edges)),
    groupDefinitions: new Map(state.groupDefinitions)
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
  groupDefinitions: new Map(),
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
    const { nodes, edges, groupDefinitions } = get()

    const targetNode = nodes.find((n) => n.id === connection.target)
    if (!targetNode) return false

    const sourceNode = nodes.find((n) => n.id === connection.source)
    if (!sourceNode) return false

    // === NEW: Handle group block connections ===
    const sourceIsGroup = sourceNode.data.blockType === 'group'
    const targetIsGroup = targetNode.data.blockType === 'group'

    if (sourceIsGroup || targetIsGroup) {
      // Validate source handle exists on group
      if (sourceIsGroup) {
        const sourceGroupData = sourceNode.data as GroupBlockData
        const sourceGroupDef = groupDefinitions.get(sourceGroupData.groupDefinitionId)
        if (!sourceGroupDef) {
          console.error('Source group definition not found')
          return false
        }

        const sourceHandleId = connection.sourceHandle || 'default'
        const sourcePort = sourceGroupDef.portMappings.find(m =>
          m.type === 'output' && m.externalPortId === sourceHandleId
        )
        if (!sourcePort) {
          console.error(`Source port ${sourceHandleId} not found on group block`)
          return false
        }
      }

      // Validate target handle exists on group
      if (targetIsGroup) {
        const targetGroupData = targetNode.data as GroupBlockData
        const targetGroupDef = groupDefinitions.get(targetGroupData.groupDefinitionId)
        if (!targetGroupDef) {
          console.error('Target group definition not found')
          return false
        }

        const targetHandleId = connection.targetHandle || 'default'
        const targetPort = targetGroupDef.portMappings.find(m =>
          m.type === 'input' && m.externalPortId === targetHandleId
        )
        if (!targetPort) {
          console.error(`Target port ${targetHandleId} not found on group block`)
          return false
        }

        // Check if target port already occupied
        const handleOccupied = edges.some(e =>
          e.target === connection.target &&
          (e.targetHandle || 'default') === targetHandleId
        )

        if (handleOccupied) {
          console.warn(`Target port ${targetHandleId} already connected`)
          return false
        }
      }

      // Both sides validated, connection is valid
      return true
    }

    // === Continue with existing regular node validation ===
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
    const { nodes, edges, groupDefinitions } = get()
    const errors: ValidationError[] = []
    
    // Check for input nodes
    const inputNodes = nodes.filter((n) => n.data.blockType === 'input')
    if (inputNodes.length === 0) {
      errors.push({
        message: 'Architecture must have at least one Input block to define the data flow',
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
      
      // Validate group blocks
      if (node.data.blockType === 'group') {
        const groupData = node.data as GroupBlockData
        const groupDef = groupDefinitions.get(groupData.groupDefinitionId)
        
        if (!groupDef) {
          errors.push({
            nodeId: node.id,
            message: `Definition not found: Block "${node.data.label}" references a deleted or missing group definition. You can delete this instance or recreate the definition.`,
            type: 'error'
          })
        } else {
          // Validate internal structure
          const shapeErrors = validateGroupBlockShapes(groupDef)
          shapeErrors.forEach(errorMsg => {
            errors.push({
              nodeId: node.id,
              message: `Internal structure error in "${node.data.label}": ${errorMsg}`,
              type: 'error'
            })
          })
        }
      } else {
        const nodeDef = getNodeDefinition(node.data.blockType as BlockType, BackendFramework.PyTorch)
        if (nodeDef) {
          const requiredFields = nodeDef.configSchema.filter((f) => f.required)
          requiredFields.forEach((field) => {
            if (!node.data.config[field.name]) {
              errors.push({
                nodeId: node.id,
                message: `Configuration error: Block "${node.data.label}" is missing required parameter "${field.label}". Please configure this block.`,
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
                message: `Input mismatch: Loss function "${node.data.config.loss_type || 'cross_entropy'}" requires exactly ${requiredPorts.length} input(s) (${requiredPorts.map((p: any) => p.label).join(', ')}), but currently has ${incomingEdges.length}. Please connect the required inputs.`,
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
                  message: `Missing connections: Loss node requires connections to the following ports: ${missingPorts.map((p: any) => p.label).join(', ')}. Please connect these inputs.`,
                  type: 'error'
                })
              }
            }
          }
        }
      }
    })
    
    set({ validationErrors: errors })
    return errors
  },

  inferDimensions: () => {
    const { nodes, edges, groupDefinitions } = get()
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
      
      // Process dependencies first
      incomingEdges.forEach((edge) => {
        if (!visited.has(edge.source)) {
          processNode(edge.source)
        }
      })
      
      // Handle group blocks specially
      if (node.data.blockType === 'group') {
        const groupData = node.data as GroupBlockData
        const groupDef = groupDefinitions.get(groupData.groupDefinitionId)
        
        if (groupDef) {
          // Gather external input shapes from incoming edges
          const externalInputShapes = new Map<string, TensorShape>()
          
          incomingEdges.forEach(edge => {
            const sourceNode = nodeMap.get(edge.source)
            const targetHandle = edge.targetHandle || 'default'
            
            if (sourceNode?.data.outputShape) {
              externalInputShapes.set(targetHandle, sourceNode.data.outputShape)
            }
          })
          
          // Compute shapes through the internal graph
          const result = computeGroupBlockShapes(groupDef, externalInputShapes)
          
          // Store input shapes on the group block node
          if (result.inputShapes.size > 0) {
            const firstInputShape = Array.from(result.inputShapes.values())[0]
            node.data.inputShape = firstInputShape
          }
          
          // Store output shapes on the group block node
          if (result.outputShapes.size > 0) {
            const firstOutputShape = Array.from(result.outputShapes.values())[0]
            node.data.outputShape = firstOutputShape
          }
          
          // Report any shape inference errors
          if (result.errors.length > 0) {
            console.warn(`Shape inference errors for group block ${groupDef.name}:`, result.errors)
          }
        }
      } else {
        // Regular node processing
        let nodeDef = getNodeDefinition(node.data.blockType, BackendFramework.PyTorch)
        
        if (node.data.blockType === 'input') {
          if (nodeDef) {
            // Use new registry method
            const outputShape = nodeDef.computeOutputShape(undefined, node.data.config)
            node.data.outputShape = outputShape
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
                if (typeof nodeDefAny.computeMultiInputShape === 'function') {
                  const outputShape = nodeDefAny.computeMultiInputShape(inputShapes, node.data.config)
                  node.data.outputShape = outputShape
                } else {
                  // Fallback to regular computation
                  const outputShape = nodeDef.computeOutputShape(node.data.inputShape, node.data.config)
                  node.data.outputShape = outputShape
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
                }
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
    const { past, nodes, edges, groupDefinitions } = get()
    if (past.length === 0) return

    const previous = past[past.length - 1]
    const newPast = past.slice(0, past.length - 1)

    set((state) => ({
      past: newPast,
      future: [...state.future, { nodes, edges, groupDefinitions }].slice(-MAX_HISTORY),
      nodes: previous.nodes,
      edges: previous.edges,
      groupDefinitions: previous.groupDefinitions
    }))

    get().inferDimensions()
  },

  redo: () => {
    const { future, nodes, edges, groupDefinitions } = get()
    if (future.length === 0) return

    const next = future[future.length - 1]
    const newFuture = future.slice(0, future.length - 1)

    set((state) => ({
      future: newFuture,
      past: [...state.past, { nodes, edges, groupDefinitions }].slice(-MAX_HISTORY),
      nodes: next.nodes,
      edges: next.edges,
      groupDefinitions: next.groupDefinitions
    }))

    get().inferDimensions()
  },

  canUndo: () => get().past.length > 0,

  canRedo: () => get().future.length > 0,

  // Group block actions
  createGroupBlock: (selectedNodeIds, config) => {
    const state = get()
    const historyUpdate = saveHistory(state)

    // Get selected nodes and their edges
    const selectedNodes = state.nodes.filter(n => selectedNodeIds.includes(n.id))
    if (selectedNodes.length < 2) return ''

    // Find all edges involving selected nodes
    const internalEdges: Edge[] = []
    const externalEdges: Edge[] = []

    state.edges.forEach(edge => {
      const sourceInSelection = selectedNodeIds.includes(edge.source)
      const targetInSelection = selectedNodeIds.includes(edge.target)

      if (sourceInSelection && targetInSelection) {
        internalEdges.push(edge)
      } else if (sourceInSelection || targetInSelection) {
        externalEdges.push(edge)
      }
    })

    // Use port mappings from config (provided by GroupCreationDialog)
    const portMappings = config.portMappings

    // Validate port mappings
    const validationErrors: string[] = []
    const selectedNodeIdSet = new Set(selectedNodes.map(n => n.id))
    const portIds = new Set<string>()

    portMappings.forEach((mapping, index) => {
      // Check internal node exists in selection
      if (!selectedNodeIdSet.has(mapping.internalNodeId)) {
        validationErrors.push(`Port mapping ${index}: Internal node ${mapping.internalNodeId} not in selection`)
      }

      // Check external port ID is unique
      if (portIds.has(mapping.externalPortId)) {
        validationErrors.push(`Port mapping ${index}: Duplicate external port ID ${mapping.externalPortId}`)
      }
      portIds.add(mapping.externalPortId)

      // Validate internal port exists on the node
      const internalNode = selectedNodes.find(n => n.id === mapping.internalNodeId)
      if (internalNode) {
        const nodeDef = getNodeDefinition(internalNode.data.blockType, BackendFramework.PyTorch)
        if (nodeDef) {
          const ports = mapping.type === 'input'
            ? nodeDef.getInputPorts(internalNode.data.config)
            : nodeDef.getOutputPorts(internalNode.data.config)

          const portExists = ports.some(p => p.id === mapping.internalPortId)
          if (!portExists) {
            validationErrors.push(`Port mapping ${index}: Port ${mapping.internalPortId} not found on node ${internalNode.data.label}`)
          }
        }
      }
    })

    if (validationErrors.length > 0) {
      console.error('Port mapping validation failed:', validationErrors)
      // Return early - don't create invalid group
      return ''
    }

    // Create group definition
    const groupId = `group-${Date.now()}`
    const groupDefinition: GroupBlockDefinition = {
      id: groupId,
      name: config.name,
      description: config.description,
      category: config.category as any,
      color: config.color,
      internalNodes: selectedNodes,
      internalEdges,
      portMappings,
      createdAt: Date.now(),
      updatedAt: Date.now()
    }

    // Calculate centroid position
    const centroidX = selectedNodes.reduce((sum, n) => sum + n.position.x, 0) / selectedNodes.length
    const centroidY = selectedNodes.reduce((sum, n) => sum + n.position.y, 0) / selectedNodes.length

    // Create group block node
    const groupNodeId = `group-block-${Date.now()}`
    const groupNode: Node<GroupBlockData> = {
      id: groupNodeId,
      type: 'group',
      position: { x: centroidX, y: centroidY },
      data: {
        blockType: 'group',
        label: config.name,
        config: {},
        category: config.category as any,
        groupDefinitionId: groupId,
        isExpanded: false
      }
    }

    // Rewire external connections to group node
    const newEdges: Edge[] = []
    state.edges.forEach(edge => {
      const sourceInSelection = selectedNodeIds.includes(edge.source)
      const targetInSelection = selectedNodeIds.includes(edge.target)

      // Keep edges that don't involve selected nodes
      if (!sourceInSelection && !targetInSelection) {
        newEdges.push(edge)
      }
      // Rewire incoming edges
      else if (!sourceInSelection && targetInSelection) {
        const mapping = portMappings.find(m =>
          m.type === 'input' &&
          m.internalNodeId === edge.target &&
          m.internalPortId === (edge.targetHandle || 'default')
        )
        if (mapping) {
          newEdges.push({
            ...edge,
            target: groupNodeId,
            targetHandle: mapping.externalPortId
          })
        }
      }
      // Rewire outgoing edges
      else if (sourceInSelection && !targetInSelection) {
        const mapping = portMappings.find(m =>
          m.type === 'output' &&
          m.internalNodeId === edge.source &&
          m.internalPortId === (edge.sourceHandle || 'default')
        )
        if (mapping) {
          newEdges.push({
            ...edge,
            source: groupNodeId,
            sourceHandle: mapping.externalPortId
          })
        }
      }
    })

    // Remove selected nodes and add group node
    const newNodes = state.nodes.filter(n => !selectedNodeIds.includes(n.id))
    newNodes.push(groupNode as any)

    // Update state
    const newGroupDefs = new Map(state.groupDefinitions)
    newGroupDefs.set(groupId, groupDefinition)

    set({
      nodes: newNodes,
      edges: newEdges,
      groupDefinitions: newGroupDefs,
      ...historyUpdate
    })

    get().inferDimensions()
    return groupNodeId
  },

  toggleGroupExpansion: (nodeId) => {
    const state = get()
    const historyUpdate = saveHistory(state)

    // First, check if we're trying to collapse (expanded nodes exist for this group)
    const internalNodeBaseIds = new Set(
      Array.from(state.groupDefinitions.values())
        .flatMap(def => def.internalNodes.map(n => n.id))
    )

    const expandedNodes = state.nodes.filter(n => {
      // Check if this node has expansion metadata matching our nodeId
      const data = n.data as any
      return data._expandedFrom === nodeId && data._isExpandedInternal === true
    })

    if (expandedNodes.length > 0) {
      // COLLAPSE: Expanded nodes exist for this group, collapse them
      const groupData = expandedNodes[0].data as any
      const groupDef = state.groupDefinitions.get(groupData._groupDefinitionId)
      if (!groupDef) return

      // Calculate centroid of expanded nodes
      const centroidX = expandedNodes.reduce((sum, n) => sum + n.position.x, 0) / expandedNodes.length
      const centroidY = expandedNodes.reduce((sum, n) => sum + n.position.y, 0) / expandedNodes.length

      // Recreate the collapsed group block at the centroid
      const collapsedGroupNode: Node<GroupBlockData> = {
        id: nodeId,
        type: 'group',
        position: { x: centroidX, y: centroidY },
        data: {
          blockType: 'group',
          label: groupDef.name,
          config: {},
          category: groupDef.category,
          groupDefinitionId: groupData._groupDefinitionId,
          isExpanded: false
        }
      }

      // Rewire external edges back to the group block
      const expandedNodeIds = new Set(expandedNodes.map(n => n.id))
      const newEdges: Edge[] = []
      const lostEdges: Edge[] = []

      state.edges.forEach(edge => {
        const sourceIsExpanded = expandedNodeIds.has(edge.source)
        const targetIsExpanded = expandedNodeIds.has(edge.target)

        // Internal edges (both nodes are expanded) - remove them
        if (sourceIsExpanded && targetIsExpanded) {
          return
        }

        // Incoming edge to expanded node - rewire to group block
        if (targetIsExpanded && !sourceIsExpanded) {
          // Find which internal node this edge connects to
          const targetExpandedNode = expandedNodes.find(n => n.id === edge.target)
          if (targetExpandedNode) {
            // Find the port mapping for this internal node
            const mapping = groupDef.portMappings.find(m =>
              m.type === 'input' &&
              targetExpandedNode.id.startsWith(m.internalNodeId) &&
              (edge.targetHandle || 'default') === m.internalPortId
            )
            if (mapping) {
              newEdges.push({
                ...edge,
                target: nodeId,
                targetHandle: mapping.externalPortId
              })
            } else {
              lostEdges.push(edge) // Track lost edge - no mapping found
            }
          } else {
            lostEdges.push(edge) // Track lost edge - target node not found
          }
        }
        // Outgoing edge from expanded node - rewire to group block
        else if (sourceIsExpanded && !targetIsExpanded) {
          const sourceExpandedNode = expandedNodes.find(n => n.id === edge.source)
          if (sourceExpandedNode) {
            const mapping = groupDef.portMappings.find(m =>
              m.type === 'output' &&
              sourceExpandedNode.id.startsWith(m.internalNodeId) &&
              (edge.sourceHandle || 'default') === m.internalPortId
            )
            if (mapping) {
              newEdges.push({
                ...edge,
                source: nodeId,
                sourceHandle: mapping.externalPortId
              })
            } else {
              lostEdges.push(edge) // Track lost edge - no mapping found
            }
          } else {
            lostEdges.push(edge) // Track lost edge - source node not found
          }
        }
        // External edge - keep it
        else {
          newEdges.push(edge)
        }
      })

      // Warn about lost edges
      if (lostEdges.length > 0) {
        console.warn(`${lostEdges.length} connection(s) could not be rewired during collapse:`, lostEdges)
      }

      // Remove expanded nodes, container, and add collapsed group block
      const containerNodeId = `${nodeId}-container`
      const newNodes = state.nodes.filter(n =>
        !expandedNodeIds.has(n.id) && n.id !== containerNodeId
      )
      newNodes.push(collapsedGroupNode as any)

      set({
        nodes: newNodes,
        edges: newEdges,
        ...historyUpdate
      })

      get().inferDimensions()
    } else {
      // EXPAND: No expanded nodes found, so find the collapsed group node and expand it
      const groupNode = state.nodes.find(n => n.id === nodeId)
      if (!groupNode || groupNode.data.blockType !== 'group') return

      const groupData = groupNode.data as GroupBlockData
      const groupDef = state.groupDefinitions.get(groupData.groupDefinitionId)
      if (!groupDef) return

      const blockX = groupNode.position.x
      const blockY = groupNode.position.y

      // Restore internal nodes with positions relative to block location
      // Add metadata to track which group they belong to
      const restoredNodes = groupDef.internalNodes.map(internalNode => ({
        ...internalNode,
        id: `${internalNode.id}-expanded-${Date.now()}`,
        data: {
          ...internalNode.data,
          _expandedFrom: nodeId,  // Track parent group node
          _isExpandedInternal: true,  // Mark as internal expanded node
          _groupDefinitionId: groupData.groupDefinitionId  // Store group definition ID
        },
        position: {
          x: blockX + (internalNode.position.x - groupDef.internalNodes[0].position.x),
          y: blockY + (internalNode.position.y - groupDef.internalNodes[0].position.y)
        }
      }))

      // Create ID mapping
      const idMapping = new Map<string, string>()
      groupDef.internalNodes.forEach((oldNode, index) => {
        idMapping.set(oldNode.id, restoredNodes[index].id)
      })

      // Restore internal edges with unique IDs
      let edgeCounter = 0
      const restoredInternalEdges = groupDef.internalEdges.map(edge => ({
        ...edge,
        id: `${edge.id}-expanded-${Date.now()}-${edgeCounter++}`,
        source: idMapping.get(edge.source) || edge.source,
        target: idMapping.get(edge.target) || edge.target
      }))

      // Rewire external edges to internal nodes
      const newEdges: Edge[] = []
      const lostEdges: Edge[] = []
      // Continue counter from internal edges

      state.edges.forEach(edge => {
        if (edge.target === nodeId) {
          const targetHandle = edge.targetHandle || 'default'
          const mapping = groupDef.portMappings.find(m =>
            m.type === 'input' && m.externalPortId === targetHandle
          )
          if (mapping) {
            const newTargetId = idMapping.get(mapping.internalNodeId)
            if (newTargetId) {
              newEdges.push({
                ...edge,
                id: `${edge.id}-rewired-${Date.now()}-${edgeCounter++}`,
                target: newTargetId,
                targetHandle: mapping.internalPortId
              })
            } else {
              lostEdges.push(edge) // Track lost edge - ID mapping failed
            }
          } else {
            lostEdges.push(edge) // Track lost edge - no mapping found
          }
        } else if (edge.source === nodeId) {
          const sourceHandle = edge.sourceHandle || 'default'
          const mapping = groupDef.portMappings.find(m =>
            m.type === 'output' && m.externalPortId === sourceHandle
          )
          if (mapping) {
            const newSourceId = idMapping.get(mapping.internalNodeId)
            if (newSourceId) {
              newEdges.push({
                ...edge,
                id: `${edge.id}-rewired-${Date.now()}-${edgeCounter++}`,
                source: newSourceId,
                sourceHandle: mapping.internalPortId
              })
            } else {
              lostEdges.push(edge) // Track lost edge - ID mapping failed
            }
          } else {
            lostEdges.push(edge) // Track lost edge - no mapping found
          }
        } else {
          newEdges.push(edge)
        }
      })

      newEdges.push(...restoredInternalEdges)

      // Warn about lost edges
      if (lostEdges.length > 0) {
        console.warn(`${lostEdges.length} connection(s) could not be rewired during expansion:`, lostEdges)
      }

      // Calculate bounding box for container
      const minX = Math.min(...restoredNodes.map(n => n.position.x))
      const minY = Math.min(...restoredNodes.map(n => n.position.y))
      const maxX = Math.max(...restoredNodes.map(n => n.position.x + (n.width || 280)))
      const maxY = Math.max(...restoredNodes.map(n => n.position.y + (n.height || 150)))

      const padding = 30
      const containerWidth = maxX - minX + (2 * padding)
      const containerHeight = maxY - minY + (2 * padding)

      // Create container node
      const containerNode = {
        id: `${nodeId}-container`,
        type: 'expandedGroupContainer',
        position: {
          x: minX - padding,
          y: minY - padding
        },
        data: {
          _expandedFrom: nodeId,
          _groupDefinitionId: groupData.groupDefinitionId,
          groupName: groupDef.name,
          groupColor: groupDef.color
        },
        style: {
          width: containerWidth,
          height: containerHeight,
          zIndex: -1  // Place behind the actual nodes
        },
        selectable: false,
        draggable: false
      }

      // Remove group node from canvas and add internal nodes + container
      // Note: We don't keep the expanded group node in the nodes array
      // The expanded internal nodes themselves represent the expanded state
      const newNodes = state.nodes.filter(n => n.id !== nodeId)
      newNodes.push(containerNode as any)
      newNodes.push(...restoredNodes)

      set({
        nodes: newNodes,
        edges: newEdges,
        ...historyUpdate
      })

      get().inferDimensions()
    }
  },

  repeatGroupBlock: (nodeId, count, spacingX, spacingY = 0) => {
    const state = get()
    const historyUpdate = saveHistory(state)

    const sourceNode = state.nodes.find(n => n.id === nodeId)
    if (!sourceNode || sourceNode.data.blockType !== 'group') {
      console.error('Source node is not a group block')
      return []
    }

    const groupData = sourceNode.data as GroupBlockData
    const groupDef = state.groupDefinitions.get(groupData.groupDefinitionId)
    if (!groupDef) {
      console.error('Group definition not found')
      return []
    }

    // Generate a unique sequence ID for this repetition
    const sequenceId = `seq-${Date.now()}`
    const newNodeIds: string[] = []

    // Update the source node with repetition metadata
    const updatedSourceNode = {
      ...sourceNode,
      data: {
        ...sourceNode.data,
        repetitionMetadata: {
          sequenceId,
          index: 0,
          totalCount: count + 1 // Including the source node
        }
      }
    }

    // Smart positioning with collision detection
    const existingNodes = state.nodes.filter(n => n.id !== nodeId) // Exclude source
    const NODE_WIDTH = 280
    const NODE_HEIGHT = 200

    // Find bounding box of existing nodes
    let boundingBox = {
      minX: Number.POSITIVE_INFINITY,
      maxX: Number.NEGATIVE_INFINITY,
      minY: Number.POSITIVE_INFINITY,
      maxY: Number.NEGATIVE_INFINITY
    }

    if (existingNodes.length > 0) {
      boundingBox = {
        minX: Math.min(...existingNodes.map(n => n.position.x)),
        maxX: Math.max(...existingNodes.map(n => n.position.x + NODE_WIDTH)),
        minY: Math.min(...existingNodes.map(n => n.position.y)),
        maxY: Math.max(...existingNodes.map(n => n.position.y + NODE_HEIGHT))
      }
    }

    // Check if requested positions would overlap
    const wouldOverlap = (x: number, y: number) => {
      return existingNodes.some(node => {
        const nodeRight = node.position.x + NODE_WIDTH
        const nodeBottom = node.position.y + NODE_HEIGHT
        const testRight = x + NODE_WIDTH
        const testBottom = y + NODE_HEIGHT

        return !(x > nodeRight || testRight < node.position.x ||
                 y > nodeBottom || testBottom < node.position.y)
      })
    }

    // Adjust spacing if needed to avoid overlaps
    let adjustedSpacingX = spacingX
    let adjustedSpacingY = spacingY

    // Check if first repeated block would overlap
    const firstBlockX = sourceNode.position.x + spacingX
    const firstBlockY = sourceNode.position.y + spacingY

    if (wouldOverlap(firstBlockX, firstBlockY)) {
      // Move to the right of existing architecture with padding
      adjustedSpacingX = Math.max(spacingX, boundingBox.maxX - sourceNode.position.x + 100)
    }

    // Create repeated instances
    const newNodes: Node<GroupBlockData>[] = []
    for (let i = 1; i <= count; i++) {
      const newNodeId = `group-block-${Date.now()}-${i}`
      newNodeIds.push(newNodeId)

      const newNode: Node<GroupBlockData> = {
        id: newNodeId,
        type: 'group',
        position: {
          x: sourceNode.position.x + (adjustedSpacingX * i),
          y: sourceNode.position.y + (adjustedSpacingY * i)
        },
        data: {
          blockType: 'group',
          label: groupDef.name,
          config: {},
          category: groupDef.category,
          groupDefinitionId: groupData.groupDefinitionId,
          isExpanded: false,
          repetitionMetadata: {
            sequenceId,
            index: i,
            totalCount: count + 1
          }
        }
      }

      newNodes.push(newNode)
    }

    // Auto-connect repeated blocks in sequence
    const newEdges: Edge[] = []
    const allNodesInSequence = [updatedSourceNode, ...newNodes]

    for (let i = 0; i < allNodesInSequence.length - 1; i++) {
      const currentBlock = allNodesInSequence[i]
      const nextBlock = allNodesInSequence[i + 1]

      // Get output ports of current block
      const outputPorts = groupDef.portMappings.filter(m => m.type === 'output')

      // Get input ports of next block
      const inputPorts = groupDef.portMappings.filter(m => m.type === 'input')

      // Create connections between compatible ports
      // Strategy: Connect first output to first input (simple sequential chaining)
      if (outputPorts.length > 0 && inputPorts.length > 0) {
        const outputPort = outputPorts[0]
        const inputPort = inputPorts[0]

        // Check semantic compatibility
        if (outputPort.semantic === inputPort.semantic ||
            outputPort.semantic === 'data' ||
            inputPort.semantic === 'data') {

          const edgeId = `e${currentBlock.id}-${nextBlock.id}-${Date.now()}-${i}`
          newEdges.push({
            id: edgeId,
            source: currentBlock.id,
            target: nextBlock.id,
            sourceHandle: outputPort.externalPortId,
            targetHandle: inputPort.externalPortId,
            animated: true,
            style: { stroke: '#6366f1', strokeWidth: 2 }
          })
        }
      }
    }

    // Update state with all new nodes and edges
    const updatedNodes = state.nodes.map(n =>
      n.id === nodeId ? updatedSourceNode : n
    )
    updatedNodes.push(...(newNodes as any[]))

    set({
      nodes: updatedNodes,
      edges: [...state.edges, ...newEdges], // Add new auto-created edges
      ...historyUpdate
    })

    get().inferDimensions()
    return newNodeIds
  },

  ungroupBlock: (nodeId) => {
    const state = get()
    const historyUpdate = saveHistory(state)

    const groupNode = state.nodes.find(n => n.id === nodeId)
    if (!groupNode || groupNode.data.blockType !== 'group') {
      console.error('Node is not a group block')
      return
    }

    const groupData = groupNode.data as GroupBlockData
    const groupDef = state.groupDefinitions.get(groupData.groupDefinitionId)
    if (!groupDef) {
      console.error('Group definition not found')
      return
    }

    // Create ID mapping for internal nodes
    const timestamp = Date.now()
    const idMapping = new Map<string, string>()

    groupDef.internalNodes.forEach(internalNode => {
      const newId = `${internalNode.id}-ungrouped-${timestamp}`
      idMapping.set(internalNode.id, newId)
    })

    // Restore internal nodes at group's position (offset from group position)
    const restoredNodes = groupDef.internalNodes.map(internalNode => {
      const newId = idMapping.get(internalNode.id)!
      return {
        ...internalNode,
        id: newId,
        position: {
          x: groupNode.position.x + internalNode.position.x,
          y: groupNode.position.y + internalNode.position.y
        },
        data: {
          ...internalNode.data
        }
      }
    })

    // Restore internal edges with new IDs
    let edgeCounter = 0
    const restoredInternalEdges = groupDef.internalEdges.map(edge => ({
      ...edge,
      id: `${edge.id}-ungrouped-${timestamp}-${edgeCounter++}`,
      source: idMapping.get(edge.source) || edge.source,
      target: idMapping.get(edge.target) || edge.target
    }))

    // Rewire external connections to internal nodes
    const rewiredEdges: Edge[] = []
    const edgesToRemove = new Set<string>()

    state.edges.forEach(edge => {
      // Input connections: edges targeting the group node
      if (edge.target === nodeId) {
        const targetHandle = edge.targetHandle || 'default'
        const inputMapping = groupDef.portMappings.find(m =>
          m.type === 'input' && m.externalPortId === targetHandle
        )

        if (inputMapping) {
          const newTargetId = idMapping.get(inputMapping.internalNodeId)
          if (newTargetId) {
            edgesToRemove.add(edge.id)
            rewiredEdges.push({
              ...edge,
              id: `${edge.id}-rewired-${timestamp}-${edgeCounter++}`,
              target: newTargetId,
              targetHandle: inputMapping.internalPortId
            })
          }
        }
      }

      // Output connections: edges sourcing from the group node
      if (edge.source === nodeId) {
        const sourceHandle = edge.sourceHandle || 'default'
        const outputMapping = groupDef.portMappings.find(m =>
          m.type === 'output' && m.externalPortId === sourceHandle
        )

        if (outputMapping) {
          const newSourceId = idMapping.get(outputMapping.internalNodeId)
          if (newSourceId) {
            edgesToRemove.add(edge.id)
            rewiredEdges.push({
              ...edge,
              id: `${edge.id}-rewired-${timestamp}-${edgeCounter++}`,
              source: newSourceId,
              sourceHandle: outputMapping.internalPortId
            })
          }
        }
      }
    })

    // Remove group node and old edges, add restored nodes and edges
    const updatedNodes = [
      ...state.nodes.filter(n => n.id !== nodeId),
      ...restoredNodes
    ]

    const updatedEdges = [
      ...state.edges.filter(e => !edgesToRemove.has(e.id)),
      ...restoredInternalEdges,
      ...rewiredEdges
    ]

    set({
      nodes: updatedNodes,
      edges: updatedEdges,
      ...historyUpdate
    })

    // Trigger shape inference
    get().inferDimensions()

    console.log(`Ungrouped block: ${groupData.label}`)
  },

  loadGroupDefinitions: (definitions) => {
    const newMap = new Map<string, GroupBlockDefinition>()
    definitions.forEach(def => newMap.set(def.id, def))
    set({ groupDefinitions: newMap })
  },

  renameGroupDefinition: (definitionId, newName) => {
    const state = get()
    const historyUpdate = saveHistory(state)

    const definition = state.groupDefinitions.get(definitionId)
    if (!definition) {
      console.error('Group definition not found')
      return
    }

    // Update definition name
    const updatedDefinition = {
      ...definition,
      name: newName,
      updatedAt: Date.now()
    }

    const newGroupDefs = new Map(state.groupDefinitions)
    newGroupDefs.set(definitionId, updatedDefinition)

    // Update all instances on canvas
    const updatedNodes = state.nodes.map(node => {
      if (node.data.blockType === 'group') {
        const groupData = node.data as GroupBlockData
        if (groupData.groupDefinitionId === definitionId) {
          return {
            ...node,
            data: {
              ...node.data,
              label: newName
            }
          }
        }
      }
      return node
    })

    set({
      groupDefinitions: newGroupDefs,
      nodes: updatedNodes,
      ...historyUpdate
    })
  },

  deleteGroupDefinition: (definitionId, cascade) => {
    const state = get()
    const historyUpdate = saveHistory(state)

    const definition = state.groupDefinitions.get(definitionId)
    if (!definition) {
      console.error('Group definition not found')
      return
    }

    // Count instances on canvas
    const instanceCount = state.nodes.filter(node => {
      if (node.data.blockType === 'group') {
        const groupData = node.data as GroupBlockData
        return groupData.groupDefinitionId === definitionId
      }
      return false
    }).length

    // Remove definition
    const newGroupDefs = new Map(state.groupDefinitions)
    newGroupDefs.delete(definitionId)

    let updatedNodes = state.nodes

    if (cascade) {
      // Find all nodes to delete: collapsed instances + expanded internal nodes
      const allNodeIdsToDelete = new Set<string>()

      // 1. Find all collapsed group block instances
      const collapsedInstances = state.nodes.filter(node => {
        if (node.data.blockType === 'group') {
          const groupData = node.data as GroupBlockData
          return groupData.groupDefinitionId === definitionId
        }
        return false
      })
      collapsedInstances.forEach(n => allNodeIdsToDelete.add(n.id))

      // 2. Find all expanded internal nodes (those with _groupDefinitionId metadata)
      const expandedInternalNodes = state.nodes.filter(node => {
        const data = node.data as any
        return data._isExpandedInternal === true && data._groupDefinitionId === definitionId
      })
      expandedInternalNodes.forEach(n => allNodeIdsToDelete.add(n.id))

      // 3. Remove all identified nodes
      updatedNodes = state.nodes.filter(node => !allNodeIdsToDelete.has(node.id))

      // 4. Remove edges connected to any deleted nodes
      const updatedEdges = state.edges.filter(
        edge => !allNodeIdsToDelete.has(edge.source) && !allNodeIdsToDelete.has(edge.target)
      )

      set({
        groupDefinitions: newGroupDefs,
        nodes: updatedNodes,
        edges: updatedEdges,
        ...historyUpdate
      })

      // Log success message with accurate count
      const totalDeleted = allNodeIdsToDelete.size
      console.log(`Deleted group definition "${definition.name}" and ${totalDeleted} node(s) (${collapsedInstances.length} collapsed, ${expandedInternalNodes.length} expanded internal)`)
    } else {
      // Mark instances as invalid (definition not found)
      // This is handled by validation in validateArchitecture
      set({
        groupDefinitions: newGroupDefs,
        ...historyUpdate
      })

      // Log warning about orphaned instances
      if (instanceCount > 0) {
        console.warn(`Deleted group definition "${definition.name}" but ${instanceCount} instance(s) remain on canvas and will show errors`)
      }
    }

    get().validateArchitecture()
  },

  duplicateGroupDefinition: (definitionId) => {
    const state = get()
    const historyUpdate = saveHistory(state)

    const definition = state.groupDefinitions.get(definitionId)
    if (!definition) {
      console.error('Group definition not found')
      return ''
    }

    // Create new definition with unique ID and name
    const newId = `group-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    const baseName = definition.name
    let newName = `${baseName} Copy`
    let counter = 2
    
    // Get all existing names from current state
    const existingNames = Array.from(state.groupDefinitions.values()).map(def => def.name)
    
    // Ensure unique name - check if "Name Copy" exists first
    if (existingNames.includes(newName)) {
      // If "Name Copy" exists, try "Name Copy 2", "Name Copy 3", etc.
      while (existingNames.includes(`${baseName} Copy ${counter}`)) {
        counter++
      }
      newName = `${baseName} Copy ${counter}`
    }

    const newDefinition: GroupBlockDefinition = {
      ...definition,
      id: newId,
      name: newName,
      createdAt: Date.now(),
      updatedAt: Date.now()
    }

    const newGroupDefs = new Map(state.groupDefinitions)
    newGroupDefs.set(newId, newDefinition)

    set({
      groupDefinitions: newGroupDefs,
      ...historyUpdate
    })

    return newId
  },

  reset: () => {
    set({
      nodes: [],
      edges: [],
      selectedNodeId: null,
      validationErrors: [],
      currentProject: null,
      groupDefinitions: new Map(),
      past: [],
      future: []
    })
  }
}))
