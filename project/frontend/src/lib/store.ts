import { create } from 'zustand'
import { Node, Edge, Connection } from '@xyflow/react'
import { BlockData, Project, ValidationError, TensorShape } from './types'
import { getBlockDefinition, validateBlockConnection, allowsMultipleInputs } from './blockDefinitions'

interface HistoryState {
  nodes: Node<BlockData>[]
  edges: Edge[]
}

interface ModelBuilderState {
  nodes: Node<BlockData>[]
  edges: Edge[]
  selectedNodeId: string | null
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
  addEdge: (edge: Edge) => void
  removeEdge: (id: string) => void
  setSelectedNodeId: (id: string | null) => void
  
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
  validationErrors: [],
  currentProject: null,
  past: [],
  future: [],

  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),

  addNode: (node) => {
    const state = get()
    const historyUpdate = saveHistory(state)
    
    // Auto-create default project if none exists
    if (!state.currentProject) {
      const defaultProject: Project = {
        id: Date.now().toString(),
        name: 'Untitled Project',
        description: 'Auto-created project',
        framework: 'pytorch',
        nodes: [],
        edges: [],
        createdAt: Date.now(),
        updatedAt: Date.now()
      }
      
      set({
        currentProject: defaultProject,
        nodes: [node],
        ...historyUpdate
      })
    } else {
      set((state) => ({
        nodes: [...state.nodes, node],
        ...historyUpdate
      }))
    }
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
      const targetDef = getBlockDefinition(targetNode.data.blockType)
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

  setSelectedNodeId: (id) => set({ selectedNodeId: id }),

  validateConnection: (connection) => {
    const { nodes, edges } = get()
    
    const targetNode = nodes.find((n) => n.id === connection.target)
    if (!targetNode) return false
    
    const sourceNode = nodes.find((n) => n.id === connection.source)
    if (!sourceNode) return false
    
    // Check if target allows multiple inputs
    if (!allowsMultipleInputs(targetNode.data.blockType)) {
      const hasExistingInput = edges.some((e) => e.target === connection.target)
      if (hasExistingInput) return false
    }
    
    // Use the new validation function
    const validationError = validateBlockConnection(
      sourceNode.data.blockType,
      targetNode.data.blockType,
      sourceNode.data.outputShape
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
      
      const def = getBlockDefinition(node.data.blockType)
      if (def) {
        const requiredFields = def.configSchema.filter((f) => f.required)
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
      
      if (node.data.blockType === 'input') {
        const def = getBlockDefinition(node.data.blockType)
        if (def) {
          const outputShape = def.computeOutputShape(undefined, node.data.config)
          node.data.outputShape = outputShape
        }
      } else {
        if (incomingEdges.length > 0) {
          const sourceNode = nodeMap.get(incomingEdges[0].source)
          
          if (sourceNode?.data.outputShape) {
            node.data.inputShape = sourceNode.data.outputShape
            
            const def = getBlockDefinition(node.data.blockType)
            if (def) {
              const outputShape = def.computeOutputShape(node.data.inputShape, node.data.config)
              node.data.outputShape = outputShape
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
