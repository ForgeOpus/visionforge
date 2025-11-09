import { create } from 'zustand'
import { Node, Edge, Connection } from '@xyflow/react'
import { BlockData, Project, ValidationError, TensorShape } from './types'
import { getBlockDefinition } from './blockDefinitions'

interface ModelBuilderState {
  nodes: Node<BlockData>[]
  edges: Edge[]
  selectedNodeId: string | null
  validationErrors: ValidationError[]
  currentProject: Project | null
  
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
  
  createProject: (name: string, description: string, framework: 'pytorch' | 'tensorflow') => void
  saveProject: () => void
  loadProject: (project: Project) => void
  updateProjectInfo: (name: string, description: string) => void
  
  reset: () => void
}

export const useModelBuilderStore = create<ModelBuilderState>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNodeId: null,
  validationErrors: [],
  currentProject: null,

  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),

  addNode: (node) => {
    set((state) => ({
      nodes: [...state.nodes, node]
    }))
  },

  updateNode: (id, data) => {
    set((state) => ({
      nodes: state.nodes.map((node) =>
        node.id === id ? { ...node, data: { ...node.data, ...data } } : node
      )
    }))
    
    get().inferDimensions()
  },

  removeNode: (id) => {
    set((state) => ({
      nodes: state.nodes.filter((node) => node.id !== id),
      edges: state.edges.filter((edge) => edge.source !== id && edge.target !== id),
      selectedNodeId: state.selectedNodeId === id ? null : state.selectedNodeId
    }))
  },

  addEdge: (edge) => {
    set((state) => ({
      edges: [...state.edges, edge]
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
    set((state) => ({
      edges: state.edges.filter((edge) => edge.id !== id)
    }))
  },

  setSelectedNodeId: (id) => set({ selectedNodeId: id }),

  validateConnection: (connection) => {
    const { nodes, edges } = get()
    
    const targetNode = nodes.find((n) => n.id === connection.target)
    if (!targetNode) return false
    
    const sourceNode = nodes.find((n) => n.id === connection.source)
    if (!sourceNode) return false
    
    if (targetNode.data.blockType !== 'concat' && targetNode.data.blockType !== 'add') {
      const hasExistingInput = edges.some((e) => e.target === connection.target)
      if (hasExistingInput) return false
    }
    
    if (!sourceNode.data.outputShape) return true
    
    const targetDef = getBlockDefinition(targetNode.data.blockType)
    if (!targetDef) return false
    
    const sourceShape = sourceNode.data.outputShape
    
    if (targetNode.data.blockType === 'add') {
      const incomingEdges = edges.filter((e) => e.target === connection.target)
      if (incomingEdges.length > 0) {
        const firstSourceNode = nodes.find((n) => n.id === incomingEdges[0].source)
        if (firstSourceNode?.data.outputShape) {
          const firstShape = firstSourceNode.data.outputShape
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

  reset: () => {
    set({
      nodes: [],
      edges: [],
      selectedNodeId: null,
      validationErrors: [],
      currentProject: null
    })
  }
}))
