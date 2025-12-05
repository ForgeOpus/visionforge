import { describe, it, expect, beforeEach } from 'vitest'
import { useModelBuilderStore } from './store'
import { Node, Edge } from '@xyflow/react'
import { BlockData, GroupBlockDefinition, GroupBlockData, PortMapping } from './types'

describe('Store - Block Expansion', () => {
  beforeEach(() => {
    // Reset store before each test
    useModelBuilderStore.getState().reset()
  })

  it('should expand a group block and restore internal nodes', () => {
    // Create internal nodes for the group
    const internalNode1: Node<BlockData> = {
      id: 'conv-1',
      type: 'custom',
      position: { x: 100, y: 100 },
      data: {
        blockType: 'conv2d',
        label: 'Conv2D',
        config: { out_channels: 64, kernel_size: 3 },
        category: 'basic'
      }
    }

    const internalNode2: Node<BlockData> = {
      id: 'relu-1',
      type: 'custom',
      position: { x: 100, y: 200 },
      data: {
        blockType: 'relu',
        label: 'ReLU',
        config: {},
        category: 'activation'
      }
    }

    const internalEdge: Edge = {
      id: 'e1',
      source: 'conv-1',
      target: 'relu-1'
    }

    // Create port mappings
    const portMappings: PortMapping[] = [
      {
        internalNodeId: 'conv-1',
        internalPortId: 'default',
        externalPortId: 'input-1',
        externalPortLabel: 'Input',
        type: 'input',
        semantic: 'data'
      },
      {
        internalNodeId: 'relu-1',
        internalPortId: 'default',
        externalPortId: 'output-1',
        externalPortLabel: 'Output',
        type: 'output',
        semantic: 'data'
      }
    ]

    // Create group definition
    const groupDef: GroupBlockDefinition = {
      id: 'group-1',
      name: 'ConvBlock',
      description: 'Conv + ReLU',
      category: 'basic',
      color: '#9333ea',
      internalNodes: [internalNode1, internalNode2],
      internalEdges: [internalEdge],
      portMappings,
      createdAt: Date.now(),
      updatedAt: Date.now()
    }

    // Add group definition to store
    useModelBuilderStore.getState().loadGroupDefinitions([groupDef])

    // Create group block node
    const groupNode: Node<GroupBlockData> = {
      id: 'group-block-1',
      type: 'group',
      position: { x: 300, y: 300 },
      data: {
        blockType: 'group',
        label: 'ConvBlock',
        config: {},
        category: 'basic',
        groupDefinitionId: 'group-1',
        isExpanded: false
      }
    }

    // Add group node to canvas
    useModelBuilderStore.getState().setNodes([groupNode as any])

    // Verify initial state
    let state = useModelBuilderStore.getState()
    expect(state.nodes).toHaveLength(1)
    expect(state.nodes[0].id).toBe('group-block-1')

    // Expand the group
    useModelBuilderStore.getState().toggleGroupExpansion('group-block-1')

    // Verify expansion
    state = useModelBuilderStore.getState()
    expect(state.nodes).toHaveLength(2) // Should have 2 internal nodes
    expect(state.nodes.some(n => n.data.blockType === 'conv2d')).toBe(true)
    expect(state.nodes.some(n => n.data.blockType === 'relu')).toBe(true)
    expect(state.edges).toHaveLength(1) // Should have 1 internal edge

    // Verify group definition is still in library
    expect(state.groupDefinitions.has('group-1')).toBe(true)
  })

  it('should rewire external connections when expanding', () => {
    // Create internal nodes
    const internalNode1: Node<BlockData> = {
      id: 'linear-1',
      type: 'custom',
      position: { x: 0, y: 0 },
      data: {
        blockType: 'linear',
        label: 'Linear',
        config: { out_features: 128 },
        category: 'basic'
      }
    }

    const internalNode2: Node<BlockData> = {
      id: 'dropout-1',
      type: 'custom',
      position: { x: 0, y: 100 },
      data: {
        blockType: 'dropout',
        label: 'Dropout',
        config: { p: 0.5 },
        category: 'basic'
      }
    }

    const internalEdge: Edge = {
      id: 'e-internal',
      source: 'linear-1',
      target: 'dropout-1'
    }

    const portMappings: PortMapping[] = [
      {
        internalNodeId: 'linear-1',
        internalPortId: 'default',
        externalPortId: 'ext-input',
        externalPortLabel: 'Input',
        type: 'input',
        semantic: 'data'
      },
      {
        internalNodeId: 'dropout-1',
        internalPortId: 'default',
        externalPortId: 'ext-output',
        externalPortLabel: 'Output',
        type: 'output',
        semantic: 'data'
      }
    ]

    const groupDef: GroupBlockDefinition = {
      id: 'group-2',
      name: 'LinearBlock',
      description: 'Linear + Dropout',
      category: 'basic',
      color: '#3b82f6',
      internalNodes: [internalNode1, internalNode2],
      internalEdges: [internalEdge],
      portMappings,
      createdAt: Date.now(),
      updatedAt: Date.now()
    }

    useModelBuilderStore.getState().loadGroupDefinitions([groupDef])

    // Create external nodes
    const inputNode: Node<BlockData> = {
      id: 'input-1',
      type: 'custom',
      position: { x: 0, y: 0 },
      data: {
        blockType: 'input',
        label: 'Input',
        config: { shape: [32, 784] },
        category: 'input'
      }
    }

    const outputNode: Node<BlockData> = {
      id: 'output-1',
      type: 'custom',
      position: { x: 600, y: 0 },
      data: {
        blockType: 'linear',
        label: 'Output',
        config: { out_features: 10 },
        category: 'basic'
      }
    }

    const groupNode: Node<GroupBlockData> = {
      id: 'group-block-2',
      type: 'group',
      position: { x: 300, y: 0 },
      data: {
        blockType: 'group',
        label: 'LinearBlock',
        config: {},
        category: 'basic',
        groupDefinitionId: 'group-2',
        isExpanded: false
      }
    }

    // Create external edges
    const edgeToGroup: Edge = {
      id: 'e-to-group',
      source: 'input-1',
      target: 'group-block-2',
      targetHandle: 'ext-input'
    }

    const edgeFromGroup: Edge = {
      id: 'e-from-group',
      source: 'group-block-2',
      sourceHandle: 'ext-output',
      target: 'output-1'
    }

    useModelBuilderStore.getState().setNodes([inputNode, groupNode as any, outputNode])
    useModelBuilderStore.getState().setEdges([edgeToGroup, edgeFromGroup])

    // Verify initial state
    let state = useModelBuilderStore.getState()
    expect(state.nodes).toHaveLength(3)
    expect(state.edges).toHaveLength(2)

    // Expand the group
    useModelBuilderStore.getState().toggleGroupExpansion('group-block-2')

    // Verify nodes after expansion
    state = useModelBuilderStore.getState()
    expect(state.nodes).toHaveLength(4) // input + 2 internal + output
    expect(state.nodes.some(n => n.data.blockType === 'linear' && n.data.label === 'Linear')).toBe(true)
    expect(state.nodes.some(n => n.data.blockType === 'dropout')).toBe(true)

    // Verify edges after expansion
    expect(state.edges).toHaveLength(3) // external-to-internal + internal + internal-to-external

    // Check that external edges are rewired to internal nodes
    const edgeToInternal = state.edges.find(e => e.source === 'input-1')
    expect(edgeToInternal).toBeDefined()
    expect(edgeToInternal?.target).toContain('linear') // Should connect to internal linear node

    const edgeFromInternal = state.edges.find(e => e.target === 'output-1')
    expect(edgeFromInternal).toBeDefined()
    expect(edgeFromInternal?.source).toContain('dropout') // Should connect from internal dropout node
  })

  it('should preserve group definition after expansion', () => {
    const internalNode: Node<BlockData> = {
      id: 'flatten-1',
      type: 'custom',
      position: { x: 0, y: 0 },
      data: {
        blockType: 'flatten',
        label: 'Flatten',
        config: {},
        category: 'basic'
      }
    }

    const portMappings: PortMapping[] = [
      {
        internalNodeId: 'flatten-1',
        internalPortId: 'default',
        externalPortId: 'in',
        externalPortLabel: 'In',
        type: 'input',
        semantic: 'data'
      },
      {
        internalNodeId: 'flatten-1',
        internalPortId: 'default',
        externalPortId: 'out',
        externalPortLabel: 'Out',
        type: 'output',
        semantic: 'data'
      }
    ]

    const groupDef: GroupBlockDefinition = {
      id: 'group-3',
      name: 'FlattenBlock',
      description: 'Just flatten',
      category: 'basic',
      color: '#10b981',
      internalNodes: [internalNode],
      internalEdges: [],
      portMappings,
      createdAt: Date.now(),
      updatedAt: Date.now()
    }

    useModelBuilderStore.getState().loadGroupDefinitions([groupDef])

    const groupNode: Node<GroupBlockData> = {
      id: 'group-block-3',
      type: 'group',
      position: { x: 200, y: 200 },
      data: {
        blockType: 'group',
        label: 'FlattenBlock',
        config: {},
        category: 'basic',
        groupDefinitionId: 'group-3',
        isExpanded: false
      }
    }

    useModelBuilderStore.getState().setNodes([groupNode as any])

    // Verify definition exists before expansion
    let state = useModelBuilderStore.getState()
    expect(state.groupDefinitions.has('group-3')).toBe(true)
    const defBefore = state.groupDefinitions.get('group-3')
    expect(defBefore?.name).toBe('FlattenBlock')

    // Expand
    useModelBuilderStore.getState().toggleGroupExpansion('group-block-3')

    // Verify definition still exists after expansion
    state = useModelBuilderStore.getState()
    expect(state.groupDefinitions.has('group-3')).toBe(true)
    const defAfter = state.groupDefinitions.get('group-3')
    expect(defAfter?.name).toBe('FlattenBlock')
    expect(defAfter?.internalNodes).toHaveLength(1)
  })
})
