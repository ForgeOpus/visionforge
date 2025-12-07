import { describe, it, expect, beforeEach } from 'vitest'
import { useModelBuilderStore } from './store'
import { GroupBlockDefinition, PortMapping, TensorShape, BlockData } from './types'
import { Node, Edge } from '@xyflow/react'

describe('Group Block Shape Inference', () => {
  beforeEach(() => {
    useModelBuilderStore.getState().reset()
  })

  it('should compute input shapes from upstream connections', () => {
    // Create an input node
    const inputNode: Node<BlockData> = {
      id: 'input-1',
      type: 'block',
      position: { x: 0, y: 0 },
      data: {
        blockType: 'input',
        label: 'Input',
        config: { shape: '[32, 3, 224, 224]' },
        category: 'input'
      }
    }

    // Create internal nodes for the group
    const internalConv: Node<BlockData> = {
      id: 'conv-internal',
      type: 'block',
      position: { x: 0, y: 0 },
      data: {
        blockType: 'conv2d',
        label: 'Conv2D',
        config: { out_channels: 64, kernel_size: 3, stride: 1, padding: 1, dilation: 1 },
        category: 'basic'
      }
    }

    const internalRelu: Node<BlockData> = {
      id: 'relu-internal',
      type: 'block',
      position: { x: 100, y: 0 },
      data: {
        blockType: 'relu',
        label: 'ReLU',
        config: {},
        category: 'activation'
      }
    }

    // Create internal edges
    const internalEdges: Edge[] = [
      {
        id: 'e1',
        source: 'conv-internal',
        target: 'relu-internal'
      }
    ]

    // Create port mappings
    const portMappings: PortMapping[] = [
      {
        internalNodeId: 'conv-internal',
        internalPortId: 'default',
        externalPortId: 'input-1',
        externalPortLabel: 'Input',
        type: 'input',
        semantic: 'data'
      },
      {
        internalNodeId: 'relu-internal',
        internalPortId: 'default',
        externalPortId: 'output-1',
        externalPortLabel: 'Output',
        type: 'output',
        semantic: 'data'
      }
    ]

    // Create group definition
    const groupDef: GroupBlockDefinition = {
      id: 'test-group-1',
      name: 'Conv Block',
      description: 'Conv + ReLU',
      category: 'custom',
      color: '#9333ea',
      internalNodes: [internalConv, internalRelu],
      internalEdges,
      portMappings,
      createdAt: Date.now(),
      updatedAt: Date.now()
    }

    useModelBuilderStore.getState().loadGroupDefinitions([groupDef])

    // Create group block node
    const groupNode: Node<BlockData> = {
      id: 'group-block-1',
      type: 'group',
      position: { x: 200, y: 0 },
      data: {
        blockType: 'group',
        label: 'Conv Block',
        config: {},
        category: 'custom',
        groupDefinitionId: 'test-group-1',
        isExpanded: false
      }
    }

    // Add nodes
    useModelBuilderStore.getState().addNode(inputNode)
    useModelBuilderStore.getState().addNode(groupNode as any)

    // Connect input to group block
    const edge: Edge = {
      id: 'e-input-group',
      source: 'input-1',
      target: 'group-block-1',
      targetHandle: 'input-1'
    }

    useModelBuilderStore.getState().addEdge(edge)

    // Trigger shape inference
    useModelBuilderStore.getState().inferDimensions()

    // Check that group block has input shape
    const nodes = useModelBuilderStore.getState().nodes
    const updatedGroupNode = nodes.find(n => n.id === 'group-block-1')

    expect(updatedGroupNode?.data.inputShape).toBeDefined()
    expect(updatedGroupNode?.data.inputShape?.dims).toEqual([32, 3, 224, 224])
  })

  it('should propagate shapes through internal graph', () => {
    // Create an input node
    const inputNode: Node<BlockData> = {
      id: 'input-1',
      type: 'block',
      position: { x: 0, y: 0 },
      data: {
        blockType: 'input',
        label: 'Input',
        config: { shape: '[32, 3, 224, 224]' },
        category: 'input'
      }
    }

    // Create internal nodes
    const internalConv: Node<BlockData> = {
      id: 'conv-internal',
      type: 'block',
      position: { x: 0, y: 0 },
      data: {
        blockType: 'conv2d',
        label: 'Conv2D',
        config: { out_channels: 64, kernel_size: 3, stride: 1, padding: 1, dilation: 1 },
        category: 'basic'
      }
    }

    const internalRelu: Node<BlockData> = {
      id: 'relu-internal',
      type: 'block',
      position: { x: 100, y: 0 },
      data: {
        blockType: 'relu',
        label: 'ReLU',
        config: {},
        category: 'activation'
      }
    }

    const internalEdges: Edge[] = [
      {
        id: 'e1',
        source: 'conv-internal',
        target: 'relu-internal'
      }
    ]

    const portMappings: PortMapping[] = [
      {
        internalNodeId: 'conv-internal',
        internalPortId: 'default',
        externalPortId: 'input-1',
        externalPortLabel: 'Input',
        type: 'input',
        semantic: 'data'
      },
      {
        internalNodeId: 'relu-internal',
        internalPortId: 'default',
        externalPortId: 'output-1',
        externalPortLabel: 'Output',
        type: 'output',
        semantic: 'data'
      }
    ]

    const groupDef: GroupBlockDefinition = {
      id: 'test-group-1',
      name: 'Conv Block',
      description: 'Conv + ReLU',
      category: 'custom',
      color: '#9333ea',
      internalNodes: [internalConv, internalRelu],
      internalEdges,
      portMappings,
      createdAt: Date.now(),
      updatedAt: Date.now()
    }

    useModelBuilderStore.getState().loadGroupDefinitions([groupDef])

    const groupNode: Node<BlockData> = {
      id: 'group-block-1',
      type: 'group',
      position: { x: 200, y: 0 },
      data: {
        blockType: 'group',
        label: 'Conv Block',
        config: {},
        category: 'custom',
        groupDefinitionId: 'test-group-1',
        isExpanded: false
      }
    }

    useModelBuilderStore.getState().addNode(inputNode)
    useModelBuilderStore.getState().addNode(groupNode as any)

    const edge: Edge = {
      id: 'e-input-group',
      source: 'input-1',
      target: 'group-block-1',
      targetHandle: 'input-1'
    }

    useModelBuilderStore.getState().addEdge(edge)
    useModelBuilderStore.getState().inferDimensions()

    const nodes = useModelBuilderStore.getState().nodes
    const updatedGroupNode = nodes.find(n => n.id === 'group-block-1')

    // Check that group block has output shape (after internal propagation)
    expect(updatedGroupNode?.data.outputShape).toBeDefined()
    // Conv2d should output [32, 64, 224, 224] (changed channels from 3 to 64)
    expect(updatedGroupNode?.data.outputShape?.dims).toEqual([32, 64, 224, 224])
  })

  it('should propagate output shapes downstream', () => {
    // Create input node
    const inputNode: Node<BlockData> = {
      id: 'input-1',
      type: 'block',
      position: { x: 0, y: 0 },
      data: {
        blockType: 'input',
        label: 'Input',
        config: { shape: '[32, 3, 224, 224]' },
        category: 'input'
      }
    }

    // Create internal nodes
    const internalConv: Node<BlockData> = {
      id: 'conv-internal',
      type: 'block',
      position: { x: 0, y: 0 },
      data: {
        blockType: 'conv2d',
        label: 'Conv2D',
        config: { out_channels: 64, kernel_size: 3, stride: 1, padding: 1, dilation: 1 },
        category: 'basic'
      }
    }

    const portMappings: PortMapping[] = [
      {
        internalNodeId: 'conv-internal',
        internalPortId: 'default',
        externalPortId: 'input-1',
        externalPortLabel: 'Input',
        type: 'input',
        semantic: 'data'
      },
      {
        internalNodeId: 'conv-internal',
        internalPortId: 'default',
        externalPortId: 'output-1',
        externalPortLabel: 'Output',
        type: 'output',
        semantic: 'data'
      }
    ]

    const groupDef: GroupBlockDefinition = {
      id: 'test-group-1',
      name: 'Conv Block',
      description: 'Conv layer',
      category: 'custom',
      color: '#9333ea',
      internalNodes: [internalConv],
      internalEdges: [],
      portMappings,
      createdAt: Date.now(),
      updatedAt: Date.now()
    }

    useModelBuilderStore.getState().loadGroupDefinitions([groupDef])

    const groupNode: Node<BlockData> = {
      id: 'group-block-1',
      type: 'group',
      position: { x: 200, y: 0 },
      data: {
        blockType: 'group',
        label: 'Conv Block',
        config: {},
        category: 'custom',
        groupDefinitionId: 'test-group-1',
        isExpanded: false
      }
    }

    // Create downstream node
    const downstreamNode: Node<BlockData> = {
      id: 'relu-1',
      type: 'block',
      position: { x: 400, y: 0 },
      data: {
        blockType: 'relu',
        label: 'ReLU',
        config: {},
        category: 'activation'
      }
    }

    useModelBuilderStore.getState().addNode(inputNode)
    useModelBuilderStore.getState().addNode(groupNode as any)
    useModelBuilderStore.getState().addNode(downstreamNode)

    // Connect input -> group -> downstream
    useModelBuilderStore.getState().addEdge({
      id: 'e1',
      source: 'input-1',
      target: 'group-block-1',
      targetHandle: 'input-1'
    })

    useModelBuilderStore.getState().addEdge({
      id: 'e2',
      source: 'group-block-1',
      sourceHandle: 'output-1',
      target: 'relu-1'
    })

    useModelBuilderStore.getState().inferDimensions()

    const nodes = useModelBuilderStore.getState().nodes
    const updatedDownstreamNode = nodes.find(n => n.id === 'relu-1')

    // Check that downstream node received the shape from group block
    expect(updatedDownstreamNode?.data.inputShape).toBeDefined()
    expect(updatedDownstreamNode?.data.inputShape?.dims).toEqual([32, 64, 224, 224])
    expect(updatedDownstreamNode?.data.outputShape).toBeDefined()
    expect(updatedDownstreamNode?.data.outputShape?.dims).toEqual([32, 64, 224, 224])
  })

  it('should trigger recomputation on configuration changes', () => {
    // Create input node
    const inputNode: Node<BlockData> = {
      id: 'input-1',
      type: 'block',
      position: { x: 0, y: 0 },
      data: {
        blockType: 'input',
        label: 'Input',
        config: { shape: '[32, 3, 224, 224]' },
        category: 'input'
      }
    }

    // Create internal conv node
    const internalConv: Node<BlockData> = {
      id: 'conv-internal',
      type: 'block',
      position: { x: 0, y: 0 },
      data: {
        blockType: 'conv2d',
        label: 'Conv2D',
        config: { out_channels: 64, kernel_size: 3, stride: 1, padding: 1, dilation: 1 },
        category: 'basic'
      }
    }

    const portMappings: PortMapping[] = [
      {
        internalNodeId: 'conv-internal',
        internalPortId: 'default',
        externalPortId: 'input-1',
        externalPortLabel: 'Input',
        type: 'input',
        semantic: 'data'
      },
      {
        internalNodeId: 'conv-internal',
        internalPortId: 'default',
        externalPortId: 'output-1',
        externalPortLabel: 'Output',
        type: 'output',
        semantic: 'data'
      }
    ]

    const groupDef: GroupBlockDefinition = {
      id: 'test-group-1',
      name: 'Conv Block',
      description: 'Conv layer',
      category: 'custom',
      color: '#9333ea',
      internalNodes: [internalConv],
      internalEdges: [],
      portMappings,
      createdAt: Date.now(),
      updatedAt: Date.now()
    }

    useModelBuilderStore.getState().loadGroupDefinitions([groupDef])

    const groupNode: Node<BlockData> = {
      id: 'group-block-1',
      type: 'group',
      position: { x: 200, y: 0 },
      data: {
        blockType: 'group',
        label: 'Conv Block',
        config: {},
        category: 'custom',
        groupDefinitionId: 'test-group-1',
        isExpanded: false
      }
    }

    useModelBuilderStore.getState().addNode(inputNode)
    useModelBuilderStore.getState().addNode(groupNode as any)

    useModelBuilderStore.getState().addEdge({
      id: 'e1',
      source: 'input-1',
      target: 'group-block-1',
      targetHandle: 'input-1'
    })

    useModelBuilderStore.getState().inferDimensions()

    let nodes = useModelBuilderStore.getState().nodes
    let updatedGroupNode = nodes.find(n => n.id === 'group-block-1')
    expect(updatedGroupNode?.data.outputShape?.dims).toEqual([32, 64, 224, 224])

    // Change input shape
    useModelBuilderStore.getState().updateNode('input-1', {
      config: { shape: '[16, 3, 112, 112]' }
    })

    // Shape inference is triggered by updateNode
    nodes = useModelBuilderStore.getState().nodes
    updatedGroupNode = nodes.find(n => n.id === 'group-block-1')

    // Check that group block output shape was recomputed
    expect(updatedGroupNode?.data.outputShape?.dims).toEqual([16, 64, 112, 112])
  })

  it('should detect shape mismatches within blocks', () => {
    // Create internal nodes with incompatible shapes
    const internalLinear: Node<BlockData> = {
      id: 'linear-internal',
      type: 'block',
      position: { x: 0, y: 0 },
      data: {
        blockType: 'linear',
        label: 'Linear',
        config: { out_features: 128 },
        category: 'basic',
        inputShape: { dims: [32, 512] },
        outputShape: { dims: [32, 128] }
      }
    }

    // Conv2d expects 4D input but will receive 2D from linear
    const internalConv: Node<BlockData> = {
      id: 'conv-internal',
      type: 'block',
      position: { x: 100, y: 0 },
      data: {
        blockType: 'conv2d',
        label: 'Conv2D',
        config: { out_channels: 64, kernel_size: 3, stride: 1, padding: 1 },
        category: 'basic'
      }
    }

    const internalEdges: Edge[] = [
      {
        id: 'e1',
        source: 'linear-internal',
        target: 'conv-internal'
      }
    ]

    const portMappings: PortMapping[] = [
      {
        internalNodeId: 'linear-internal',
        internalPortId: 'default',
        externalPortId: 'input-1',
        externalPortLabel: 'Input',
        type: 'input',
        semantic: 'data'
      },
      {
        internalNodeId: 'conv-internal',
        internalPortId: 'default',
        externalPortId: 'output-1',
        externalPortLabel: 'Output',
        type: 'output',
        semantic: 'data'
      }
    ]

    const groupDef: GroupBlockDefinition = {
      id: 'test-group-1',
      name: 'Invalid Block',
      description: 'Block with shape mismatch',
      category: 'custom',
      color: '#9333ea',
      internalNodes: [internalLinear, internalConv],
      internalEdges,
      portMappings,
      createdAt: Date.now(),
      updatedAt: Date.now()
    }

    useModelBuilderStore.getState().loadGroupDefinitions([groupDef])

    const groupNode: Node<BlockData> = {
      id: 'group-block-1',
      type: 'group',
      position: { x: 200, y: 0 },
      data: {
        blockType: 'group',
        label: 'Invalid Block',
        config: {},
        category: 'custom',
        groupDefinitionId: 'test-group-1',
        isExpanded: false
      }
    }

    useModelBuilderStore.getState().addNode(groupNode as any)

    // Validate architecture
    const errors = useModelBuilderStore.getState().validateArchitecture()

    // Should have validation errors for the group block
    const groupErrors = errors.filter(e => e.nodeId === 'group-block-1')
    expect(groupErrors.length).toBeGreaterThan(0)
  })
})
