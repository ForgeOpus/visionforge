import { describe, it, expect, beforeEach } from 'vitest'
import { useModelBuilderStore } from './store'
import { GroupBlockDefinition, PortMapping } from './types'

describe('Block Instance Management', () => {
  beforeEach(() => {
    useModelBuilderStore.getState().reset()
  })

  it('should create unique node IDs for each group block instance', () => {
    // Create a group definition
    const groupDef: GroupBlockDefinition = {
      id: 'test-group-1',
      name: 'Test Block',
      description: 'Test description',
      category: 'custom',
      color: '#9333ea',
      internalNodes: [],
      internalEdges: [],
      portMappings: [],
      createdAt: Date.now(),
      updatedAt: Date.now()
    }

    useModelBuilderStore.getState().loadGroupDefinitions([groupDef])

    // Add first instance
    const instance1 = {
      id: 'group-block-1',
      type: 'group' as const,
      position: { x: 0, y: 0 },
      data: {
        blockType: 'group' as const,
        label: 'Test Block',
        config: {},
        category: 'custom' as const,
        groupDefinitionId: 'test-group-1',
        isExpanded: false
      }
    }

    useModelBuilderStore.getState().addNode(instance1 as any)

    // Add second instance
    const instance2 = {
      id: 'group-block-2',
      type: 'group' as const,
      position: { x: 100, y: 100 },
      data: {
        blockType: 'group' as const,
        label: 'Test Block',
        config: {},
        category: 'custom' as const,
        groupDefinitionId: 'test-group-1',
        isExpanded: false
      }
    }

    useModelBuilderStore.getState().addNode(instance2 as any)

    const nodes = useModelBuilderStore.getState().nodes
    expect(nodes).toHaveLength(2)
    expect(nodes[0].id).not.toBe(nodes[1].id)
    expect(nodes[0].id).toBe('group-block-1')
    expect(nodes[1].id).toBe('group-block-2')
  })

  it('should maintain independent configurations per instance', () => {
    // Create a group definition
    const groupDef: GroupBlockDefinition = {
      id: 'test-group-1',
      name: 'Test Block',
      description: 'Test description',
      category: 'custom',
      color: '#9333ea',
      internalNodes: [],
      internalEdges: [],
      portMappings: [],
      createdAt: Date.now(),
      updatedAt: Date.now()
    }

    useModelBuilderStore.getState().loadGroupDefinitions([groupDef])

    // Add two instances
    const instance1 = {
      id: 'group-block-1',
      type: 'group' as const,
      position: { x: 0, y: 0 },
      data: {
        blockType: 'group' as const,
        label: 'Test Block',
        config: { param1: 'value1' },
        category: 'custom' as const,
        groupDefinitionId: 'test-group-1',
        isExpanded: false
      }
    }

    const instance2 = {
      id: 'group-block-2',
      type: 'group' as const,
      position: { x: 100, y: 100 },
      data: {
        blockType: 'group' as const,
        label: 'Test Block',
        config: { param1: 'value2' },
        category: 'custom' as const,
        groupDefinitionId: 'test-group-1',
        isExpanded: false
      }
    }

    useModelBuilderStore.getState().addNode(instance1 as any)
    useModelBuilderStore.getState().addNode(instance2 as any)

    // Modify first instance config
    useModelBuilderStore.getState().updateNode('group-block-1', { config: { param1: 'modified' } })

    const nodes = useModelBuilderStore.getState().nodes
    expect(nodes[0].data.config.param1).toBe('modified')
    expect(nodes[1].data.config.param1).toBe('value2')
  })

  it('should add repetition metadata when using repeatGroupBlock', () => {
    // Create a group definition
    const groupDef: GroupBlockDefinition = {
      id: 'test-group-1',
      name: 'Test Block',
      description: 'Test description',
      category: 'custom',
      color: '#9333ea',
      internalNodes: [],
      internalEdges: [],
      portMappings: [],
      createdAt: Date.now(),
      updatedAt: Date.now()
    }

    useModelBuilderStore.getState().loadGroupDefinitions([groupDef])

    // Add initial instance
    const instance1 = {
      id: 'group-block-1',
      type: 'group' as const,
      position: { x: 0, y: 0 },
      data: {
        blockType: 'group' as const,
        label: 'Test Block',
        config: {},
        category: 'custom' as const,
        groupDefinitionId: 'test-group-1',
        isExpanded: false
      }
    }

    useModelBuilderStore.getState().addNode(instance1 as any)

    // Repeat the block 2 times
    const newIds = useModelBuilderStore.getState().repeatGroupBlock('group-block-1', 2, 200, 0)

    expect(newIds).toHaveLength(2)

    const nodes = useModelBuilderStore.getState().nodes
    expect(nodes).toHaveLength(3) // Original + 2 repeats

    // Check source node has repetition metadata
    const sourceNode = nodes.find(n => n.id === 'group-block-1')
    expect(sourceNode?.data.repetitionMetadata).toBeDefined()
    expect(sourceNode?.data.repetitionMetadata?.index).toBe(0)
    expect(sourceNode?.data.repetitionMetadata?.totalCount).toBe(3)

    // Check repeated nodes have repetition metadata
    const repeatedNodes = nodes.filter(n => newIds.includes(n.id))
    expect(repeatedNodes).toHaveLength(2)
    
    repeatedNodes.forEach((node, idx) => {
      expect(node.data.repetitionMetadata).toBeDefined()
      expect(node.data.repetitionMetadata?.index).toBe(idx + 1)
      expect(node.data.repetitionMetadata?.totalCount).toBe(3)
      expect(node.data.repetitionMetadata?.sequenceId).toBe(sourceNode?.data.repetitionMetadata?.sequenceId)
    })
  })

  it('should position repeated blocks with correct spacing', () => {
    // Create a group definition
    const groupDef: GroupBlockDefinition = {
      id: 'test-group-1',
      name: 'Test Block',
      description: 'Test description',
      category: 'custom',
      color: '#9333ea',
      internalNodes: [],
      internalEdges: [],
      portMappings: [],
      createdAt: Date.now(),
      updatedAt: Date.now()
    }

    useModelBuilderStore.getState().loadGroupDefinitions([groupDef])

    // Add initial instance
    const instance1 = {
      id: 'group-block-1',
      type: 'group' as const,
      position: { x: 100, y: 100 },
      data: {
        blockType: 'group' as const,
        label: 'Test Block',
        config: {},
        category: 'custom' as const,
        groupDefinitionId: 'test-group-1',
        isExpanded: false
      }
    }

    useModelBuilderStore.getState().addNode(instance1 as any)

    // Repeat with spacing
    const spacingX = 250
    const spacingY = 50
    useModelBuilderStore.getState().repeatGroupBlock('group-block-1', 2, spacingX, spacingY)

    const nodes = useModelBuilderStore.getState().nodes
    const sourceNode = nodes.find(n => n.id === 'group-block-1')
    const repeatedNodes = nodes.filter(n => n.id !== 'group-block-1')

    expect(repeatedNodes[0].position.x).toBe(sourceNode!.position.x + spacingX)
    expect(repeatedNodes[0].position.y).toBe(sourceNode!.position.y + spacingY)

    expect(repeatedNodes[1].position.x).toBe(sourceNode!.position.x + spacingX * 2)
    expect(repeatedNodes[1].position.y).toBe(sourceNode!.position.y + spacingY * 2)
  })

  it('should reference the same group definition for all instances', () => {
    // Create a group definition
    const groupDef: GroupBlockDefinition = {
      id: 'test-group-1',
      name: 'Test Block',
      description: 'Test description',
      category: 'custom',
      color: '#9333ea',
      internalNodes: [],
      internalEdges: [],
      portMappings: [],
      createdAt: Date.now(),
      updatedAt: Date.now()
    }

    useModelBuilderStore.getState().loadGroupDefinitions([groupDef])

    // Add multiple instances
    for (let i = 0; i < 3; i++) {
      const instance = {
        id: `group-block-${i}`,
        type: 'group' as const,
        position: { x: i * 100, y: 0 },
        data: {
          blockType: 'group' as const,
          label: 'Test Block',
          config: {},
          category: 'custom' as const,
          groupDefinitionId: 'test-group-1',
          isExpanded: false
        }
      }
      useModelBuilderStore.getState().addNode(instance as any)
    }

    const nodes = useModelBuilderStore.getState().nodes
    expect(nodes).toHaveLength(3)

    // All instances should reference the same definition
    nodes.forEach(node => {
      expect(node.data.groupDefinitionId).toBe('test-group-1')
    })

    // Verify the definition exists and is shared
    const definition = useModelBuilderStore.getState().groupDefinitions.get('test-group-1')
    expect(definition).toBeDefined()
    expect(definition?.id).toBe('test-group-1')
  })
})
