import { describe, it, expect, beforeEach } from 'vitest'
import { useModelBuilderStore } from './store'
import { GroupBlockDefinition } from './types'

describe('Block Library Management', () => {
  beforeEach(() => {
    useModelBuilderStore.getState().reset()
  })

  it('should rename a group definition and update all instances', () => {
    // Create a group definition
    const groupDef: GroupBlockDefinition = {
      id: 'test-group-1',
      name: 'Original Name',
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
        label: 'Original Name',
        config: {},
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
        label: 'Original Name',
        config: {},
        category: 'custom' as const,
        groupDefinitionId: 'test-group-1',
        isExpanded: false
      }
    }

    useModelBuilderStore.getState().addNode(instance1 as any)
    useModelBuilderStore.getState().addNode(instance2 as any)

    // Rename the definition
    useModelBuilderStore.getState().renameGroupDefinition('test-group-1', 'New Name')

    // Check definition was renamed
    const updatedDef = useModelBuilderStore.getState().groupDefinitions.get('test-group-1')
    expect(updatedDef?.name).toBe('New Name')

    // Check all instances were updated
    const nodes = useModelBuilderStore.getState().nodes
    expect(nodes[0].data.label).toBe('New Name')
    expect(nodes[1].data.label).toBe('New Name')
  })

  it('should delete a group definition without cascade', () => {
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

    // Add an instance
    const instance = {
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

    useModelBuilderStore.getState().addNode(instance as any)

    // Delete definition without cascade
    useModelBuilderStore.getState().deleteGroupDefinition('test-group-1', false)

    // Definition should be removed
    const definition = useModelBuilderStore.getState().groupDefinitions.get('test-group-1')
    expect(definition).toBeUndefined()

    // Instance should still exist
    const nodes = useModelBuilderStore.getState().nodes
    expect(nodes).toHaveLength(1)
    expect(nodes[0].id).toBe('group-block-1')

    // Validation should report error
    const errors = useModelBuilderStore.getState().validateArchitecture()
    const hasDefinitionError = errors.some(e => 
      e.nodeId === 'group-block-1' && 
      e.message.includes('missing definition')
    )
    expect(hasDefinitionError).toBe(true)
  })

  it('should delete a group definition with cascade', () => {
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
        config: {},
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
        config: {},
        category: 'custom' as const,
        groupDefinitionId: 'test-group-1',
        isExpanded: false
      }
    }

    useModelBuilderStore.getState().addNode(instance1 as any)
    useModelBuilderStore.getState().addNode(instance2 as any)

    // Add a regular node to ensure it's not deleted
    const regularNode = {
      id: 'linear-1',
      type: 'default' as const,
      position: { x: 200, y: 200 },
      data: {
        blockType: 'linear' as const,
        label: 'Linear',
        config: {},
        category: 'basic' as const
      }
    }

    useModelBuilderStore.getState().addNode(regularNode as any)

    // Delete definition with cascade
    useModelBuilderStore.getState().deleteGroupDefinition('test-group-1', true)

    // Definition should be removed
    const definition = useModelBuilderStore.getState().groupDefinitions.get('test-group-1')
    expect(definition).toBeUndefined()

    // All instances should be removed
    const nodes = useModelBuilderStore.getState().nodes
    expect(nodes).toHaveLength(1)
    expect(nodes[0].id).toBe('linear-1')
  })

  it('should delete edges connected to cascaded instances', () => {
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

    // Add instance
    const instance = {
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

    const regularNode = {
      id: 'linear-1',
      type: 'default' as const,
      position: { x: 200, y: 200 },
      data: {
        blockType: 'linear' as const,
        label: 'Linear',
        config: {},
        category: 'basic' as const
      }
    }

    useModelBuilderStore.getState().addNode(instance as any)
    useModelBuilderStore.getState().addNode(regularNode as any)

    // Add edge connecting them
    const edge = {
      id: 'edge-1',
      source: 'group-block-1',
      target: 'linear-1'
    }

    useModelBuilderStore.getState().addEdge(edge as any)

    // Verify edge exists
    expect(useModelBuilderStore.getState().edges).toHaveLength(1)

    // Delete definition with cascade
    useModelBuilderStore.getState().deleteGroupDefinition('test-group-1', true)

    // Edge should be removed
    expect(useModelBuilderStore.getState().edges).toHaveLength(0)
  })

  it('should duplicate a group definition with unique name', () => {
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

    // Duplicate the definition
    const newId = useModelBuilderStore.getState().duplicateGroupDefinition('test-group-1')

    expect(newId).toBeTruthy()
    expect(newId).not.toBe('test-group-1')

    // Check new definition exists
    const newDef = useModelBuilderStore.getState().groupDefinitions.get(newId)
    expect(newDef).toBeDefined()
    expect(newDef?.name).toBe('Test Block Copy')
    expect(newDef?.description).toBe('Test description')
    expect(newDef?.color).toBe('#9333ea')

    // Original should still exist
    const originalDef = useModelBuilderStore.getState().groupDefinitions.get('test-group-1')
    expect(originalDef).toBeDefined()
  })

  it('should generate unique names for multiple duplicates', () => {
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

    // Duplicate multiple times
    const id1 = useModelBuilderStore.getState().duplicateGroupDefinition('test-group-1')
    const id2 = useModelBuilderStore.getState().duplicateGroupDefinition('test-group-1')
    const id3 = useModelBuilderStore.getState().duplicateGroupDefinition('test-group-1')

    const def1 = useModelBuilderStore.getState().groupDefinitions.get(id1)
    const def2 = useModelBuilderStore.getState().groupDefinitions.get(id2)
    const def3 = useModelBuilderStore.getState().groupDefinitions.get(id3)

    expect(def1?.name).toBe('Test Block Copy')
    expect(def2?.name).toBe('Test Block Copy 2')
    expect(def3?.name).toBe('Test Block Copy 3')

    // All should have unique IDs
    expect(new Set([id1, id2, id3]).size).toBe(3)
  })

  it('should not affect instances when duplicating a definition', () => {
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

    // Add instance
    const instance = {
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

    useModelBuilderStore.getState().addNode(instance as any)

    // Duplicate the definition
    const newId = useModelBuilderStore.getState().duplicateGroupDefinition('test-group-1')

    // Original instance should still reference original definition
    const nodes = useModelBuilderStore.getState().nodes
    expect(nodes).toHaveLength(1)
    expect(nodes[0].data.groupDefinitionId).toBe('test-group-1')
    expect(nodes[0].data.label).toBe('Test Block')
  })
})
