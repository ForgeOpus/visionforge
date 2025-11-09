import { useCallback, DragEvent } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Connection,
  useReactFlow,
  ReactFlowProvider,
  applyNodeChanges,
  applyEdgeChanges
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useModelBuilderStore } from '@/lib/store'
import { getBlockDefinition } from '@/lib/blockDefinitions'
import { BlockData } from '@/lib/types'
import BlockNode from './BlockNode'
import { toast } from 'sonner'

const nodeTypes = {
  custom: BlockNode
}

function FlowCanvas() {
  const {
    nodes,
    edges,
    setNodes,
    setEdges,
    addNode,
    addEdge,
    removeEdge,
    setSelectedNodeId,
    validateConnection
  } = useModelBuilderStore()

  const { screenToFlowPosition } = useReactFlow()

  const onDragOver = useCallback((event: DragEvent) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault()

      const type = (window as any).draggedBlockTypeGlobal
      if (!type) return

      const definition = getBlockDefinition(type)
      if (!definition) return

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY
      })

      const newNode = {
        id: `${type}-${Date.now()}`,
        type: 'custom',
        position,
        data: {
          blockType: definition.type,
          label: definition.label,
          config: {},
          category: definition.category
        } as BlockData
      }

      Object.values(definition.configSchema).forEach((field) => {
        if (field.default !== undefined) {
          newNode.data.config[field.name] = field.default
        }
      })

      addNode(newNode)

      setTimeout(() => {
        useModelBuilderStore.getState().inferDimensions()
      }, 0)

      ;(window as any).draggedBlockTypeGlobal = null
    },
    [addNode, screenToFlowPosition]
  )

  const onConnect = useCallback(
    (connection: Connection) => {
      const isValid = validateConnection(connection)

      if (!isValid) {
        const sourceNode = nodes.find((n) => n.id === connection.source)
        const targetNode = nodes.find((n) => n.id === connection.target)

        if (targetNode?.data.blockType !== 'concat' && targetNode?.data.blockType !== 'add') {
          const hasInput = edges.some((e) => e.target === connection.target)
          if (hasInput) {
            toast.error('Block already has an input connection', {
              description: 'Use a Concatenate or Add block for multiple inputs'
            })
            return
          }
        }

        if (sourceNode?.data.outputShape && targetNode) {
          const sourceShape = sourceNode.data.outputShape
          const targetType = targetNode.data.blockType

          if (targetType === 'linear' && sourceShape.dims.length !== 2) {
            toast.error('Shape incompatible with Linear layer', {
              description: 'Linear requires 2D input. Add a Flatten layer first.'
            })
            return
          }

          if (targetType === 'conv2d' && sourceShape.dims.length !== 4) {
            toast.error('Shape incompatible with Conv2D layer', {
              description: 'Conv2D requires 4D input [B, C, H, W]'
            })
            return
          }

          if (targetType === 'add') {
            toast.error('Shape incompatible for addition', {
              description: 'All inputs to Add must have the same shape'
            })
            return
          }
        }

        toast.error('Connection not allowed')
        return
      }

      const edge = {
        ...connection,
        id: `e${connection.source}-${connection.target}-${Date.now()}`,
        animated: true,
        style: { stroke: 'var(--color-accent)', strokeWidth: 2 }
      }

      addEdge(edge)
      
      toast.success('Connection created', {
        description: 'Input shape automatically configured'
      })
    },
    [validateConnection, addEdge, nodes, edges]
  )

  const checkCollision = useCallback((nodeA: any, nodeB: any) => {
    const padding = 20 // Extra space between nodes
    const nodeWidth = 220
    const nodeHeight = 100 // Approximate height

    const aLeft = nodeA.position.x
    const aRight = nodeA.position.x + nodeWidth
    const aTop = nodeA.position.y
    const aBottom = nodeA.position.y + nodeHeight

    const bLeft = nodeB.position.x
    const bRight = nodeB.position.x + nodeWidth
    const bTop = nodeB.position.y
    const bBottom = nodeB.position.y + nodeHeight

    return !(
      aRight + padding < bLeft ||
      aLeft - padding > bRight ||
      aBottom + padding < bTop ||
      aTop - padding > bBottom
    )
  }, [])

  const resolveCollisions = useCallback((updatedNodes: any[]) => {
    const resolvedNodes = [...updatedNodes]

    for (let i = 0; i < resolvedNodes.length; i++) {
      for (let j = i + 1; j < resolvedNodes.length; j++) {
        const nodeA = resolvedNodes[i]
        const nodeB = resolvedNodes[j]

        if (checkCollision(nodeA, nodeB)) {
          // Calculate push direction
          const dx = nodeB.position.x - nodeA.position.x
          const dy = nodeB.position.y - nodeA.position.y
          const distance = Math.sqrt(dx * dx + dy * dy)

          if (distance === 0) continue

          // Normalize and apply bounce
          const pushDistance = 240 // How far to push apart
          const nx = (dx / distance) * pushDistance
          const ny = (dy / distance) * pushDistance

          // Push the node being dragged (nodeB is typically the one being moved)
          resolvedNodes[j] = {
            ...nodeB,
            position: {
              x: nodeA.position.x + nx,
              y: nodeA.position.y + ny
            }
          }
        }
      }
    }

    return resolvedNodes
  }, [checkCollision])

  const onNodesChange = useCallback(
    (changes: any) => {
      let updatedNodes = applyNodeChanges(changes, nodes)

      // Check if any position changes occurred
      const hasPositionChange = changes.some((change: any) => change.type === 'position')

      if (hasPositionChange) {
        updatedNodes = resolveCollisions(updatedNodes)
      }

      setNodes(updatedNodes)
    },
    [nodes, setNodes, resolveCollisions]
  )

  const onEdgesChange = useCallback(
    (changes: any) => {
      setEdges(applyEdgeChanges(changes, edges))
      // Handle edge removal for store cleanup
      changes.forEach((change: any) => {
        if (change.type === 'remove') {
          removeEdge(change.id)
        }
      })
    },
    [edges, setEdges, removeEdge]
  )

  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: any) => {
      setSelectedNodeId(node.id)
    },
    [setSelectedNodeId]
  )

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null)
  }, [setSelectedNodeId])

  return (
    <div
      className="flex-1 h-full"
      onDrop={onDrop}
      onDragOver={onDragOver}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        fitView
        minZoom={0.5}
        maxZoom={1.5}
        defaultEdgeOptions={{
          animated: true,
          style: { stroke: 'var(--color-accent)', strokeWidth: 2 }
        }}
      >
        <Background gap={20} size={1} />
        <Controls />
        <MiniMap
          nodeColor={(node) => {
            const def = getBlockDefinition((node.data as BlockData).blockType)
            return def?.color || '#3b82f6'
          }}
          nodeStrokeColor={(node) => {
            const def = getBlockDefinition((node.data as BlockData).blockType)
            const baseColor = def?.color || '#3b82f6'
            // Return a slightly darker version for the stroke
            return baseColor
          }}
          nodeStrokeWidth={2}
          nodeBorderRadius={4}
          maskColor="rgba(0, 0, 0, 0.05)"
          className="bg-card border border-border rounded-lg shadow-lg"
          style={{
            backgroundColor: 'var(--card)',
            borderColor: 'var(--border)',
          }}
          zoomable
          pannable
          position="bottom-right"
        />
      </ReactFlow>
    </div>
  )
}

export default function Canvas({ onDragStart }: { onDragStart: (type: string) => void }) {
  return (
    <ReactFlowProvider>
      <FlowCanvas />
    </ReactFlowProvider>
  )
}

export const draggedBlockTypeGlobal = null
