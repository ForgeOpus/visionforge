import { useCallback, DragEvent } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Connection,
  useReactFlow,
  ReactFlowProvider
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

      if (definition.type === 'input') {
        newNode.data.config = {
          batch_size: 1,
          channels: 3,
          height: 224,
          width: 224
        }
      }

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

        if (targetNode?.data.blockType !== 'concat') {
          const hasInput = edges.some((e) => e.target === connection.target)
          if (hasInput) {
            toast.error('Block already has an input connection', {
              description: 'Use a Concatenate block for multiple inputs'
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
        }

        toast.error('Connection not allowed')
        return
      }

      const edge = {
        ...connection,
        id: `e${connection.source}-${connection.target}`,
        animated: true,
        style: { stroke: 'var(--color-accent)', strokeWidth: 2 }
      }

      addEdge(edge)
    },
    [validateConnection, addEdge, nodes, edges]
  )

  const onNodesChange = useCallback(
    (changes: any) => {
      const updatedNodes = [...nodes]
      changes.forEach((change: any) => {
        if (change.type === 'position' && change.dragging === false) {
          const nodeIndex = updatedNodes.findIndex((n) => n.id === change.id)
          if (nodeIndex !== -1) {
            updatedNodes[nodeIndex] = {
              ...updatedNodes[nodeIndex],
              position: change.position
            }
          }
        }
      })
      setNodes(updatedNodes)
    },
    [nodes, setNodes]
  )

  const onEdgesChange = useCallback(
    (changes: any) => {
      changes.forEach((change: any) => {
        if (change.type === 'remove') {
          removeEdge(change.id)
        }
      })
    },
    [removeEdge]
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
            return def?.color || '#ccc'
          }}
          className="bg-card border border-border"
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
