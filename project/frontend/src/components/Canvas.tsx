import { useCallback, DragEvent, useRef, useEffect } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Connection,
  useReactFlow,
  ReactFlowProvider,
  applyNodeChanges,
  applyEdgeChanges,
  Node
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useModelBuilderStore } from '@/lib/store'
import { getNodeDefinition, BackendFramework } from '@/lib/nodes/registry'
import { BlockData, BlockType } from '@/lib/types'
import BlockNode from './BlockNode'
import CustomConnectionLine from './CustomConnectionLine'
import { HistoryToolbar } from './HistoryToolbar'
import { toast } from 'sonner'

const nodeTypes = {
  custom: BlockNode
}

interface CanvasProps {
  onDragStart: (type: string) => void
  onRegisterAddNode: (handler: (blockType: string) => void) => void
}

function FlowCanvas({ onRegisterAddNode }: { onRegisterAddNode: (handler: (blockType: string) => void) => void }) {
  const {
    nodes,
    edges,
    setNodes,
    setEdges,
    addNode,
    addEdge,
    removeEdge,
    setSelectedNodeId,
    validateConnection,
    undo,
    redo
  } = useModelBuilderStore()

  const { screenToFlowPosition, getViewport } = useReactFlow()
  const nextPositionOffset = useRef({ x: 0, y: 0 })

  // Keyboard shortcuts for undo/redo
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Check for Ctrl (Windows/Linux) or Cmd (Mac)
      const isMod = e.ctrlKey || e.metaKey
      
      if (isMod && e.key === 'z' && !e.shiftKey) {
        e.preventDefault()
        undo()
      } else if (isMod && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
        e.preventDefault()
        redo()
      }
    }
    
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [undo, redo])

  // Find a suitable position for a new node
  const findAvailablePosition = useCallback(() => {
    const viewport = getViewport()
    const GRID_SIZE = 50
    
    // Start from center of viewport
    const centerX = -viewport.x / viewport.zoom + (window.innerWidth / 2) / viewport.zoom
    const centerY = -viewport.y / viewport.zoom + (window.innerHeight / 2) / viewport.zoom
    
    // Use offset for new nodes (allows overlapping)
    const x = centerX + nextPositionOffset.current.x
    const y = centerY + nextPositionOffset.current.y
    
    // Update offset for next node
    nextPositionOffset.current = {
      x: (nextPositionOffset.current.x + GRID_SIZE) % (GRID_SIZE * 4),
      y: nextPositionOffset.current.y
    }
    
    if (nextPositionOffset.current.x === 0) {
      nextPositionOffset.current.y = (nextPositionOffset.current.y + GRID_SIZE) % (GRID_SIZE * 4)
    }
    
    return { x, y }
  }, [getViewport])

  // Handle block click from palette
  useEffect(() => {
    const handleBlockClickInternal = (blockType: string) => {
      const nodeDef = getNodeDefinition(blockType as BlockType, BackendFramework.PyTorch)
      if (!nodeDef) return

      const position = findAvailablePosition()

      const newNode = {
        id: `${blockType}-${Date.now()}`,
        type: 'custom',
        position,
        data: {
          blockType: nodeDef.metadata.type,
          label: nodeDef.metadata.label,
          config: {},
          category: nodeDef.metadata.category
        } as BlockData
      }

      nodeDef.configSchema.forEach((field) => {
        if (field.default !== undefined) {
          newNode.data.config[field.name] = field.default
        }
      })

      addNode(newNode)

      setTimeout(() => {
        useModelBuilderStore.getState().inferDimensions()
      }, 0)

      toast.success(`Added ${nodeDef.metadata.label}`, {
        description: 'Block added to canvas'
      })
    }
    
    // Register the handler with parent
    onRegisterAddNode(handleBlockClickInternal)
  }, [addNode, findAvailablePosition, onRegisterAddNode])

  const onDragOver = useCallback((event: DragEvent) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault()

      const type = (window as any).draggedBlockTypeGlobal
      if (!type) return

      const nodeDef = getNodeDefinition(type as BlockType, BackendFramework.PyTorch)
      if (!nodeDef) return

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY
      })

      const newNode = {
        id: `${type}-${Date.now()}`,
        type: 'custom',
        position,
        data: {
          blockType: nodeDef.metadata.type,
          label: nodeDef.metadata.label,
          config: {},
          category: nodeDef.metadata.category
        } as BlockData
      }

      nodeDef.configSchema.forEach((field) => {
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

        // Use the validation function to get specific error message
        if (sourceNode && targetNode) {
          const targetNodeDef = getNodeDefinition(
            targetNode.data.blockType as BlockType,
            BackendFramework.PyTorch
          )
          
          if (!targetNodeDef) {
            toast.error('Connection Invalid', {
              description: 'Invalid target node type'
            })
            return
          }
          
          const errorMessage = targetNodeDef.validateIncomingConnection(
            sourceNode.data.blockType as BlockType,
            sourceNode.data.outputShape,
            targetNode.data.config
          )
          
          if (errorMessage) {
            toast.error('Connection Invalid', {
              description: errorMessage
            })
            return
          }
        }

        // Fallback for other validation errors
        if (targetNode?.data.blockType !== 'concat' && targetNode?.data.blockType !== 'add') {
          const hasInput = edges.some((e) => e.target === connection.target)
          if (hasInput) {
            toast.error('Block already has an input connection', {
              description: 'Use a Concatenate or Add block for multiple inputs'
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

  const onNodesChange = useCallback(
    (changes: any) => {
      const updatedNodes = applyNodeChanges(changes, nodes)
      setNodes(updatedNodes)
    },
    [nodes, setNodes]
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
      <HistoryToolbar />
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        connectionLineComponent={CustomConnectionLine}
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
            const nodeDef = getNodeDefinition(
              (node.data as BlockData).blockType as BlockType,
              BackendFramework.PyTorch
            )
            return nodeDef?.metadata.color || '#3b82f6'
          }}
          nodeStrokeColor={(node) => {
            const nodeDef = getNodeDefinition(
              (node.data as BlockData).blockType as BlockType,
              BackendFramework.PyTorch
            )
            const baseColor = nodeDef?.metadata.color || '#3b82f6'
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

export default function Canvas({ onDragStart, onRegisterAddNode }: CanvasProps) {
  return (
    <ReactFlowProvider>
      <FlowCanvas onRegisterAddNode={onRegisterAddNode} />
    </ReactFlowProvider>
  )
}

export const draggedBlockTypeGlobal = null
