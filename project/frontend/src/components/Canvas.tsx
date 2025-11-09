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
import { getBlockDefinition, validateBlockConnection } from '@/lib/blockDefinitions'
import { BlockData } from '@/lib/types'
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

  // Helper function to check if a position overlaps with existing nodes
  const isPositionOverlapping = useCallback((x: number, y: number, nodes: Node<BlockData>[]) => {
    const NODE_WIDTH = 200
    const NODE_HEIGHT = 80
    const MARGIN = 20

    return nodes.some(node => {
      const nodeX = node.position.x
      const nodeY = node.position.y
      
      return (
        x < nodeX + NODE_WIDTH + MARGIN &&
        x + NODE_WIDTH + MARGIN > nodeX &&
        y < nodeY + NODE_HEIGHT + MARGIN &&
        y + NODE_HEIGHT + MARGIN > nodeY
      )
    })
  }, [])

  // Find a suitable position for a new node
  const findAvailablePosition = useCallback(() => {
    const viewport = getViewport()
    const NODE_WIDTH = 200
    const NODE_HEIGHT = 80
    const GRID_SIZE = 50
    
    // Start from center of viewport
    const centerX = -viewport.x / viewport.zoom + (window.innerWidth / 2) / viewport.zoom
    const centerY = -viewport.y / viewport.zoom + (window.innerHeight / 2) / viewport.zoom
    
    // Try positions in a spiral pattern from center
    let x = centerX + nextPositionOffset.current.x
    let y = centerY + nextPositionOffset.current.y
    
    // If this position overlaps, try nearby positions
    let attempts = 0
    const maxAttempts = 100
    
    while (isPositionOverlapping(x, y, nodes) && attempts < maxAttempts) {
      attempts++
      // Try in a grid pattern
      const offset = Math.ceil(attempts / 4) * GRID_SIZE
      const direction = attempts % 4
      
      switch (direction) {
        case 0: // right
          x = centerX + offset
          y = centerY
          break
        case 1: // down
          x = centerX
          y = centerY + offset
          break
        case 2: // left
          x = centerX - offset
          y = centerY
          break
        case 3: // up
          x = centerX
          y = centerY - offset
          break
      }
    }
    
    // Update offset for next node
    nextPositionOffset.current = {
      x: (nextPositionOffset.current.x + GRID_SIZE) % (GRID_SIZE * 4),
      y: nextPositionOffset.current.y
    }
    
    if (nextPositionOffset.current.x === 0) {
      nextPositionOffset.current.y = (nextPositionOffset.current.y + GRID_SIZE) % (GRID_SIZE * 4)
    }
    
    return { x, y }
  }, [getViewport, isPositionOverlapping, nodes])

  // Handle block click from palette
  useEffect(() => {
    const handleBlockClickInternal = (blockType: string) => {
      const definition = getBlockDefinition(blockType)
      if (!definition) return

      const position = findAvailablePosition()

      const newNode = {
        id: `${blockType}-${Date.now()}`,
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

      toast.success(`Added ${definition.label}`, {
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

        // Use the validation function to get specific error message
        if (sourceNode && targetNode) {
          const errorMessage = validateBlockConnection(
            sourceNode.data.blockType,
            targetNode.data.blockType,
            sourceNode.data.outputShape
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

export default function Canvas({ onDragStart, onRegisterAddNode }: CanvasProps) {
  return (
    <ReactFlowProvider>
      <FlowCanvas onRegisterAddNode={onRegisterAddNode} />
    </ReactFlowProvider>
  )
}

export const draggedBlockTypeGlobal = null
