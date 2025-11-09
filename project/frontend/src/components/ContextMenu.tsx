import { useEffect, useRef } from 'react'
import { Card } from './ui/card'
import { getNodeDefinition, BackendFramework } from '@/lib/nodes/registry'
import { BlockType } from '@/lib/types'
import * as Icons from '@phosphor-icons/react'

interface ContextMenuProps {
  x: number
  y: number
  type: 'canvas' | 'node'
  nodeId?: string
  recentlyUsedNodes?: BlockType[]
  onClose: () => void
  onAddNode?: (nodeType: BlockType, x: number, y: number) => void
  onDeleteNode?: (nodeId: string) => void
  onDuplicateNode?: (nodeId: string) => void
  onReplicateNode?: (nodeId: string) => void
}

export function ContextMenu({
  x,
  y,
  type,
  nodeId,
  recentlyUsedNodes = [],
  onClose,
  onAddNode,
  onDeleteNode,
  onDuplicateNode,
  onReplicateNode
}: ContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (e: Event) => {
      const target = e.target as Node
      if (menuRef.current && !menuRef.current.contains(target)) {
        onClose()
      }
    }

    // Use pointerdown which works better with ReactFlow
    // Add listener with capture phase to catch events before ReactFlow
    const timeoutId = setTimeout(() => {
      document.addEventListener('pointerdown', handleClickOutside, true)
    }, 100) // Increased delay to ensure menu is fully rendered

    return () => {
      clearTimeout(timeoutId)
      document.removeEventListener('pointerdown', handleClickOutside, true)
    }
  }, [onClose])

  if (type === 'canvas') {
    return (
      <Card
        ref={menuRef}
        className="fixed z-[100] py-1 gap-0 min-w-[200px] shadow-lg border rounded-md"
        style={{ left: `${x}px`, top: `${y}px` }}
      >
        <div className="text-xs font-semibold text-muted-foreground px-2 py-1.5">
          Recently Used Nodes
        </div>
        {recentlyUsedNodes.length > 0 ? (
          recentlyUsedNodes.map((nodeType) => {
            const nodeDef = getNodeDefinition(nodeType, BackendFramework.PyTorch)
            if (!nodeDef) return null

            const Icon = (Icons as any)[nodeDef.metadata.icon]

            return (
              <button
                key={nodeType}
                className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-accent rounded-sm text-sm transition-colors"
                onClick={() => {
                  onAddNode?.(nodeType, x, y)
                  onClose()
                }}
              >
                {Icon && <Icon size={16} />}
                <span>{nodeDef.metadata.label}</span>
              </button>
            )
          })
        ) : (
          <div className="text-xs text-muted-foreground px-2 py-1.5">
            No recently used nodes
          </div>
        )}
      </Card>
    )
  }

  // Node context menu
  return (
    <Card
      ref={menuRef}
      className="fixed z-[100] py-1 gap-0 min-w-[180px] shadow-lg border rounded-md"
      style={{ left: `${x}px`, top: `${y}px` }}
    >
      <button
        className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-accent rounded-sm text-sm transition-colors"
        onClick={() => {
          if (nodeId) onDuplicateNode?.(nodeId)
          onClose()
        }}
      >
        <Icons.Copy size={16} />
        <span>Duplicate</span>
      </button>
      <button
        className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-accent rounded-sm text-sm transition-colors"
        onClick={() => {
          if (nodeId) onReplicateNode?.(nodeId)
          onClose()
        }}
      >
        <Icons.Code size={16} />
        <span>Replicate as Custom</span>
      </button>
      <div className="h-px bg-border my-1 -mx-1" />
      <button
        className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-destructive/10 hover:text-destructive rounded-sm text-sm transition-colors"
        onClick={() => {
          if (nodeId) onDeleteNode?.(nodeId)
          onClose()
        }}
      >
        <Icons.Trash size={16} />
        <span>Delete</span>
      </button>
    </Card>
  )
}
