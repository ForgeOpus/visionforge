import { useEffect, useRef } from 'react'
import { Card } from './ui/card'
import * as Icons from '@phosphor-icons/react'

interface BlockDefinitionContextMenuProps {
  x: number
  y: number
  definitionId: string
  definitionName: string
  instanceCount: number
  onClose: () => void
  onRename: (definitionId: string) => void
  onDuplicate: (definitionId: string) => void
  onDelete: (definitionId: string) => void
}

export function BlockDefinitionContextMenu({
  x,
  y,
  definitionId,
  definitionName,
  instanceCount,
  onClose,
  onRename,
  onDuplicate,
  onDelete
}: BlockDefinitionContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (e: Event) => {
      const target = e.target as Node
      if (menuRef.current && !menuRef.current.contains(target)) {
        onClose()
      }
    }

    const timeoutId = setTimeout(() => {
      document.addEventListener('pointerdown', handleClickOutside, true)
    }, 100)

    return () => {
      clearTimeout(timeoutId)
      document.removeEventListener('pointerdown', handleClickOutside, true)
    }
  }, [onClose])

  return (
    <Card
      ref={menuRef}
      className="fixed z-[100] py-1 gap-0 min-w-[200px] shadow-lg border rounded-md"
      style={{ left: `${x}px`, top: `${y}px` }}
    >
      <div className="text-xs font-semibold text-muted-foreground px-2 py-1.5 border-b">
        {definitionName}
      </div>
      
      <button
        className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-accent rounded-sm text-sm transition-colors"
        onClick={() => {
          onRename(definitionId)
          onClose()
        }}
      >
        <Icons.PencilSimple size={16} />
        <span>Rename</span>
      </button>
      
      <button
        className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-accent rounded-sm text-sm transition-colors"
        onClick={() => {
          onDuplicate(definitionId)
          onClose()
        }}
      >
        <Icons.Copy size={16} />
        <span>Duplicate</span>
      </button>
      
      <div className="h-px bg-border my-1 -mx-1" />
      
      <button
        className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-destructive/10 hover:text-destructive rounded-sm text-sm transition-colors"
        onClick={() => {
          onDelete(definitionId)
          onClose()
        }}
      >
        <Icons.Trash size={16} />
        <span>Delete</span>
        {instanceCount > 0 && (
          <span className="ml-auto text-xs">({instanceCount} instances)</span>
        )}
      </button>
    </Card>
  )
}
