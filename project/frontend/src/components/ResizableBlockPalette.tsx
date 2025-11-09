import { useState, useRef, useEffect } from 'react'
import BlockPalette from './BlockPalette'
import * as Icons from '@phosphor-icons/react'

interface ResizableBlockPaletteProps {
  onDragStart: (blockType: string) => void
  onBlockClick: (blockType: string) => void
}

const COLLAPSED_WIDTH = 64 // 16 * 4 = 64px (w-16)
const MIN_WIDTH = 200
const MAX_WIDTH = 500
const PRESET_WIDTHS = [250, 320, 400] // Small, Medium, Large presets

export default function ResizableBlockPalette({ onDragStart, onBlockClick }: ResizableBlockPaletteProps) {
  const [width, setWidth] = useState(256) // Default 256px (w-64)
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [isResizing, setIsResizing] = useState(false)
  const resizeRef = useRef<HTMLDivElement>(null)
  const presetIndexRef = useRef(0)

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault()
    setIsResizing(true)
  }

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return

      const newWidth = e.clientX
      if (newWidth >= MIN_WIDTH && newWidth <= MAX_WIDTH) {
        setWidth(newWidth)
      }
    }

    const handleMouseUp = () => {
      setIsResizing(false)
    }

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = 'col-resize'
      document.body.style.userSelect = 'none'
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
  }, [isResizing])

  const handleToggleCollapse = () => {
    setIsCollapsed(!isCollapsed)
  }

  const cyclePresetSizes = () => {
    if (isCollapsed) {
      setIsCollapsed(false)
      setWidth(PRESET_WIDTHS[0])
      presetIndexRef.current = 0
    } else {
      presetIndexRef.current = (presetIndexRef.current + 1) % (PRESET_WIDTHS.length + 1)

      if (presetIndexRef.current === PRESET_WIDTHS.length) {
        // Collapse after last preset
        setIsCollapsed(true)
      } else {
        setWidth(PRESET_WIDTHS[presetIndexRef.current])
      }
    }
  }

  return (
    <div
      ref={resizeRef}
      className="relative h-full bg-card border-r border-border flex-shrink-0 transition-none"
      style={{ width: isCollapsed ? COLLAPSED_WIDTH : width }}
    >
      {/* Top resize/toggle button - VS Code style */}
      <div className="absolute -top-0 left-0 right-0 z-40 flex items-center justify-between px-2 py-1 bg-card border-b border-border">
        <button
          onClick={cyclePresetSizes}
          className="p-1 hover:bg-accent rounded transition-colors group"
          title={isCollapsed ? "Expand panel" : "Cycle panel size"}
        >
          {isCollapsed ? (
            <Icons.CaretRight size={16} weight="bold" className="text-muted-foreground group-hover:text-foreground" />
          ) : (
            <Icons.ArrowsLeftRight size={16} weight="bold" className="text-muted-foreground group-hover:text-foreground" />
          )}
        </button>

        {!isCollapsed && (
          <div className="flex items-center gap-1">
            <span className="text-xs text-muted-foreground">{width}px</span>
            <button
              onClick={handleToggleCollapse}
              className="p-1 hover:bg-accent rounded transition-colors group"
              title="Collapse panel"
            >
              <Icons.CaretLeft size={14} weight="bold" className="text-muted-foreground group-hover:text-foreground" />
            </button>
          </div>
        )}
      </div>

      {/* Block Palette Content */}
      <div className="h-full pt-8">
        <BlockPalette
          onDragStart={onDragStart}
          onBlockClick={onBlockClick}
          isCollapsed={isCollapsed}
        />
      </div>

      {/* Resize Handle - Right Edge */}
      {!isCollapsed && (
        <div
          className={`absolute top-0 right-0 bottom-0 w-1 cursor-col-resize hover:bg-primary/50 transition-colors ${
            isResizing ? 'bg-primary' : ''
          }`}
          onMouseDown={handleMouseDown}
        >
          {/* Visual indicator on hover */}
          <div className="absolute inset-y-0 -left-1 -right-1" />
        </div>
      )}
    </div>
  )
}
