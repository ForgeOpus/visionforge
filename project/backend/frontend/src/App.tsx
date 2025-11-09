import { useState } from 'react'
import { Toaster } from 'sonner'
import Header from './components/Header'
import BlockPalette from './components/BlockPalette'
import Canvas from './components/Canvas'
import ConfigPanel from './components/ConfigPanel'
import ChatBot from './components/ChatBot'
import { useModelBuilderStore } from './lib/store'
import * as Icons from '@phosphor-icons/react'
import { Button } from './components/ui/button'

function App() {
  const [draggedType, setDraggedType] = useState<string | null>(null)
  const [isPaletteCollapsed, setIsPaletteCollapsed] = useState(false)
  const { selectedNodeId } = useModelBuilderStore()

  const handleDragStart = (type: string) => {
    setDraggedType(type)
  }

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-background">
      <Header />

      <div className="flex-1 flex overflow-hidden relative">
        {/* Toggle Button for Block Palette */}
        <Button
          variant="ghost"
          size="icon"
          className="absolute left-2 top-2 z-20 bg-card border border-border shadow-sm"
          onClick={() => setIsPaletteCollapsed(!isPaletteCollapsed)}
        >
          {isPaletteCollapsed ? (
            <Icons.CaretRight size={18} />
          ) : (
            <Icons.CaretLeft size={18} />
          )}
        </Button>

        <BlockPalette onDragStart={handleDragStart} isCollapsed={isPaletteCollapsed} />
        <Canvas onDragStart={handleDragStart} />
        {selectedNodeId && <ConfigPanel />}
      </div>

      <ChatBot />
      <Toaster position="bottom-right" richColors />
    </div>
  )
}

export default App
