import { useState, useRef } from 'react'
import { Toaster } from 'sonner'
import Header from './components/Header'
import BlockPalette from './components/BlockPalette'
import Canvas from './components/Canvas'
import ConfigPanel from './components/ConfigPanel'
import ChatBot from './components/ChatBot'
import { useModelBuilderStore } from './lib/store'

function App() {
  const [draggedType, setDraggedType] = useState<string | null>(null)
  const [isPaletteCollapsed, setIsPaletteCollapsed] = useState(false)
  const { selectedNodeId } = useModelBuilderStore()
  const addNodeFromPaletteRef = useRef<((blockType: string) => void) | null>(null)

  const handleDragStart = (type: string) => {
    setDraggedType(type)
  }

  const handleBlockClick = (blockType: string) => {
    if (addNodeFromPaletteRef.current) {
      addNodeFromPaletteRef.current(blockType)
    }
  }

  const registerAddNodeHandler = (handler: (blockType: string) => void) => {
    addNodeFromPaletteRef.current = handler
  }

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-background">
      <Header />

      <div className="flex-1 flex overflow-hidden relative">
        <BlockPalette 
          onDragStart={handleDragStart}
          onBlockClick={handleBlockClick}
          isCollapsed={isPaletteCollapsed}
          onToggleCollapse={() => setIsPaletteCollapsed(!isPaletteCollapsed)}
        />
        <Canvas 
          onDragStart={handleDragStart}
          onRegisterAddNode={registerAddNodeHandler}
        />
        {selectedNodeId && <ConfigPanel />}
      </div>

      <ChatBot />
      <Toaster position="bottom-right" richColors />
    </div>
  )
}

export default App
