import { useState } from 'react'
import { Toaster } from 'sonner'
import Header from './components/Header'
import BlockPalette from './components/BlockPalette'
import Canvas from './components/Canvas'
import ConfigPanel from './components/ConfigPanel'

function App() {
  const [draggedType, setDraggedType] = useState<string | null>(null)

  const handleDragStart = (type: string) => {
    setDraggedType(type)
  }

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-background">
      <Header />
      
      <div className="flex-1 flex overflow-hidden">
        <BlockPalette onDragStart={handleDragStart} />
        <Canvas onDragStart={handleDragStart} />
        <ConfigPanel />
      </div>

      <Toaster position="bottom-right" richColors />
    </div>
  )
}

export default App
