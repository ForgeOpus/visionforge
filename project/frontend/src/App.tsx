import { useState, useRef, useEffect } from 'react'
import { Routes, Route, useNavigate, useParams } from 'react-router-dom'
import { Toaster } from 'sonner'
import { toast } from 'sonner'
import Header from './components/Header'
import ResizableBlockPalette from './components/ResizableBlockPalette'
import Canvas from './components/Canvas'
import ConfigPanel from './components/ConfigPanel'
import ChatBot from './components/ChatBot'
import { useModelBuilderStore } from './lib/store'
import { fetchProject, loadArchitecture, convertToFrontendProject } from './lib/projectApi'
import { LandingPage } from './landing'

function ProjectCanvas() {
  const { projectId } = useParams<{ projectId: string }>()
  const navigate = useNavigate()
  const { setNodes, setEdges, loadProject, currentProject, reset } = useModelBuilderStore()
  const [isLoading, setIsLoading] = useState(false)
  const [draggedType, setDraggedType] = useState<string | null>(null)
  const { selectedNodeId } = useModelBuilderStore()
  const addNodeFromPaletteRef = useRef<((blockType: string) => void) | null>(null)

  // Load project from URL parameter
  useEffect(() => {
    if (projectId && (!currentProject || currentProject.id !== projectId)) {
      setIsLoading(true)
      fetchProject(projectId)
        .then(async (backendProject) => {
          // Load architecture if it exists
          try {
            const { nodes, edges } = await loadArchitecture(projectId)
            const project = convertToFrontendProject(backendProject, nodes, edges)
            loadProject(project)
          } catch (error) {
            // No architecture yet, just load project metadata
            const project = convertToFrontendProject(backendProject)
            loadProject(project)
          }
        })
        .catch((error) => {
          console.error('Failed to load project:', error)
          toast.error('Failed to load project', {
            description: error instanceof Error ? error.message : 'Unknown error'
          })
          // Don't navigate away, just show error
        })
        .finally(() => {
          setIsLoading(false)
        })
    }
  }, [projectId, currentProject, setNodes, setEdges, loadProject, navigate])

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

  if (isLoading) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading project...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-background">
      <Header />

      <div className="flex-1 flex overflow-hidden relative">
        <ResizableBlockPalette
          onDragStart={handleDragStart}
          onBlockClick={handleBlockClick}
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

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/project" element={<ProjectCanvas />} />
      <Route path="/project/:projectId" element={<ProjectCanvas />} />
    </Routes>
  )
}

export default App
