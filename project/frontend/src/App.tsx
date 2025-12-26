import { useState, useRef, useEffect, lazy, Suspense } from 'react'
import { Routes, Route, useNavigate, useParams } from 'react-router-dom'
import { Toaster } from 'sonner'
import { toast } from 'sonner'
import Header from './components/Header'
import { useModelBuilderStore } from './lib/store'
import { fetchProject, loadArchitecture, convertToFrontendProject } from './lib/projectApi'
import { ProtectedRoute } from './components/ProtectedRoute'
import { useAuth } from './contexts/AuthContext'
import { useSessionTracking, useCanvasTimeTracking } from './hooks/useSessionTracking'

// Lazy load heavy components for better performance
const Canvas = lazy(() => import('./components/Canvas'))
const ResizableBlockPalette = lazy(() => import('./components/ResizableBlockPalette'))
const ConfigPanel = lazy(() => import('./components/ConfigPanel'))
const ChatBot = lazy(() => import('./components/ChatBot'))
const LandingPage = lazy(() => import('./landing').then(module => ({ default: module.LandingPage })))
const Dashboard = lazy(() => import('./pages/Dashboard').then(module => ({ default: module.Dashboard })))

// Loading spinner component
function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
    </div>
  )
}

function ProjectCanvas() {
  const { projectId } = useParams<{ projectId: string }>()
  const navigate = useNavigate()
  const { isGuest, user } = useAuth()
  const { setNodes, setEdges, loadProject, loadGroupDefinitions, currentProject, reset } = useModelBuilderStore()
  const [isLoading, setIsLoading] = useState(false)
  const [draggedType, setDraggedType] = useState<string | null>(null)
  const { selectedNodeId } = useModelBuilderStore()
  const addNodeFromPaletteRef = useRef<((blockType: string) => void) | null>(null)

  // Track canvas time
  useCanvasTimeTracking()

  // Load project from URL parameter
  useEffect(() => {
    // Clear canvas if in guest mode
    if (isGuest) {
      reset()
      return
    }

    // If no projectId in URL (blank canvas) and we have a currentProject loaded, clear it
    if (!projectId && user && currentProject) {
      reset()
      return
    }

    // Wait for auth to finish loading before attempting to fetch project
    if (!projectId || !user) {
      return
    }

    // Only load if project ID changed
    if (currentProject && currentProject.id === projectId) {
      return
    }

    if (projectId) {
      setIsLoading(true)
      fetchProject(projectId)
        .then(async (backendProject) => {
          // Load architecture if it exists
          try {
            const { nodes, edges, groupDefinitions } = await loadArchitecture(projectId)
            const project = convertToFrontendProject(backendProject, nodes, edges)
            loadProject(project)

            // Load group definitions if they exist
            if (groupDefinitions && groupDefinitions.length > 0) {
              loadGroupDefinitions(groupDefinitions)
            }
          } catch (error) {
            // No architecture yet, just load project metadata
            const project = convertToFrontendProject(backendProject)
            loadProject(project)
          }
        })
        .catch((error) => {
          console.error('Failed to load project:', error)

          // Clear the invalid project state
          reset()

          // Show error message
          toast.error('Project not found', {
            description: 'This project may have been deleted or you may not have access to it.'
          })

          // Navigate to clean canvas
          navigate('/project', { replace: true })
        })
        .finally(() => {
          setIsLoading(false)
        })
    }
  }, [projectId, currentProject, isGuest, user, setNodes, setEdges, loadProject, loadGroupDefinitions, navigate, reset])

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

      <Suspense fallback={<LoadingSpinner />}>
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
      </Suspense>
      <Toaster position="bottom-right" richColors />
    </div>
  )
}

function App() {
  // Track user session metrics
  useSessionTracking();

  return (
    <Suspense fallback={<LoadingSpinner />}>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />
        <Route path="/project" element={<ProjectCanvas />} />
        <Route path="/project/:projectId" element={<ProjectCanvas />} />
      </Routes>
    </Suspense>
  )
}

export default App
