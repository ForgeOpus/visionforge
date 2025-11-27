/**
 * VisionForge - Local Desktop Version
 *
 * Main application component that uses shared core components
 * and local inference client
 */

import { useState, useEffect, useRef } from 'react'
import { ErrorBoundary } from 'react-error-boundary'
import { Toaster } from 'sonner'
import Canvas from './components/Canvas'
import ResizableBlockPalette from './components/ResizableBlockPalette'
import ConfigPanel from './components/ConfigPanel'
import ChatBot from './components/ChatBot'
import Header from './components/Header'
import { localClient } from './lib/inference'
import { useModelBuilderStore } from '@visionforge/core/store'
import { ApiKeyProvider } from './lib/apiKeyContext'

function ErrorFallback({ error, resetErrorBoundary }: any) {
  return (
    <div className="flex h-screen items-center justify-center bg-gray-50">
      <div className="max-w-md rounded-lg bg-white p-8 shadow-lg">
        <h2 className="mb-4 text-2xl font-bold text-red-600">Something went wrong</h2>
        <pre className="mb-4 overflow-auto rounded bg-gray-100 p-4 text-sm">
          {error.message}
        </pre>
        <button
          onClick={resetErrorBoundary}
          className="rounded bg-blue-500 px-4 py-2 text-white hover:bg-blue-600"
        >
          Try again
        </button>
      </div>
    </div>
  )
}

function App() {
  const [serverStatus, setServerStatus] = useState<{
    connected: boolean
    aiEnabled: boolean
    loading: boolean
    error?: string
  }>({
    connected: false,
    aiEnabled: false,
    loading: true,
  })

  const { selectedNodeId } = useModelBuilderStore()
  const addNodeFromPaletteRef = useRef<((blockType: string) => void) | null>(null)

  // Check server health on mount
  useEffect(() => {
    const checkServer = async () => {
      try {
        const health = await localClient.healthCheck()
        setServerStatus({
          connected: true,
          aiEnabled: health.ai_enabled,
          loading: false,
        })
      } catch (error) {
        setServerStatus({
          connected: false,
          aiEnabled: false,
          loading: false,
          error: error instanceof Error ? error.message : 'Server not available',
        })
      }
    }

    checkServer()
  }, [])

  const handleDragStart = (_type: string) => {
    // Drag type handling for future use
  }

  const handleBlockClick = (blockType: string) => {
    if (addNodeFromPaletteRef.current) {
      addNodeFromPaletteRef.current(blockType)
    }
  }

  const registerAddNodeHandler = (handler: (blockType: string) => void) => {
    addNodeFromPaletteRef.current = handler
  }

  if (serverStatus.loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="mb-4 h-12 w-12 animate-spin rounded-full border-4 border-blue-500 border-t-transparent"></div>
          <p className="text-gray-600">Connecting to VisionForge server...</p>
        </div>
      </div>
    )
  }

  if (!serverStatus.connected) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50">
        <div className="max-w-md rounded-lg bg-white p-8 shadow-lg">
          <h2 className="mb-4 text-2xl font-bold text-red-600">Server Not Running</h2>
          <p className="mb-4 text-gray-600">
            Cannot connect to VisionForge server. Make sure it's running:
          </p>
          <pre className="mb-4 overflow-auto rounded bg-gray-100 p-4 text-sm">
            vision-forge start
          </pre>
          <p className="text-sm text-gray-500">
            Error: {serverStatus.error}
          </p>
        </div>
      </div>
    )
  }

  return (
    <ApiKeyProvider>
      <ErrorBoundary FallbackComponent={ErrorFallback}>
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
      </ErrorBoundary>
    </ApiKeyProvider>
  )
}

export default App
