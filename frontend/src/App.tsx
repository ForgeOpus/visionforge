/**
 * VisionForge - Local Desktop Version
 *
 * Main application component that uses shared core components
 * and local inference client
 */

import { useState, useEffect } from 'react'
import { ErrorBoundary } from 'react-error-boundary'
import { Canvas, BlockPalette, ChatBot } from '@visionforge/core/components'
import { localClient } from './lib/inference'

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
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <div className="h-screen w-screen">
        {/* Status Bar */}
        <div className="flex items-center justify-between border-b bg-white px-4 py-2">
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-bold">VisionForge</h1>
            <span className="text-xs text-gray-500">Local Desktop</span>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-green-500"></div>
              <span className="text-xs text-gray-600">Server Connected</span>
            </div>
            {serverStatus.aiEnabled ? (
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-600">🤖 AI Enabled</span>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400">
                  AI Disabled (add keys to .env)
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Main Application */}
        <div className="flex h-[calc(100vh-48px)]">
          {/* Block Palette */}
          <div className="w-64 border-r bg-white">
            <BlockPalette />
          </div>

          {/* Canvas */}
          <div className="flex-1">
            <Canvas inferenceClient={localClient} />
          </div>

          {/* ChatBot */}
          {serverStatus.aiEnabled && (
            <div className="w-96 border-l bg-white">
              <ChatBot inferenceClient={localClient} />
            </div>
          )}
        </div>
      </div>
    </ErrorBoundary>
  )
}

export default App
