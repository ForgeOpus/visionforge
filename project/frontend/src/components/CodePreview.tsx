/**
 * CodePreview component
 * Displays rendered code for a node based on its spec and config
 */

import { useEffect, useState } from 'react'
import { renderNodeCode } from '@/lib/api'
import type { Framework } from '@/lib/nodeSpec.types'

interface CodePreviewProps {
  nodeType: string
  framework: Framework
  config: Record<string, any>
  className?: string
}

export function CodePreview({ nodeType, framework, config, className = '' }: CodePreviewProps) {
  const [code, setCode] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchCode() {
      setLoading(true)
      setError(null)

      try {
        const response = await renderNodeCode(nodeType, framework, config)
        
        if (response.success && response.data) {
          setCode(response.data.code)
        } else {
          setError(response.error || 'Failed to render code')
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
      } finally {
        setLoading(false)
      }
    }

    if (nodeType) {
      fetchCode()
    }
  }, [nodeType, framework, config])

  if (loading) {
    return (
      <div className={`rounded-md border border-gray-700 bg-gray-900 p-4 ${className}`}>
        <div className="animate-pulse text-gray-400">Loading code preview...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className={`rounded-md border border-red-700 bg-red-950/50 p-4 ${className}`}>
        <div className="text-red-400">Error: {error}</div>
      </div>
    )
  }

  if (!code) {
    return null
  }

  return (
    <div className={`rounded-md border border-gray-700 bg-gray-900 ${className}`}>
      <div className="border-b border-gray-700 px-4 py-2">
        <h4 className="text-sm font-medium text-gray-300">Code Preview</h4>
      </div>
      <pre className="overflow-x-auto p-4">
        <code className="text-sm text-gray-100">{code}</code>
      </pre>
    </div>
  )
}
