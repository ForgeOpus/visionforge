/**
 * React hook for fetching and managing node specifications from backend
 */

import { useState, useEffect, useCallback } from 'react'
import { getNodeDefinitions, getNodeDefinition, renderNodeCode } from './api'
import type { NodeSpec, Framework } from './nodeSpec.types'

interface UseNodeSpecsOptions {
  framework?: Framework
  autoFetch?: boolean
}

interface UseNodeSpecsReturn {
  specs: NodeSpec[]
  loading: boolean
  error: string | null
  refetch: () => Promise<void>
  getSpec: (nodeType: string) => NodeSpec | undefined
  renderCode: (nodeType: string, config: Record<string, any>) => Promise<string | null>
}

/**
 * Hook to fetch and manage all node specifications for a framework
 */
export function useNodeSpecs(options: UseNodeSpecsOptions = {}): UseNodeSpecsReturn {
  const { framework = 'pytorch', autoFetch = true } = options
  
  const [specs, setSpecs] = useState<NodeSpec[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchSpecs = useCallback(async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await getNodeDefinitions(framework)
      
      if (response.success && response.data) {
        setSpecs(response.data.definitions || [])
      } else {
        setError(response.error || 'Failed to fetch node definitions')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [framework])

  useEffect(() => {
    if (autoFetch) {
      fetchSpecs()
    }
  }, [autoFetch, fetchSpecs])

  const getSpec = useCallback((nodeType: string): NodeSpec | undefined => {
    return specs.find(spec => spec.type === nodeType)
  }, [specs])

  const renderCode = useCallback(async (
    nodeType: string, 
    config: Record<string, any>
  ): Promise<string | null> => {
    try {
      const response = await renderNodeCode(nodeType, framework, config)
      
      if (response.success && response.data) {
        return response.data.code
      }
      
      console.error('Failed to render code:', response.error)
      return null
    } catch (err) {
      console.error('Error rendering code:', err)
      return null
    }
  }, [framework])

  return {
    specs,
    loading,
    error,
    refetch: fetchSpecs,
    getSpec,
    renderCode,
  }
}

/**
 * Hook to fetch a single node specification
 */
export function useNodeSpec(nodeType: string, framework: Framework = 'pytorch') {
  const [spec, setSpec] = useState<NodeSpec | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchSpec() {
      setLoading(true)
      setError(null)
      
      try {
        const response = await getNodeDefinition(nodeType, framework)
        
        if (response.success && response.data) {
          setSpec(response.data.definition)
        } else {
          setError(response.error || 'Failed to fetch node definition')
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
      } finally {
        setLoading(false)
      }
    }

    if (nodeType) {
      fetchSpec()
    }
  }, [nodeType, framework])

  return { spec, loading, error }
}
