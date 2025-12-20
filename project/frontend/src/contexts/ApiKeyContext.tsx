import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'

interface ApiKeyContextType {
  openrouterApiKey: string | null
  isProduction: boolean
  requiresApiKey: boolean
  provider: 'OpenRouter' | null
  environment: string | null
  isLoading: boolean
  selectedModel: string | null
  setOpenRouterApiKey: (key: string | null) => void
  setSelectedModel: (model: string) => void
  clearKeys: () => void
  hasRequiredKey: () => boolean
}

const ApiKeyContext = createContext<ApiKeyContextType | undefined>(undefined)

const STORAGE_KEY_OPENROUTER = 'visionforge_openrouter_api_key'
const STORAGE_KEY_SELECTED_MODEL = 'visionforge_selected_model'

interface ApiKeyProviderProps {
  children: ReactNode
}

export function ApiKeyProvider({ children }: ApiKeyProviderProps) {
  const [openrouterApiKey, setOpenRouterApiKeyState] = useState<string | null>(null)
  const [isProduction, setIsProduction] = useState(false)
  const [requiresApiKey, setRequiresApiKey] = useState(false)
  const [provider, setProvider] = useState<'OpenRouter' | null>(null)
  const [environment, setEnvironment] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Selected model state (user's choice for which model to use)
  const [selectedModel, setSelectedModelState] = useState<string | null>(() => {
    const saved = localStorage.getItem(STORAGE_KEY_SELECTED_MODEL)
    return saved || 'gemini-3-flash' // default to latest free tier Gemini model
  })

  // Load OpenRouter API key from sessionStorage on mount
  useEffect(() => {
    const savedOpenRouterKey = sessionStorage.getItem(STORAGE_KEY_OPENROUTER)
    if (savedOpenRouterKey) setOpenRouterApiKeyState(savedOpenRouterKey)
  }, [])

  // Fetch environment info from backend
  useEffect(() => {
    const fetchEnvironmentInfo = async () => {
      try {
        const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'}/environment`)
        if (response.ok) {
          const data = await response.json()
          setIsProduction(data.isProduction)
          setRequiresApiKey(data.requiresApiKey)
          setProvider(data.provider)
          setEnvironment(data.environment)
        }
      } catch (error) {
        console.error('Failed to fetch environment info:', error)
        // Default to requiring API key if we can't fetch config
        setRequiresApiKey(true)
      } finally {
        setIsLoading(false)
      }
    }

    fetchEnvironmentInfo()
  }, [])

  const setOpenRouterApiKey = (key: string | null) => {
    setOpenRouterApiKeyState(key)
    if (key) {
      sessionStorage.setItem(STORAGE_KEY_OPENROUTER, key)
    } else {
      sessionStorage.removeItem(STORAGE_KEY_OPENROUTER)
    }
  }

  const setSelectedModel = (model: string) => {
    setSelectedModelState(model)
    localStorage.setItem(STORAGE_KEY_SELECTED_MODEL, model)
  }

  const clearKeys = () => {
    setOpenRouterApiKey(null)
  }

  const hasRequiredKey = (): boolean => {
    if (!requiresApiKey) return true // DEV mode doesn't need client-side keys
    return !!openrouterApiKey
  }

  return (
    <ApiKeyContext.Provider
      value={{
        openrouterApiKey,
        isProduction,
        requiresApiKey,
        provider,
        environment,
        isLoading,
        selectedModel,
        setOpenRouterApiKey,
        setSelectedModel,
        clearKeys,
        hasRequiredKey
      }}
    >
      {children}
    </ApiKeyContext.Provider>
  )
}

export function useApiKeys() {
  const context = useContext(ApiKeyContext)
  if (context === undefined) {
    throw new Error('useApiKeys must be used within an ApiKeyProvider')
  }
  return context
}
