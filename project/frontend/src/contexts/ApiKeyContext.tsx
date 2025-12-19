import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'

interface ApiKeyContextType {
  geminiApiKey: string | null
  anthropicApiKey: string | null
  openaiApiKey: string | null
  isProduction: boolean
  requiresApiKey: boolean
  provider: 'Gemini' | 'Claude' | 'OpenAI' | null
  environment: string | null
  isLoading: boolean
  activeProvider: 'gemini' | 'claude' | 'openai'
  setGeminiApiKey: (key: string | null) => void
  setAnthropicApiKey: (key: string | null) => void
  setOpenAIApiKey: (key: string | null) => void
  setActiveProvider: (provider: 'gemini' | 'claude' | 'openai') => void
  clearKeys: () => void
  hasRequiredKey: () => boolean
}

const ApiKeyContext = createContext<ApiKeyContextType | undefined>(undefined)

const STORAGE_KEY_GEMINI = 'visionforge_gemini_api_key'
const STORAGE_KEY_ANTHROPIC = 'visionforge_anthropic_api_key'
const STORAGE_KEY_OPENAI = 'visionforge_openai_api_key'
const STORAGE_KEY_ACTIVE_PROVIDER = 'visionforge_active_provider'

interface ApiKeyProviderProps {
  children: ReactNode
}

export function ApiKeyProvider({ children }: ApiKeyProviderProps) {
  const [geminiApiKey, setGeminiApiKeyState] = useState<string | null>(null)
  const [anthropicApiKey, setAnthropicApiKeyState] = useState<string | null>(null)
  const [openaiApiKey, setOpenAIApiKeyState] = useState<string | null>(null)
  const [isProduction, setIsProduction] = useState(false)
  const [requiresApiKey, setRequiresApiKey] = useState(false)
  const [provider, setProvider] = useState<'Gemini' | 'Claude' | 'OpenAI' | null>(null)
  const [environment, setEnvironment] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Active provider state (user's choice for which AI provider to use)
  const [activeProvider, setActiveProviderState] = useState<'gemini' | 'claude' | 'openai'>(() => {
    const saved = localStorage.getItem(STORAGE_KEY_ACTIVE_PROVIDER)
    if (saved === 'claude' || saved === 'openai' || saved === 'gemini') {
      return saved as 'gemini' | 'claude' | 'openai'
    }
    return 'gemini' // default
  })

  // Load keys from sessionStorage on mount
  useEffect(() => {
    const savedGeminiKey = sessionStorage.getItem(STORAGE_KEY_GEMINI)
    const savedAnthropicKey = sessionStorage.getItem(STORAGE_KEY_ANTHROPIC)
    const savedOpenAIKey = sessionStorage.getItem(STORAGE_KEY_OPENAI)

    if (savedGeminiKey) setGeminiApiKeyState(savedGeminiKey)
    if (savedAnthropicKey) setAnthropicApiKeyState(savedAnthropicKey)
    if (savedOpenAIKey) setOpenAIApiKeyState(savedOpenAIKey)
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

  const setGeminiApiKey = (key: string | null) => {
    setGeminiApiKeyState(key)
    if (key) {
      sessionStorage.setItem(STORAGE_KEY_GEMINI, key)
    } else {
      sessionStorage.removeItem(STORAGE_KEY_GEMINI)
    }
  }

  const setAnthropicApiKey = (key: string | null) => {
    setAnthropicApiKeyState(key)
    if (key) {
      sessionStorage.setItem(STORAGE_KEY_ANTHROPIC, key)
    } else {
      sessionStorage.removeItem(STORAGE_KEY_ANTHROPIC)
    }
  }

  const setOpenAIApiKey = (key: string | null) => {
    setOpenAIApiKeyState(key)
    if (key) {
      sessionStorage.setItem(STORAGE_KEY_OPENAI, key)
    } else {
      sessionStorage.removeItem(STORAGE_KEY_OPENAI)
    }
  }

  const setActiveProvider = (provider: 'gemini' | 'claude' | 'openai') => {
    setActiveProviderState(provider)
    localStorage.setItem(STORAGE_KEY_ACTIVE_PROVIDER, provider)
  }

  const clearKeys = () => {
    setGeminiApiKey(null)
    setAnthropicApiKey(null)
    setOpenAIApiKey(null)
  }

  const hasRequiredKey = (): boolean => {
    if (!requiresApiKey) return true // DEV mode doesn't need client-side keys

    // Check if user has API key for their active provider
    if (activeProvider === 'gemini') {
      return !!geminiApiKey
    } else if (activeProvider === 'claude') {
      return !!anthropicApiKey
    } else if (activeProvider === 'openai') {
      return !!openaiApiKey
    }

    return false
  }

  return (
    <ApiKeyContext.Provider
      value={{
        geminiApiKey,
        anthropicApiKey,
        openaiApiKey,
        isProduction,
        requiresApiKey,
        provider,
        environment,
        isLoading,
        activeProvider,
        setGeminiApiKey,
        setAnthropicApiKey,
        setOpenAIApiKey,
        setActiveProvider,
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
