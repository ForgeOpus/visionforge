import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { SecureStorage } from '../lib/encryption'

interface ApiKeyContextType {
  geminiApiKey: string | null
  anthropicApiKey: string | null
  isProduction: boolean
  requiresApiKey: boolean
  provider: 'Gemini' | 'Claude' | null
  environment: string | null
  isLoading: boolean
  setGeminiApiKey: (key: string | null) => void
  setAnthropicApiKey: (key: string | null) => void
  clearKeys: () => void
  hasRequiredKey: () => boolean
}

const ApiKeyContext = createContext<ApiKeyContextType | undefined>(undefined)

const STORAGE_KEY_GEMINI = 'visionforge_gemini_api_key'
const STORAGE_KEY_ANTHROPIC = 'visionforge_anthropic_api_key'

interface ApiKeyProviderProps {
  children: ReactNode
}

export function ApiKeyProvider({ children }: ApiKeyProviderProps) {
  const [geminiApiKey, setGeminiApiKeyState] = useState<string | null>(null)
  const [anthropicApiKey, setAnthropicApiKeyState] = useState<string | null>(null)
  const [isProduction, setIsProduction] = useState(false)
  const [requiresApiKey, setRequiresApiKey] = useState(false)
  const [provider, setProvider] = useState<'Gemini' | 'Claude' | null>(null)
  const [environment, setEnvironment] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Load encrypted keys from sessionStorage on mount
  useEffect(() => {
    const loadKeys = async () => {
      try {
        const savedGeminiKey = await SecureStorage.getItem(STORAGE_KEY_GEMINI)
        const savedAnthropicKey = await SecureStorage.getItem(STORAGE_KEY_ANTHROPIC)

        if (savedGeminiKey) setGeminiApiKeyState(savedGeminiKey)
        if (savedAnthropicKey) setAnthropicApiKeyState(savedAnthropicKey)
      } catch (error) {
        console.error('Failed to load API keys:', error)
      }
    }

    loadKeys()
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

  const setGeminiApiKey = async (key: string | null) => {
    setGeminiApiKeyState(key)
    try {
      if (key) {
        await SecureStorage.setItem(STORAGE_KEY_GEMINI, key)
      } else {
        SecureStorage.removeItem(STORAGE_KEY_GEMINI)
      }
    } catch (error) {
      console.error('Failed to store Gemini API key:', error)
    }
  }

  const setAnthropicApiKey = async (key: string | null) => {
    setAnthropicApiKeyState(key)
    try {
      if (key) {
        await SecureStorage.setItem(STORAGE_KEY_ANTHROPIC, key)
      } else {
        SecureStorage.removeItem(STORAGE_KEY_ANTHROPIC)
      }
    } catch (error) {
      console.error('Failed to store Anthropic API key:', error)
    }
  }

  const clearKeys = () => {
    setGeminiApiKey(null)
    setAnthropicApiKey(null)
  }

  const hasRequiredKey = (): boolean => {
    if (!requiresApiKey) return true // DEV mode doesn't need client-side keys

    if (provider === 'Gemini') {
      return !!geminiApiKey
    } else if (provider === 'Claude') {
      return !!anthropicApiKey
    }

    return false
  }

  return (
    <ApiKeyContext.Provider
      value={{
        geminiApiKey,
        anthropicApiKey,
        isProduction,
        requiresApiKey,
        provider,
        environment,
        isLoading,
        setGeminiApiKey,
        setAnthropicApiKey,
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
