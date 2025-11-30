import { createContext, useContext, useState, useEffect, ReactNode } from 'react'

interface ApiKeyContextType {
  apiKey: string | null
  setApiKey: (key: string | null) => void
  hasApiKey: boolean
  clearApiKey: () => void
}

const ApiKeyContext = createContext<ApiKeyContextType | undefined>(undefined)

const API_KEY_STORAGE_KEY = 'visionforge_gemini_api_key'

export function ApiKeyProvider({ children }: { children: ReactNode }) {
  const [apiKey, setApiKeyState] = useState<string | null>(null)

  // Load API key from sessionStorage on mount
  useEffect(() => {
    const storedKey = sessionStorage.getItem(API_KEY_STORAGE_KEY)
    if (storedKey) {
      setApiKeyState(storedKey)
    }
  }, [])

  const setApiKey = (key: string | null) => {
    setApiKeyState(key)
    if (key) {
      sessionStorage.setItem(API_KEY_STORAGE_KEY, key)
    } else {
      sessionStorage.removeItem(API_KEY_STORAGE_KEY)
    }
  }

  const clearApiKey = () => {
    setApiKeyState(null)
    sessionStorage.removeItem(API_KEY_STORAGE_KEY)
  }

  return (
    <ApiKeyContext.Provider
      value={{
        apiKey,
        setApiKey,
        hasApiKey: !!apiKey,
        clearApiKey,
      }}
    >
      {children}
    </ApiKeyContext.Provider>
  )
}

export function useApiKey() {
  const context = useContext(ApiKeyContext)
  if (context === undefined) {
    throw new Error('useApiKey must be used within an ApiKeyProvider')
  }
  return context
}
