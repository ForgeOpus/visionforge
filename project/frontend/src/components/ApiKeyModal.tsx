import { useState, useEffect } from 'react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Info, Eye, EyeSlash } from '@phosphor-icons/react'
import { useApiKeys } from '@/contexts/ApiKeyContext'

interface ApiKeyModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  required?: boolean
}

export default function ApiKeyModal({ open, onOpenChange, required = false }: ApiKeyModalProps) {
  const {
    geminiApiKey,
    anthropicApiKey,
    provider,
    setGeminiApiKey,
    setAnthropicApiKey,
    hasRequiredKey
  } = useApiKeys()

  const [inputKey, setInputKey] = useState('')
  const [showKey, setShowKey] = useState(false)

  // Load existing key when modal opens
  useEffect(() => {
    if (open) {
      if (provider === 'Gemini' && geminiApiKey) {
        setInputKey(geminiApiKey)
      } else if (provider === 'Claude' && anthropicApiKey) {
        setInputKey(anthropicApiKey)
      }
    }
  }, [open, provider, geminiApiKey, anthropicApiKey])

  const handleSave = () => {
    if (!inputKey.trim()) {
      return
    }

    if (provider === 'Gemini') {
      setGeminiApiKey(inputKey.trim())
    } else if (provider === 'Claude') {
      setAnthropicApiKey(inputKey.trim())
    }

    onOpenChange(false)
  }

  const handleSkip = () => {
    if (!required) {
      onOpenChange(false)
    }
  }

  const getProviderInfo = () => {
    if (provider === 'Gemini') {
      return {
        name: 'Gemini',
        url: 'https://aistudio.google.com/app/apikey',
        placeholder: 'AIza...'
      }
    } else if (provider === 'Claude') {
      return {
        name: 'Claude',
        url: 'https://console.anthropic.com/',
        placeholder: 'sk-ant-...'
      }
    }
    return {
      name: 'AI Provider',
      url: '#',
      placeholder: 'Enter API key'
    }
  }

  const providerInfo = getProviderInfo()

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px]" onPointerDownOutside={(e) => {
        if (required && !hasRequiredKey()) {
          e.preventDefault()
        }
      }}>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Info size={24} className="text-primary" />
            {providerInfo.name} API Key Required
          </DialogTitle>
          <DialogDescription>
            This is a remote deployment of VisionForge. To use the AI assistant, please provide your own {providerInfo.name} API key.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <Alert>
            <Info size={16} />
            <AlertDescription>
              Your API key is stored only in your browser's session storage and is never sent to our servers.
              It's only used to communicate directly with {providerInfo.name}'s API.
            </AlertDescription>
          </Alert>

          <div className="space-y-2">
            <Label htmlFor="api-key">{providerInfo.name} API Key</Label>
            <div className="relative">
              <Input
                id="api-key"
                type={showKey ? 'text' : 'password'}
                placeholder={providerInfo.placeholder}
                value={inputKey}
                onChange={(e) => setInputKey(e.target.value)}
                className="pr-10"
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="absolute right-0 top-0 h-full"
                onClick={() => setShowKey(!showKey)}
              >
                {showKey ? <EyeSlash size={18} /> : <Eye size={18} />}
              </Button>
            </div>
            <p className="text-sm text-muted-foreground">
              Don't have an API key?{' '}
              <a
                href={providerInfo.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                Get one from {providerInfo.name}
              </a>
            </p>
          </div>
        </div>

        <DialogFooter className="flex-row justify-between sm:justify-between">
          {!required && (
            <Button
              type="button"
              variant="ghost"
              onClick={handleSkip}
            >
              Skip for now
            </Button>
          )}
          <Button
            type="button"
            onClick={handleSave}
            disabled={!inputKey.trim()}
            className="ml-auto"
          >
            Save API Key
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
