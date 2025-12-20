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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Info, Eye, EyeSlash } from '@phosphor-icons/react'
import { useApiKeys } from '@/contexts/ApiKeyContext'

interface ApiKeyModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  required?: boolean
}

type ModelType =
  // Gemini models (Free tier available)
  | 'gemini-3-flash'
  | 'gemini-3-pro'
  | 'gemini-2.5-flash'
  | 'gemini-2.5-pro'
  // OpenAI models
  | 'gpt-5.2'
  | 'gpt-4o'
  | 'gpt-4o-mini'
  // Claude models
  | 'claude-opus-4.5'
  | 'claude-sonnet-4.5'
  | 'claude-haiku-4.5'

type ProviderType = 'gemini' | 'claude' | 'openai'

// Map models to their providers
const modelToProvider: Record<ModelType, ProviderType> = {
  'gemini-3-flash': 'gemini',
  'gemini-3-pro': 'gemini',
  'gemini-2.5-flash': 'gemini',
  'gemini-2.5-pro': 'gemini',
  'gpt-5.2': 'openai',
  'gpt-4o': 'openai',
  'gpt-4o-mini': 'openai',
  'claude-opus-4.5': 'claude',
  'claude-sonnet-4.5': 'claude',
  'claude-haiku-4.5': 'claude',
}

export default function ApiKeyModal({ open, onOpenChange, required = false }: ApiKeyModalProps) {
  const {
    openrouterApiKey,
    selectedModel: contextSelectedModel,
    setOpenRouterApiKey,
    setSelectedModel: setContextSelectedModel,
    clearKeys,
    hasRequiredKey
  } = useApiKeys()

  const [selectedModel, setSelectedModel] = useState<ModelType>((contextSelectedModel as ModelType) || 'gemini-3-flash')
  const [inputKey, setInputKey] = useState('')
  const [showKey, setShowKey] = useState(false)

  // Update context when model changes
  const handleModelChange = (model: ModelType) => {
    setSelectedModel(model)
    setContextSelectedModel(model)
  }

  // Load existing OpenRouter key when modal opens
  useEffect(() => {
    if (open && openrouterApiKey) {
      setInputKey(openrouterApiKey)
    } else if (open) {
      setInputKey('')
    }
  }, [open, openrouterApiKey])

  const handleSave = () => {
    if (!inputKey.trim()) {
      return
    }

    const trimmedKey = inputKey.trim()

    // Save the OpenRouter API key
    setOpenRouterApiKey(trimmedKey)

    onOpenChange(false)
  }

  const handleSkip = () => {
    if (!required) {
      onOpenChange(false)
    }
  }

  const handleClearKeys = () => {
    clearKeys()
    setInputKey('')
    onOpenChange(false)
  }

  const providerInfo = {
    name: 'OpenRouter',
    displayName: 'OpenRouter',
    url: 'https://openrouter.ai/keys',
    placeholder: 'Enter OpenRouter API key',
    description: 'Unified access to all AI models'
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[550px]" onPointerDownOutside={(e) => {
        if (required && !hasRequiredKey()) {
          e.preventDefault()
        }
      }}>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Info size={24} className="text-primary" />
            API Key Configuration
          </DialogTitle>
          <DialogDescription>
            Choose your AI provider and enter your API key to enable AI assistant features.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <Alert>
            <Info size={16} />
            <AlertDescription>
              Choose from the latest AI models. Gemini models offer a free tier for development, while OpenAI and Claude models are pay-as-you-go.
            </AlertDescription>
          </Alert>

          <div className="space-y-2">
            <Label htmlFor="model-select">Select AI Model</Label>
            <Select
              value={selectedModel}
              onValueChange={(v) => handleModelChange(v as ModelType)}
            >
              <SelectTrigger id="model-select">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="max-h-[300px]">
                {/* Free Tier Models (Gemini) */}
                <SelectItem value="gemini-3-flash">
                  <div className="flex items-center justify-between w-full gap-2">
                    <div className="flex flex-col items-start">
                      <span className="font-medium">Gemini 3 Flash</span>
                      <span className="text-xs text-muted-foreground">Newest - frontier intelligence for speed (Dec 2025)</span>
                    </div>
                    <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded">Free Tier</span>
                  </div>
                </SelectItem>
                <SelectItem value="gemini-3-pro">
                  <div className="flex items-center justify-between w-full gap-2">
                    <div className="flex flex-col items-start">
                      <span className="font-medium">Gemini 3 Pro</span>
                      <span className="text-xs text-muted-foreground">Best multimodal understanding (Nov 2025)</span>
                    </div>
                    <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded">Free Tier</span>
                  </div>
                </SelectItem>
                <SelectItem value="gemini-2.5-flash">
                  <div className="flex items-center justify-between w-full gap-2">
                    <div className="flex flex-col items-start">
                      <span className="font-medium">Gemini 2.5 Flash</span>
                      <span className="text-xs text-muted-foreground">Stable - intelligent speed, production-ready</span>
                    </div>
                    <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded">Free Tier</span>
                  </div>
                </SelectItem>
                <SelectItem value="gemini-2.5-pro">
                  <div className="flex items-center justify-between w-full gap-2">
                    <div className="flex flex-col items-start">
                      <span className="font-medium">Gemini 2.5 Pro</span>
                      <span className="text-xs text-muted-foreground">Stable - state-of-the-art thinking model</span>
                    </div>
                    <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded">Free Tier</span>
                  </div>
                </SelectItem>

                {/* OpenAI Models */}
                <SelectItem value="gpt-5.2">
                  <div className="flex flex-col items-start">
                    <span className="font-medium">GPT-5.2</span>
                    <span className="text-xs text-muted-foreground">Newest flagship - professional work (Dec 2025)</span>
                  </div>
                </SelectItem>
                <SelectItem value="gpt-4o">
                  <div className="flex flex-col items-start">
                    <span className="font-medium">GPT-4o</span>
                    <span className="text-xs text-muted-foreground">Multimodal omni model</span>
                  </div>
                </SelectItem>
                <SelectItem value="gpt-4o-mini">
                  <div className="flex flex-col items-start">
                    <span className="font-medium">GPT-4o Mini</span>
                    <span className="text-xs text-muted-foreground">Fast and cost-effective</span>
                  </div>
                </SelectItem>

                {/* Claude Models */}
                <SelectItem value="claude-opus-4.5">
                  <div className="flex flex-col items-start">
                    <span className="font-medium">Claude Opus 4.5</span>
                    <span className="text-xs text-muted-foreground">Newest - best for coding & agents (Nov 2025)</span>
                  </div>
                </SelectItem>
                <SelectItem value="claude-sonnet-4.5">
                  <div className="flex flex-col items-start">
                    <span className="font-medium">Claude Sonnet 4.5</span>
                    <span className="text-xs text-muted-foreground">Balanced performance and speed</span>
                  </div>
                </SelectItem>
                <SelectItem value="claude-haiku-4.5">
                  <div className="flex flex-col items-start">
                    <span className="font-medium">Claude Haiku 4.5</span>
                    <span className="text-xs text-muted-foreground">Fast, cost-effective, near-frontier quality</span>
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="api-key-input">{providerInfo.name} API Key</Label>
            <div className="relative">
              <Input
                id="api-key-input"
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
              Don't have an {providerInfo.name} API key?{' '}
              <a
                href={providerInfo.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                Get one from {providerInfo.name} (Free tier: 50 requests/day)
              </a>
            </p>
          </div>
        </div>

        <Alert>
          <Info size={16} />
          <AlertDescription className="text-xs">
            Your API key is stored only in your browser's session storage and is never sent to our servers.
            OpenRouter routes your requests to the selected AI provider.
          </AlertDescription>
        </Alert>

        <DialogFooter className="flex-row justify-between sm:justify-between gap-2">
          <div className="flex gap-2">
            {!required && !hasRequiredKey() && (
              <Button
                type="button"
                variant="ghost"
                onClick={handleSkip}
              >
                Skip for now
              </Button>
            )}
            {hasRequiredKey() && (
              <Button
                type="button"
                variant="destructive"
                onClick={handleClearKeys}
              >
                Clear All Keys
              </Button>
            )}
          </div>
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
