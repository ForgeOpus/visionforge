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
import { Info, Eye, EyeSlash, Check, X } from '@phosphor-icons/react'
import { useApiKeys } from '@/contexts/ApiKeyContext'
import { validateApiKey, getAvailableModelsForKey } from '@/lib/api'
import { toast } from 'sonner'

interface UniversalApiKeyModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  required?: boolean
}

type ModelType =
  // FREE OpenRouter models (VERIFIED WORKING - 404 errors removed)
  | 'llama-3.3-70b'
  | 'llama-3.1-70b'
  | 'llama-3.1-8b'
  | 'gemini-2.0-flash'
  | 'gemini-3-flash'
  | 'gemini-2.5-flash'
  | 'mistral-nemo'
  | 'deepseek-chat-v3'
  | 'deepseek-chat-v3.1'
  | 'deepseek-v3.2'
  | 'nemotron-nano-30b'
  // Gemini models (Paid on OpenRouter, Free on Google AI)
  | 'gemini-3-pro'
  | 'gemini-2.5-pro'
  // OpenAI models (Paid)
  | 'gpt-5.2'
  | 'gpt-4o'
  | 'gpt-4o-mini'
  // Claude models (Paid)
  | 'claude-opus-4.5'
  | 'claude-sonnet-4.5'
  | 'claude-haiku-4.5'
  | 'claude-3.5-sonnet'
  | 'claude-3.5-haiku'
  // Affordable PAID OpenRouter models
  | 'llama-3.1-405b'
  | 'deepseek-v3'
  | 'deepseek-coder-v2'
  | 'qwen-2.5-72b'
  | 'mistral-large-2'

export default function UniversalApiKeyModal({ open, onOpenChange, required = false }: UniversalApiKeyModalProps) {
  const {
    openrouterApiKey,
    selectedModel: contextSelectedModel,
    setOpenRouterApiKey,
    setSelectedModel: setContextSelectedModel,
    clearKeys,
    hasRequiredKey
  } = useApiKeys()

  const [inputKey, setInputKey] = useState('')
  const [showKey, setShowKey] = useState(false)
  const [isValidating, setIsValidating] = useState(false)
  const [validationResult, setValidationResult] = useState<{
    valid: boolean
    provider: string | null
    displayName: string | null
    availableModels: string[]
    isFreeTier: boolean
    message: string
  } | null>(null)
  const [selectedModel, setSelectedModel] = useState<ModelType>((contextSelectedModel as ModelType) || 'llama-3.3-70b')

  // Load existing key when modal opens
  useEffect(() => {
    if (open && openrouterApiKey) {
      setInputKey(openrouterApiKey)
      // Validate existing key
      validateKey(openrouterApiKey)
    } else if (open) {
      setInputKey('')
      setValidationResult(null)
    }
  }, [open, openrouterApiKey])

  // Validate API key and detect provider
  const validateKey = async (key: string) => {
    if (!key || key.length < 10) {
      setValidationResult(null)
      return
    }

    setIsValidating(true)
    try {
      const response = await validateApiKey(key)

      if (response.success && response.data) {
        setValidationResult({
          valid: response.data.valid,
          provider: response.data.provider,
          displayName: response.data.displayName,
          availableModels: response.data.models,
          isFreeTier: response.data.isFreeTier,
          message: response.data.message
        })

        // If a model is selected that's not available with this key, switch to first available
        if (response.data.valid && response.data.models.length > 0) {
          if (!response.data.models.includes(selectedModel)) {
            setSelectedModel(response.data.models[0] as ModelType)
            setContextSelectedModel(response.data.models[0])
          }
        }
      } else {
        setValidationResult({
          valid: false,
          provider: null,
          displayName: null,
          availableModels: [],
          isFreeTier: false,
          message: 'Unknown API key format'
        })
      }
    } catch (error) {
      console.error('Validation error:', error)
      setValidationResult({
        valid: false,
        provider: null,
        displayName: null,
        availableModels: [],
        isFreeTier: false,
        message: 'Error validating API key'
      })
    } finally {
      setIsValidating(false)
    }
  }

  // Handle key input change with debounced validation
  const handleKeyChange = (value: string) => {
    setInputKey(value)

    // Clear validation if key is too short
    if (value.length < 10) {
      setValidationResult(null)
      return
    }

    // Debounce validation
    const timeoutId = setTimeout(() => {
      validateKey(value)
    }, 500)

    return () => clearTimeout(timeoutId)
  }

  const handleSave = () => {
    if (!inputKey.trim()) {
      toast.error('Please enter an API key')
      return
    }

    if (validationResult && !validationResult.valid) {
      toast.error('Invalid API key format')
      return
    }

    const trimmedKey = inputKey.trim()
    setOpenRouterApiKey(trimmedKey)

    toast.success(validationResult?.displayName
      ? `${validationResult.displayName} API key saved!`
      : 'API key saved!')

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
    setValidationResult(null)
    onOpenChange(false)
    toast.success('API keys cleared')
  }

  // Get all models organized by provider and pricing
  const allModels = {
    // Truly FREE models (VERIFIED WORKING - 404 errors removed)
    free: [
      { id: 'llama-3.3-70b', label: 'Llama 3.3 70B', desc: '70B capable - FREE', free: true },
      { id: 'llama-3.1-70b', label: 'Llama 3.1 70B', desc: '70B instruct - FREE', free: true },
      { id: 'llama-3.1-8b', label: 'Llama 3.1 8B', desc: '8B efficient - FREE', free: true },
      { id: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash', desc: 'Fast Gemini - FREE', free: true },
      { id: 'gemini-3-flash', label: 'Gemini 3 Flash', desc: 'Latest Gemini - FREE', free: true },
      { id: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash', desc: 'Gemini 2.5 - FREE', free: true },
      { id: 'mistral-nemo', label: 'Mistral Nemo', desc: 'Mistral model - FREE', free: true },
      { id: 'deepseek-chat-v3', label: 'DeepSeek Chat V3', desc: 'Chat optimized - FREE', free: true },
      { id: 'deepseek-chat-v3.1', label: 'DeepSeek Chat V3.1', desc: 'Latest chat - FREE', free: true },
      { id: 'deepseek-v3.2', label: 'DeepSeek V3.2', desc: 'DeepSeek flagship - FREE', free: true },
      { id: 'nemotron-nano-30b', label: 'Nemotron Nano 30B', desc: 'NVIDIA 30B - FREE', free: true },
    ],
    // Gemini models (PAID on OpenRouter, FREE on direct Google AI)
    gemini: [
      { id: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro', desc: 'Advanced thinking ($1.25/M)', free: false },
      { id: 'gemini-3-pro', label: 'Gemini 3 Pro', desc: 'Best multimodal ($2/M)', free: false },
    ],
    // OpenAI models (PAID)
    openai: [
      { id: 'gpt-5.2', label: 'GPT-5.2', desc: 'Newest flagship (Dec 2025)', free: false },
      { id: 'gpt-4o', label: 'GPT-4o', desc: 'Multimodal omni model ($2.50/M)', free: false },
      { id: 'gpt-4o-mini', label: 'GPT-4o Mini', desc: 'Fast and affordable ($0.15/M)', free: false },
    ],
    // Claude flagship models (PAID)
    claude: [
      { id: 'claude-3.5-haiku', label: 'Claude 3.5 Haiku', desc: 'Most affordable Claude ($0.80/M)', free: false },
      { id: 'claude-3.5-sonnet', label: 'Claude 3.5 Sonnet', desc: 'Excellent value ($3/M)', free: false },
      { id: 'claude-haiku-4.5', label: 'Claude Haiku 4.5', desc: 'Fast, near-frontier quality', free: false },
      { id: 'claude-sonnet-4.5', label: 'Claude Sonnet 4.5', desc: 'Balanced performance ($3/M)', free: false },
      { id: 'claude-opus-4.5', label: 'Claude Opus 4.5', desc: 'Best for coding (Nov 2025)', free: false },
    ],
    // Affordable OpenRouter models (PAID but cheap)
    affordable: [
      { id: 'deepseek-v3', label: 'DeepSeek V3', desc: 'Latest flagship ($0.27/M)', free: false },
      { id: 'deepseek-coder-v2', label: 'DeepSeek Coder V2', desc: 'Coding optimized ($0.27/M)', free: false },
      { id: 'mistral-large-2', label: 'Mistral Large 2', desc: 'Mistral flagship ($2/M)', free: false },
      { id: 'llama-3.1-405b', label: 'Llama 3.1 405B', desc: 'Powerful 405B ($2.70/M)', free: false },
      { id: 'qwen-2.5-72b', label: 'Qwen 2.5 72B', desc: 'Chinese model - affordable', free: false },
    ]
  }

  // Filter models based on detected provider
  const getAvailableModelsForUI = () => {
    if (!validationResult || !validationResult.valid) {
      // Show all models when no key or invalid key - FREE models first, then PAID
      return [
        // TRULY FREE MODELS (11 total - VERIFIED WORKING, 404 errors removed)
        ...allModels.free,        // Free (11 models)
        // AFFORDABLE PAID MODELS (5 total)
        ...allModels.affordable,  // Affordable (5 models)
        // PAID GEMINI MODELS (2 total - free on Google AI)
        ...allModels.gemini,      // Paid on OpenRouter (2 models)
        // FLAGSHIP PAID MODELS (8 total)
        ...allModels.openai,      // Paid (3 models)
        ...allModels.claude,      // Paid (5 models)
      ]
    }

    // Show only models available with this provider
    const provider = validationResult.provider
    if (provider === 'openrouter') {
      // OpenRouter has access to ALL models - FREE models first, then PAID
      return [
        // TRULY FREE MODELS (11 total - VERIFIED WORKING, 404 errors removed)
        ...allModels.free,        // Free (11 models)
        // AFFORDABLE PAID MODELS (5 total)
        ...allModels.affordable,  // Affordable (5 models)
        // PAID GEMINI MODELS (2 total)
        ...allModels.gemini,      // Paid on OpenRouter (2 models)
        // FLAGSHIP PAID MODELS (8 total)
        ...allModels.openai,      // Paid (3 models)
        ...allModels.claude,      // Paid (5 models)
      ]
    } else if (provider === 'google') {
      return allModels.gemini
    } else if (provider === 'openai') {
      return allModels.openai
    } else if (provider === 'anthropic') {
      return allModels.claude
    }

    return []
  }

  const availableModels = getAvailableModelsForUI()

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
            API Key Setup
          </DialogTitle>
          <DialogDescription>
            Enter an API key from any supported provider to use the AI assistant.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <Alert>
            <Info size={16} />
            <AlertDescription>
              <div className="mb-1">You can use a key you already have! Supported:</div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-2">
                <div><strong>OpenRouter</strong> (all models)</div>
                <div><strong>Google AI</strong> (Gemini)</div>
                <div><strong>OpenAI</strong> (GPT)</div>
                <div><strong>Anthropic</strong> (Claude)</div>
              </div>
            </AlertDescription>
          </Alert>

          <div className="space-y-2">
            <Label htmlFor="api-key-input">Enter your API key</Label>
            <div className="relative">
              <Input
                id="api-key-input"
                type={showKey ? 'text' : 'password'}
                placeholder="Paste any API key (sk-or-v1-..., AIza..., sk-proj-..., sk-ant-...)"
                value={inputKey}
                onChange={(e) => handleKeyChange(e.target.value)}
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

            {/* Validation Status */}
            {isValidating && (
              <div className="text-sm text-muted-foreground flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-primary border-t-transparent" />
                Detecting provider...
              </div>
            )}

            {validationResult && !isValidating && (
              <div className={`text-sm flex items-center gap-2 ${validationResult.valid ? 'text-green-600' : 'text-red-600'}`}>
                {validationResult.valid ? <Check size={16} weight="bold" /> : <X size={16} weight="bold" />}
                {validationResult.message}
                {validationResult.valid && validationResult.isFreeTier && (
                  <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded ml-2">Free Tier</span>
                )}
              </div>
            )}

            <p className="text-sm text-muted-foreground">
              Don't have an API key?{' '}
              <a
                href="https://openrouter.ai/keys"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                Get OpenRouter (free, all models)
              </a>
              {' or '}
              <a
                href="https://aistudio.google.com/app/apikey"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                Google AI (free, Gemini only)
              </a>
            </p>
          </div>

          {validationResult && validationResult.valid && (
            <div className="space-y-2">
              <Label htmlFor="model-select">
                Select Model
                {validationResult.availableModels.length > 0 && (
                  <span className="text-sm text-muted-foreground ml-2">
                    ({validationResult.availableModels.length} available)
                  </span>
                )}
              </Label>
              <Select
                value={selectedModel}
                onValueChange={(v) => {
                  setSelectedModel(v as ModelType)
                  setContextSelectedModel(v)
                }}
              >
                <SelectTrigger id="model-select">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="max-h-[400px] w-[560px]">
                  <div className="grid grid-cols-2 gap-2 p-2">
                    {availableModels.map((model) => (
                      <SelectItem
                        key={model.id}
                        value={model.id}
                        disabled={validationResult.availableModels.length > 0 && !validationResult.availableModels.includes(model.id)}
                        className="col-span-1 h-auto"
                      >
                        <div className="flex flex-col items-start w-full pr-12">
                          <div className="flex items-center gap-2 w-full">
                            <span className="font-medium text-sm">{model.label}</span>
                            {model.free && (
                              <span className="text-xs bg-green-100 text-green-700 px-1.5 py-0.5 rounded shrink-0">Free</span>
                            )}
                          </div>
                          <span className="text-xs text-muted-foreground mt-0.5">{model.desc}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </div>
                </SelectContent>
              </Select>
            </div>
          )}
        </div>

        <Alert>
          <Info size={16} />
          <AlertDescription className="text-xs">
            Your API key is stored only in your browser's session storage and is never sent to our servers.
            {validationResult?.provider === 'openrouter'
              ? ' OpenRouter routes your requests to the selected AI provider.'
              : validationResult?.displayName
                ? ` Requests go directly to ${validationResult.displayName}.`
                : ' Requests go directly to the detected provider.'}
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
                Clear Key
              </Button>
            )}
          </div>
          <Button
            type="button"
            onClick={handleSave}
            disabled={!inputKey.trim() || (validationResult !== null && !validationResult.valid)}
            className="ml-auto"
          >
            Save & Continue
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
