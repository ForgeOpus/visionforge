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
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Info, Eye, EyeSlash, Sparkle, Crown } from '@phosphor-icons/react'
import { useApiKeys } from '@/contexts/ApiKeyContext'
import { useAuth } from '@/contexts/AuthContext'

interface ApiKeyModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  required?: boolean
}

type ProviderType = 'gemini' | 'claude' | 'openai'

export default function ApiKeyModal({ open, onOpenChange, required = false }: ApiKeyModalProps) {
  const {
    geminiApiKey,
    anthropicApiKey,
    openaiApiKey,
    activeProvider,
    setGeminiApiKey,
    setAnthropicApiKey,
    setOpenAIApiKey,
    setActiveProvider,
    clearKeys,
    hasRequiredKey
  } = useApiKeys()

  const { user } = useAuth()
  const isPro = user?.tier === 'pro'

  const [tier, setTier] = useState<'free' | 'pro'>('free')
  const [selectedProvider, setSelectedProvider] = useState<ProviderType>('gemini')
  const [inputKey, setInputKey] = useState('')
  const [showKey, setShowKey] = useState(false)

  // Load existing key when modal opens or provider changes
  useEffect(() => {
    if (open) {
      // Set tier based on user or default to free
      setTier(isPro ? 'pro' : 'free')

      // Load the active provider's key
      if (selectedProvider === 'gemini' && geminiApiKey) {
        setInputKey(geminiApiKey)
      } else if (selectedProvider === 'claude' && anthropicApiKey) {
        setInputKey(anthropicApiKey)
      } else if (selectedProvider === 'openai' && openaiApiKey) {
        setInputKey(openaiApiKey)
      } else {
        setInputKey('')
      }
    }
  }, [open, selectedProvider, geminiApiKey, anthropicApiKey, openaiApiKey, isPro])

  // Set selected provider based on active provider
  useEffect(() => {
    if (open) {
      setSelectedProvider(activeProvider)
    }
  }, [open, activeProvider])

  const handleSave = () => {
    if (!inputKey.trim()) {
      return
    }

    const trimmedKey = inputKey.trim()

    // Save the API key for the selected provider
    if (selectedProvider === 'gemini') {
      setGeminiApiKey(trimmedKey)
    } else if (selectedProvider === 'claude') {
      setAnthropicApiKey(trimmedKey)
    } else if (selectedProvider === 'openai') {
      setOpenAIApiKey(trimmedKey)
    }

    // Set active provider
    setActiveProvider(selectedProvider)

    onOpenChange(false)
  }

  const handleSkip = () => {
    if (!required) {
      onOpenChange(false)
    }
  }

  const getProviderInfo = (provider: ProviderType) => {
    switch (provider) {
      case 'gemini':
        return {
          name: 'Gemini',
          displayName: 'Google Gemini',
          url: 'https://aistudio.google.com/app/apikey',
          placeholder: 'AIza...',
          description: 'Fast, free tier available'
        }
      case 'openai':
        return {
          name: 'OpenAI',
          displayName: 'OpenAI (GPT-4, GPT-3.5)',
          url: 'https://platform.openai.com/api-keys',
          placeholder: 'sk-proj-...',
          description: 'Industry standard, most popular'
        }
      case 'claude':
        return {
          name: 'Claude',
          displayName: 'Anthropic Claude',
          url: 'https://console.anthropic.com/',
          placeholder: 'sk-ant-...',
          description: 'Advanced reasoning, latest models'
        }
    }
  }

  const providerInfo = getProviderInfo(selectedProvider)

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

        <Tabs value={tier} onValueChange={(v) => setTier(v as 'free' | 'pro')} className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="free" className="flex items-center gap-2">
              <Sparkle size={16} />
              Free Tier
            </TabsTrigger>
            <TabsTrigger value="pro" className="flex items-center gap-2" disabled={!isPro}>
              <Crown size={16} />
              Pro Tier
              {!isPro && <span className="text-xs">(Upgrade Required)</span>}
            </TabsTrigger>
          </TabsList>

          {/* Free Tier Tab */}
          <TabsContent value="free" className="space-y-4 mt-4">
            <Alert>
              <Info size={16} />
              <AlertDescription>
                <strong>Free Tier:</strong> Uses Google Gemini with your personal API key.
                Gemini offers a generous free tier perfect for getting started.
              </AlertDescription>
            </Alert>

            <div className="space-y-2">
              <Label htmlFor="free-api-key" className="flex items-center gap-2">
                Gemini API Key
                <span className="text-xs text-muted-foreground">(Free tier available)</span>
              </Label>
              <div className="relative">
                <Input
                  id="free-api-key"
                  type={showKey ? 'text' : 'password'}
                  placeholder="AIza..."
                  value={tier === 'free' && selectedProvider === 'gemini' ? inputKey : geminiApiKey || ''}
                  onChange={(e) => {
                    if (tier === 'free') {
                      setSelectedProvider('gemini')
                      setInputKey(e.target.value)
                    }
                  }}
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
                  href="https://aistudio.google.com/app/apikey"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  Get one from Google AI Studio (Free)
                </a>
              </p>
            </div>
          </TabsContent>

          {/* Pro Tier Tab */}
          <TabsContent value="pro" className="space-y-4 mt-4">
            {isPro ? (
              <>
                <Alert>
                  <Crown size={16} className="text-yellow-500" />
                  <AlertDescription>
                    <strong>Pro Tier:</strong> Choose from multiple AI providers including OpenAI (GPT-4),
                    Anthropic (Claude), and Google (Gemini Pro models).
                  </AlertDescription>
                </Alert>

                <div className="space-y-2">
                  <Label htmlFor="provider-select">Select AI Provider</Label>
                  <Select
                    value={selectedProvider}
                    onValueChange={(v) => setSelectedProvider(v as ProviderType)}
                  >
                    <SelectTrigger id="provider-select">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="openai">
                        <div className="flex flex-col items-start">
                          <span className="font-medium">OpenAI (GPT-4, GPT-3.5)</span>
                          <span className="text-xs text-muted-foreground">Industry standard, most popular</span>
                        </div>
                      </SelectItem>
                      <SelectItem value="claude">
                        <div className="flex flex-col items-start">
                          <span className="font-medium">Anthropic (Claude)</span>
                          <span className="text-xs text-muted-foreground">Advanced reasoning, latest models</span>
                        </div>
                      </SelectItem>
                      <SelectItem value="gemini">
                        <div className="flex flex-col items-start">
                          <span className="font-medium">Google (Gemini)</span>
                          <span className="text-xs text-muted-foreground">Fast, multimodal capabilities</span>
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="pro-api-key">{providerInfo.name} API Key</Label>
                  <div className="relative">
                    <Input
                      id="pro-api-key"
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
                    Don't have a {providerInfo.name} API key?{' '}
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
              </>
            ) : (
              <div className="text-center py-8 space-y-4">
                <Crown size={48} className="mx-auto text-yellow-500" />
                <div>
                  <h3 className="font-semibold text-lg">Upgrade to Pro</h3>
                  <p className="text-sm text-muted-foreground mt-2">
                    Access multiple AI providers including OpenAI's GPT-4, Anthropic's Claude, and more.
                  </p>
                </div>
                <Button variant="default" className="mt-4">
                  Upgrade to Pro
                </Button>
              </div>
            )}
          </TabsContent>
        </Tabs>

        <Alert>
          <Info size={16} />
          <AlertDescription className="text-xs">
            Your API key is stored only in your browser's session storage and is never sent to our servers.
            It's used to communicate directly with the selected AI provider's API.
          </AlertDescription>
        </Alert>

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
            onClick={() => {
              if (tier === 'free') {
                setSelectedProvider('gemini')
                if (geminiApiKey) {
                  setInputKey(geminiApiKey)
                  handleSave()
                }
              } else {
                handleSave()
              }
            }}
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
