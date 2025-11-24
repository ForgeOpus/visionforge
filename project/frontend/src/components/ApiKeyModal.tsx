import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import * as Icons from '@phosphor-icons/react'
import { useApiKey } from '@/lib/apiKeyContext'
import { toast } from 'sonner'

interface ApiKeyModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSuccess?: () => void
}

export default function ApiKeyModal({ open, onOpenChange, onSuccess }: ApiKeyModalProps) {
  const [keyInput, setKeyInput] = useState('')
  const [showKey, setShowKey] = useState(false)
  const { setApiKey } = useApiKey()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    const trimmedKey = keyInput.trim()

    // Basic validation - Gemini API keys typically start with "AI"
    if (!trimmedKey) {
      toast.error('API key required', {
        description: 'Please enter your Gemini API key'
      })
      return
    }

    if (trimmedKey.length < 20) {
      toast.error('Invalid API key', {
        description: 'The API key appears to be too short'
      })
      return
    }

    setApiKey(trimmedKey)
    setKeyInput('')
    onOpenChange(false)

    toast.success('API key saved', {
      description: 'Your Gemini API key has been saved for this session'
    })

    if (onSuccess) {
      onSuccess()
    }
  }

  const handleClose = () => {
    setKeyInput('')
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Icons.Key size={20} className="text-primary" />
            Gemini API Key Required
          </DialogTitle>
          <DialogDescription>
            To use the AI assistant, please provide your own Gemini API key.
            Your key is stored only in your browser session and is never saved on our servers.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit}>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="api-key">API Key</Label>
              <div className="relative">
                <Input
                  id="api-key"
                  type={showKey ? 'text' : 'password'}
                  placeholder="Enter your Gemini API key"
                  value={keyInput}
                  onChange={(e) => setKeyInput(e.target.value)}
                  className="pr-10"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="absolute right-0 top-0 h-full px-3"
                  onClick={() => setShowKey(!showKey)}
                >
                  {showKey ? (
                    <Icons.EyeSlash size={16} />
                  ) : (
                    <Icons.Eye size={16} />
                  )}
                </Button>
              </div>
            </div>

            <div className="rounded-md bg-muted p-3 text-sm">
              <p className="font-medium mb-1">How to get an API key:</p>
              <ol className="list-decimal list-inside space-y-1 text-muted-foreground">
                <li>Visit <a
                  href="https://aistudio.google.com/apikey"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  Google AI Studio
                </a></li>
                <li>Sign in with your Google account</li>
                <li>Click "Create API key"</li>
                <li>Copy and paste the key above</li>
              </ol>
            </div>

            <div className="flex items-start gap-2 text-xs text-muted-foreground">
              <Icons.ShieldCheck size={16} className="shrink-0 mt-0.5 text-green-500" />
              <p>
                Your API key is stored only in your browser's session storage and will be
                cleared when you close this tab. It is sent directly to Google's API and
                is never stored on our servers.
              </p>
            </div>
          </div>

          <DialogFooter>
            <Button type="button" variant="outline" onClick={handleClose}>
              Cancel
            </Button>
            <Button type="submit">
              <Icons.Check size={16} className="mr-2" />
              Save Key
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
