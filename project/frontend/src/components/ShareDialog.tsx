import { useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Copy, Link, Check } from '@phosphor-icons/react'
import { toast } from 'sonner'
import { enableSharing, disableSharing } from '@/lib/projectApi'

interface ShareDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  projectId: string
  initialIsShared: boolean
  initialShareToken: string | null
}

export default function ShareDialog({
  open,
  onOpenChange,
  projectId,
  initialIsShared,
  initialShareToken,
}: ShareDialogProps) {
  const [isShared, setIsShared] = useState(initialIsShared)
  const [shareToken, setShareToken] = useState<string | null>(initialShareToken)
  const [isLoading, setIsLoading] = useState(false)
  const [copied, setCopied] = useState(false)

  const shareUrl = shareToken
    ? `${window.location.origin}/shared/${shareToken}`
    : null

  const handleToggle = async (checked: boolean) => {
    setIsLoading(true)
    try {
      if (checked) {
        const result = await enableSharing(projectId)
        setShareToken(result.share_token)
        setIsShared(true)
        toast.success('Sharing enabled', {
          description: 'Anyone with the link can view this project',
        })
      } else {
        await disableSharing(projectId)
        setIsShared(false)
        toast.success('Sharing disabled', {
          description: 'The link is no longer active',
        })
      }
    } catch (error) {
      toast.error('Failed to update sharing settings', {
        description: error instanceof Error ? error.message : 'Unknown error',
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleCopy = () => {
    if (!shareUrl) return
    navigator.clipboard.writeText(shareUrl)
    setCopied(true)
    toast.success('Link copied to clipboard!')
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Link size={18} />
            Share Project
          </DialogTitle>
          <DialogDescription>
            Generate a read-only link so others can view and copy your architecture.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-5 pt-2">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="share-toggle" className="text-sm font-medium">
                Enable public link
              </Label>
              <p className="text-xs text-muted-foreground">
                Anyone with the link can view this project in read-only mode
              </p>
            </div>
            <Switch
              id="share-toggle"
              checked={isShared}
              onCheckedChange={handleToggle}
              disabled={isLoading}
            />
          </div>

          {isShared && shareUrl && (
            <div className="space-y-2">
              <Label className="text-sm font-medium">Shareable link</Label>
              <div className="flex gap-2">
                <Input
                  value={shareUrl}
                  readOnly
                  className="font-mono text-xs"
                  onClick={(e) => (e.target as HTMLInputElement).select()}
                />
                <Button
                  variant="outline"
                  size="icon"
                  onClick={handleCopy}
                  className="shrink-0"
                  title="Copy link"
                >
                  {copied ? (
                    <Check size={16} className="text-green-500" />
                  ) : (
                    <Copy size={16} />
                  )}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Viewers can pan, zoom, and inspect nodes. Signed-in viewers can make a copy or export the code.
                Turning the link off deactivates it — turning it back on restores the same URL.
              </p>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
