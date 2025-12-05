import { useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { Alert, AlertDescription } from '@/components/ui/alert'
import * as Icons from '@phosphor-icons/react'

interface DeleteBlockDialogProps {
  isOpen: boolean
  onClose: () => void
  onConfirm: (cascade: boolean) => void
  blockName: string
  instanceCount: number
}

export default function DeleteBlockDialog({
  isOpen,
  onClose,
  onConfirm,
  blockName,
  instanceCount
}: DeleteBlockDialogProps) {
  const [cascade, setCascade] = useState(false)

  const handleConfirm = () => {
    onConfirm(cascade)
    onClose()
    setCascade(false)
  }

  return (
    <Dialog open={isOpen} onOpenChange={(open) => {
      if (!open) {
        onClose()
        setCascade(false)
      }
    }}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-destructive">
            <Icons.Warning size={20} />
            Delete Block Definition
          </DialogTitle>
          <DialogDescription>
            Are you sure you want to delete "{blockName}"?
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {instanceCount > 0 && (
            <Alert variant="destructive">
              <Icons.Info size={16} className="h-4 w-4" />
              <AlertDescription>
                This block has <strong>{instanceCount}</strong> instance{instanceCount !== 1 ? 's' : ''} on the canvas.
              </AlertDescription>
            </Alert>
          )}

          {instanceCount > 0 && (
            <div className="flex items-start gap-3 p-3 rounded-lg border bg-card">
              <Checkbox
                id="cascade"
                checked={cascade}
                onCheckedChange={(checked) => setCascade(checked as boolean)}
                className="mt-1"
              />
              <div className="flex-1 space-y-1">
                <Label htmlFor="cascade" className="cursor-pointer font-medium">
                  Delete all instances from canvas
                </Label>
                <p className="text-xs text-muted-foreground">
                  {cascade
                    ? `All ${instanceCount} instance${instanceCount !== 1 ? 's' : ''} will be removed from the canvas.`
                    : `Instances will remain but show "Definition not found" error.`}
                </p>
              </div>
            </div>
          )}

          <div className="text-sm text-muted-foreground">
            This action cannot be undone.
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={handleConfirm}
          >
            <Icons.Trash size={16} className="mr-2" />
            Delete {cascade && instanceCount > 0 ? 'All' : 'Definition'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
