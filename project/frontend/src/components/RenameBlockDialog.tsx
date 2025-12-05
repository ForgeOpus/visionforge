import { useState, useEffect } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { validateBlockName } from '@/lib/blockValidation'
import * as Icons from '@phosphor-icons/react'

interface RenameBlockDialogProps {
  isOpen: boolean
  onClose: () => void
  onSave: (newName: string) => void
  currentName: string
  existingNames: string[]
}

export default function RenameBlockDialog({
  isOpen,
  onClose,
  onSave,
  currentName,
  existingNames
}: RenameBlockDialogProps) {
  const [name, setName] = useState(currentName)
  const [nameError, setNameError] = useState('')

  useEffect(() => {
    if (isOpen) {
      setName(currentName)
      setNameError('')
    }
  }, [isOpen, currentName])

  const validateName = (value: string) => {
    // Filter out current name from existing names
    const otherNames = existingNames.filter(n => n !== currentName)
    const result = validateBlockName(value, otherNames)
    
    if (!result.isValid && result.errors.length > 0) {
      setNameError(result.errors[0])
      return false
    }
    
    setNameError('')
    return true
  }

  const handleSave = () => {
    if (!validateName(name)) return
    onSave(name.trim())
    onClose()
  }

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Icons.PencilSimple size={20} />
            Rename Block
          </DialogTitle>
          <DialogDescription>
            Enter a new name for this block definition
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="name">
              Block Name <span className="text-red-500">*</span>
            </Label>
            <Input
              id="name"
              placeholder="e.g., MultiHeadAttentionBlock"
              value={name}
              onChange={(e) => {
                setName(e.target.value)
                validateName(e.target.value)
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && name.trim() && !nameError) {
                  handleSave()
                }
              }}
              className={nameError ? 'border-red-500' : ''}
              autoFocus
            />
            {nameError && (
              <p className="text-sm text-red-500">{nameError}</p>
            )}
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            disabled={!name.trim() || !!nameError || name === currentName}
          >
            <Icons.Check size={16} className="mr-2" />
            Rename
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
