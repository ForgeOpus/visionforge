import { useState } from 'react'
import CodeEditor from '@/components/CodeEditor'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Code } from '@phosphor-icons/react'
import { toast } from 'sonner'

interface ViewCodeModalProps {
  isOpen: boolean
  onClose: () => void
  code: string
  nodeType: string
  framework: 'pytorch' | 'tensorflow'
  isLoading?: boolean
}

export default function ViewCodeModal({
  isOpen,
  onClose,
  code,
  nodeType,
  framework,
  isLoading = false
}: ViewCodeModalProps) {
  const handleCopyCode = () => {
    navigator.clipboard.writeText(code)
    toast.success('Code copied to clipboard!')
  }

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="w-[80vw] sm:max-w-[80vw] max-h-[90vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>
            View Code: {nodeType} ({framework.toUpperCase()})
          </DialogTitle>
          <DialogDescription>
            Read-only view of generated Python code for this block. Press Esc to close.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto py-4">
          {isLoading ? (
            <div className="space-y-2">
              <Skeleton className="h-6 w-full" />
              <Skeleton className="h-6 w-4/5" />
              <Skeleton className="h-6 w-full" />
              <Skeleton className="h-6 w-3/4" />
              <Skeleton className="h-6 w-full" />
              <Skeleton className="h-6 w-5/6" />
            </div>
          ) : (
            <CodeEditor
              value={code}
              height="80vh"
              readOnly={true}
              enableSearch={true}
              onEscape={onClose}
              ariaLabel={`Read-only Python code for ${nodeType} block`}
            />
          )}
        </div>

        <DialogFooter className="flex gap-2">
          <Button
            variant="default"
            onClick={handleCopyCode}
            disabled={isLoading || !code}
          >
            <Code size={16} className="mr-2" />
            Copy Code
          </Button>
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
