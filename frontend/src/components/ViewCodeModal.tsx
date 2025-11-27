import { useState } from 'react'
import CodeMirror from '@uiw/react-codemirror'
import { python } from '@codemirror/lang-python'
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogDescription,
  DialogFooter 
} from '@visionforge/core/components/ui/dialog'
import { Button } from '@visionforge/core/components/ui/button'
import { Skeleton } from '@visionforge/core/components/ui/skeleton'
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
      <DialogContent className="max-w-3xl max-h-[90vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>
            View Code: {nodeType} ({framework.toUpperCase()})
          </DialogTitle>
          <DialogDescription>
            Read-only view of generated Python code for this block
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
            <div className="border rounded-md overflow-hidden">
              <CodeMirror
                value={code}
                height="400px"
                extensions={[python()]}
                editable={false}
                readOnly={true}
                theme="light"
                basicSetup={{
                  lineNumbers: true,
                  highlightActiveLineGutter: false,
                  highlightActiveLine: false,
                  foldGutter: true
                }}
              />
            </div>
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
