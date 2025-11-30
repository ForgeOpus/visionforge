import { useState, useEffect } from 'react'
import CodeEditor from '@/components/CodeEditor'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'

// Professional custom layer template following PyTorch conventions
const DEFAULT_CUSTOM_LAYER_TEMPLATE = `"""Custom Layer Template - PyTorch

TODO: Implement your custom layer following PyTorch conventions.
"""
import torch
import torch.nn as nn

class CustomLayer(nn.Module):
    """
    Custom User-Defined Layer

    Shape:
        - Input: [batch, *] (Define your input shape)
        - Output: [batch, *] (Define your output shape)
    """

    def __init__(self):
        """Initialize the custom layer."""
        super(CustomLayer, self).__init__()

        # TODO: Define your layer parameters here
        # Examples:
        # self.linear = nn.Linear(in_features, out_features)
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # self.activation = nn.ReLU()

        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the custom layer."""
        # TODO: Implement your forward pass logic here

        # Placeholder: returns input unchanged
        return x`

interface CustomLayerModalProps {
  isOpen: boolean
  onClose: () => void
  onSave: (config: {
    name: string
    code: string
    output_shape?: string
    description?: string
  }) => void
  initialConfig?: {
    name?: string
    code?: string
    output_shape?: string
    description?: string
  }
}

export default function CustomLayerModal({
  isOpen,
  onClose,
  onSave,
  initialConfig
}: CustomLayerModalProps) {
  const [name, setName] = useState(initialConfig?.name || '')
  const [code, setCode] = useState(
    initialConfig?.code || DEFAULT_CUSTOM_LAYER_TEMPLATE
  )
  const [outputShape, setOutputShape] = useState(initialConfig?.output_shape || '')
  const [description, setDescription] = useState(initialConfig?.description || '')

  // Update state when initialConfig changes
  useEffect(() => {
    if (isOpen) {
      setName(initialConfig?.name || '')
      setCode(initialConfig?.code || DEFAULT_CUSTOM_LAYER_TEMPLATE)
      setOutputShape(initialConfig?.output_shape || '')
      setDescription(initialConfig?.description || '')
    }
  }, [isOpen, initialConfig])

  const handleSave = () => {
    if (!name.trim()) {
      return // Name is required
    }

    onSave({
      name: name.trim(),
      code: code.trim(),
      output_shape: outputShape.trim() || undefined,
      description: description.trim() || undefined
    })
    onClose()
  }

  const handleCancel = () => {
    onClose()
  }

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="w-[80vw] sm:max-w-[80vw] max-h-[90vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>Configure Custom Layer</DialogTitle>
          <DialogDescription>
            Define your custom layer implementation in Python. Press Ctrl/Cmd+S to save, Esc to close.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="layer-name">
              Layer Name <span className="text-destructive">*</span>
            </Label>
            <Input
              id="layer-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="my_custom_layer"
              className="font-mono"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="layer-code">
              Python Code <span className="text-destructive">*</span>
            </Label>
            <CodeEditor
              value={code}
              onChange={(value) => setCode(value)}
              height="60vh"
              editable={true}
              enableSearch={true}
              onSave={handleSave}
              onEscape={onClose}
              ariaLabel="Custom layer Python code editor"
              placeholder="Write your custom PyTorch layer code here..."
            />
            <p className="text-xs text-muted-foreground">
              Write the forward pass logic. Input tensor is available as <code className="text-xs bg-muted px-1 py-0.5 rounded">x</code>
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="output-shape">Output Shape (optional)</Label>
            <Input
              id="output-shape"
              value={outputShape}
              onChange={(e) => setOutputShape(e.target.value)}
              placeholder="[batch, features]"
              className="font-mono text-sm"
            />
            <p className="text-xs text-muted-foreground">
              JSON array format. Leave empty to match input shape.
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="description">Description (optional)</Label>
            <Textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe what this layer does..."
              rows={3}
            />
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleCancel}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={!name.trim()}>
            Save Layer
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
