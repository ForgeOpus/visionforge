import { useState, useEffect, useRef } from 'react'
import { useModelBuilderStore } from '@visionforge/core/store'
import { getNodeDefinition, BackendFramework } from '@visionforge/core/nodes'
import { Input } from '@visionforge/core/components/ui/input'
import { Label } from '@visionforge/core/components/ui/label'
import { Switch } from '@visionforge/core/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@visionforge/core/components/ui/select'
import { Button } from '@visionforge/core/components/ui/button'
import { Card } from '@visionforge/core/components/ui/card'
import { X, UploadSimple } from '@phosphor-icons/react'
import { toast } from 'sonner'
import CustomLayerModal from './CustomLayerModal'

export default function ConfigPanel() {
  const { nodes, selectedNodeId, updateNode, setSelectedNodeId, removeNode } = useModelBuilderStore()
  const [isCustomModalOpen, setIsCustomModalOpen] = useState(false)
  const fileInputRefs = useRef<{ [key: string]: HTMLInputElement | null }>({})

  const selectedNode = nodes.find((n) => n.id === selectedNodeId)

  const handleCustomLayerSave = (config: {
    name: string
    code: string
    output_shape?: string
    description?: string
  }) => {
    if (selectedNode) {
      updateNode(selectedNode.id, {
        config: {
          ...selectedNode.data.config,
          ...config
        }
      })
    }
  }

  // Automatically open modal when custom block is selected
  useEffect(() => {
    if (selectedNode?.data.blockType === 'custom') {
      setIsCustomModalOpen(true)
    }
  }, [selectedNode?.id, selectedNode?.data.blockType])

  // For custom blocks, don't show the sidebar at all - only the modal
  if (selectedNode?.data.blockType === 'custom') {
    return (
      <CustomLayerModal
        isOpen={isCustomModalOpen}
        onClose={() => {
          setIsCustomModalOpen(false)
          setSelectedNodeId(null) // Deselect the node when modal closes
        }}
        onSave={handleCustomLayerSave}
        initialConfig={{
          name: selectedNode.data.config.name as string | undefined,
          code: selectedNode.data.config.code as string | undefined,
          output_shape: selectedNode.data.config.output_shape as string | undefined,
          description: selectedNode.data.config.description as string | undefined
        }}
      />
    )
  }

  if (!selectedNode) {
    return (
      <div className="w-80 bg-card border-l border-border h-full flex items-center justify-center">
        <div className="text-center text-muted-foreground p-6">
          <p className="text-sm">Select a block to configure its parameters</p>
        </div>
      </div>
    )
  }

  const nodeDef = getNodeDefinition(selectedNode.data.blockType, BackendFramework.PyTorch)
  if (!nodeDef) return null
  
  const definition = {
    label: nodeDef.metadata.label,
    description: nodeDef.metadata.description,
    configSchema: nodeDef.configSchema
  }

  const handleConfigChange = (fieldName: string, value: any) => {
    updateNode(selectedNode.id, {
      config: {
        ...selectedNode.data.config,
        [fieldName]: value
      }
    })
  }

  const isValidTensorShape = (shapeStr: string): boolean => {
    try {
      const dims = JSON.parse(shapeStr)
      return Array.isArray(dims) && dims.length > 0 && dims.every(d => typeof d === 'number' && d > 0)
    } catch {
      return false
    }
  }

  const handleDelete = () => {
    removeNode(selectedNode.id)
  }

  const handleFileUpload = async (fieldName: string, file: File) => {
    try {
      // Read file as base64 for storage
      const reader = new FileReader()
      reader.onload = (e) => {
        const fileContent = e.target?.result as string

        // Store both the file content and filename
        handleConfigChange(fieldName, fileContent)
        handleConfigChange(`${fieldName}name`, file.name)

        toast.success('File uploaded', {
          description: `${file.name} loaded successfully`
        })
      }
      reader.onerror = () => {
        toast.error('Failed to read file')
      }
      reader.readAsDataURL(file)
    } catch (error) {
      toast.error('File upload failed', {
        description: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }

  return (
    <div className="w-80 bg-card border-l border-border h-full flex flex-col overflow-hidden">
      <div className="p-4 border-b border-border flex items-center justify-between shrink-0">
        <div>
          <h2 className="font-semibold text-lg">{definition.label}</h2>
          <p className="text-xs text-muted-foreground">{definition.description}</p>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setSelectedNodeId(null)}
        >
          <X size={18} />
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="p-4 space-y-6">
          {definition.configSchema.length > 0 ? (
            definition.configSchema
              .filter(field => field.name !== 'code') // Skip code field, it's handled in modal
              .filter(field => field.name !== 'csv_filename') // Skip csv_filename, it's auto-populated
              .map((field) => (
              <div key={field.name} className="space-y-2">
                <Label className="text-sm font-medium">
                  {field.label}
                  {field.required && <span className="text-destructive ml-1">*</span>}
                </Label>

                {field.description && (
                  <p className="text-xs text-muted-foreground">{field.description}</p>
                )}

                {field.type === 'text' && (
                  <div className="space-y-2">
                    <Input
                      type="text"
                      value={String(selectedNode.data.config[field.name] ?? field.default ?? '')}
                      onChange={(e) => handleConfigChange(field.name, e.target.value)}
                      placeholder={field.placeholder || `Enter ${field.label.toLowerCase()}`}
                      className={`font-mono text-sm ${
                        field.name === 'shape' && selectedNode.data.config[field.name] && !isValidTensorShape(String(selectedNode.data.config[field.name]))
                          ? 'border-destructive focus-visible:ring-destructive'
                          : ''
                      }`}
                    />
                    {field.name === 'shape' && selectedNode.data.config[field.name] && !isValidTensorShape(String(selectedNode.data.config[field.name])) && (
                      <p className="text-xs text-destructive">Invalid shape format. Use JSON array like [1, 3, 224, 224]</p>
                    )}
                    {field.name === 'shape' && (
                      <div className="space-y-1 mt-3">
                        <p className="text-xs font-medium text-muted-foreground">Quick Presets:</p>
                        <div className="flex flex-wrap gap-1.5">
                          <button
                            type="button"
                            onClick={() => handleConfigChange(field.name, '[1, 3, 224, 224]')}
                            className="text-xs px-2 py-1 rounded bg-secondary hover:bg-secondary/80 font-mono"
                          >
                            Image
                          </button>
                          <button
                            type="button"
                            onClick={() => handleConfigChange(field.name, '[32, 512, 768]')}
                            className="text-xs px-2 py-1 rounded bg-secondary hover:bg-secondary/80 font-mono"
                          >
                            Text
                          </button>
                          <button
                            type="button"
                            onClick={() => handleConfigChange(field.name, '[16, 1, 16000]')}
                            className="text-xs px-2 py-1 rounded bg-secondary hover:bg-secondary/80 font-mono"
                          >
                            Audio
                          </button>
                          <button
                            type="button"
                            onClick={() => handleConfigChange(field.name, '[8, 100, 13]')}
                            className="text-xs px-2 py-1 rounded bg-secondary hover:bg-secondary/80 font-mono"
                          >
                            Tabular
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {field.type === 'number' && (
                  <Input
                    type="number"
                    min={field.min}
                    max={field.max}
                    value={Number(selectedNode.data.config[field.name] ?? field.default ?? '')}
                    onChange={(e) => {
                      const value = e.target.value === '' ? undefined : parseFloat(e.target.value)
                      if (value !== undefined && !isNaN(value)) {
                        handleConfigChange(field.name, value)
                      } else if (e.target.value === '') {
                        handleConfigChange(field.name, undefined)
                      }
                    }}
                    placeholder={`Enter ${field.label.toLowerCase()}`}
                  />
                )}

                {field.type === 'boolean' && (
                  <div className="flex items-center gap-2">
                    <Switch
                      checked={selectedNode.data.config[field.name] as boolean ?? field.default}
                      onCheckedChange={(checked) => handleConfigChange(field.name, checked)}
                    />
                    <span className="text-sm">
                      {selectedNode.data.config[field.name] ? 'Enabled' : 'Disabled'}
                    </span>
                  </div>
                )}

                {field.type === 'select' && field.options && (
                  <Select
                    value={String(selectedNode.data.config[field.name] ?? field.default ?? '')}
                    onValueChange={(value) => handleConfigChange(field.name, value)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder={`Select ${field.label.toLowerCase()}`} />
                    </SelectTrigger>
                    <SelectContent>
                      {field.options.map((opt) => (
                        <SelectItem key={opt.value} value={String(opt.value)}>
                          {opt.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}

                {field.type === 'file' && (
                  <div className="space-y-2">
                    <input
                      ref={(el) => { fileInputRefs.current[field.name] = el }}
                      type="file"
                      accept={field.accept || '*'}
                      onChange={(e) => {
                        const file = e.target.files?.[0]
                        if (file) {
                          handleFileUpload(field.name, file)
                        }
                      }}
                      className="hidden"
                    />
                    <Button
                      type="button"
                      variant="outline"
                      className="w-full"
                      onClick={() => fileInputRefs.current[field.name]?.click()}
                    >
                      <UploadSimple size={16} className="mr-2" />
                      {selectedNode.data.config[`${field.name}name`] || 'Choose File'}
                    </Button>
                    {selectedNode.data.config[`${field.name}name`] && (
                      <div className="flex items-center justify-between p-2 bg-muted rounded text-xs">
                        <span className="truncate">{selectedNode.data.config[`${field.name}name`]}</span>
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          className="h-5 w-5 hover:bg-destructive/10"
                          onClick={() => {
                            handleConfigChange(field.name, '')
                            handleConfigChange(`${field.name}name`, '')
                            toast.info('File removed')
                          }}
                        >
                          <X size={12} />
                        </Button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))
          ) : (
            <div className="text-sm text-muted-foreground">
              No configuration required for this block.
            </div>
          )}

          {selectedNode.data.inputShape && (
            <Card className="p-3 bg-muted/50">
              <div className="text-xs font-medium mb-1">Input Shape</div>
              <div className="font-mono text-sm">
                [{selectedNode.data.inputShape.dims.join(', ')}]
              </div>
            </Card>
          )}

          {selectedNode.data.outputShape && (
            <Card className="p-3 bg-accent/10 border-accent/30">
              <div className="text-xs font-medium mb-1">Output Shape</div>
              <div className="font-mono text-sm font-semibold">
                [{selectedNode.data.outputShape.dims.join(', ')}]
              </div>
            </Card>
          )}
        </div>
      </div>

      <div className="p-4 border-t border-border shrink-0">
        <Button
          variant="destructive"
          className="w-full"
          onClick={handleDelete}
        >
          <X size={16} className="mr-2" />
          Delete Block
        </Button>
      </div>
    </div>
  )
}
