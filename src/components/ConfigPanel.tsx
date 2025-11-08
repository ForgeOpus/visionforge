import { useModelBuilderStore } from '@/lib/store'
import { getBlockDefinition } from '@/lib/blockDefinitions'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Card } from '@/components/ui/card'
import { X } from '@phosphor-icons/react'

export default function ConfigPanel() {
  const { nodes, selectedNodeId, updateNode, setSelectedNodeId, removeNode } = useModelBuilderStore()

  const selectedNode = nodes.find((n) => n.id === selectedNodeId)

  if (!selectedNode) {
    return (
      <div className="w-80 bg-card border-l border-border h-full flex items-center justify-center">
        <div className="text-center text-muted-foreground p-6">
          <p className="text-sm">Select a block to configure its parameters</p>
        </div>
      </div>
    )
  }

  const definition = getBlockDefinition(selectedNode.data.blockType)
  if (!definition) return null

  const handleConfigChange = (fieldName: string, value: any) => {
    updateNode(selectedNode.id, {
      config: {
        ...selectedNode.data.config,
        [fieldName]: value
      }
    })
  }

  const handleDelete = () => {
    removeNode(selectedNode.id)
  }

  return (
    <div className="w-80 bg-card border-l border-border h-full flex flex-col">
      <div className="p-4 border-b border-border flex items-center justify-between">
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

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-6">
          {definition.configSchema.length > 0 ? (
            definition.configSchema.map((field) => (
              <div key={field.name} className="space-y-2">
                <Label className="text-sm font-medium">
                  {field.label}
                  {field.required && <span className="text-destructive ml-1">*</span>}
                </Label>

                {field.description && (
                  <p className="text-xs text-muted-foreground">{field.description}</p>
                )}

                {field.type === 'number' && (
                  <Input
                    type="number"
                    min={field.min}
                    max={field.max}
                    value={Number(selectedNode.data.config[field.name] ?? field.default ?? 0)}
                    onChange={(e) => handleConfigChange(field.name, parseFloat(e.target.value) || 0)}
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
      </ScrollArea>

      <div className="p-4 border-t border-border">
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
