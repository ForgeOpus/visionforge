import { useState } from 'react'
import { useModelBuilderStore } from '@/lib/store'
import { getNodeDefinition, BackendFramework } from '@/lib/nodes/registry'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { X, ArrowLeft, ArrowCounterClockwise, PencilSimple } from '@phosphor-icons/react'
import { toast } from 'sonner'
import { BlockType, BlockConfig } from '@/lib/types'

interface InternalNodeConfigPanelProps {
  selectedNodeId: string
  parentGroupNodeId: string
  groupDefinitionId: string
  internalNodeId: string
  onClose: () => void
}

export default function InternalNodeConfigPanel({
  selectedNodeId,
  parentGroupNodeId,
  groupDefinitionId,
  internalNodeId,
  onClose
}: InternalNodeConfigPanelProps) {
  const {
    nodes,
    groupDefinitions,
    updateGroupInternalNodeConfig,
    getEffectiveInternalNodeConfig,
    resetGroupInternalNodeConfig,
    hasConfigOverrides
  } = useModelBuilderStore()

  const groupDef = groupDefinitions.get(groupDefinitionId)
  
  // When a group is expanded, the parent group node is removed from the canvas
  // We need to find it by looking for the container node which has the same _expandedFrom ID
  // OR we can work with a virtual parent node data structure
  // For now, let's find the container which has the parent info
  const containerNode = nodes.find(n => 
    n.id === `${parentGroupNodeId}-container` && 
    (n.data as any)._expandedFrom === parentGroupNodeId
  )
  
  // Create a virtual parent node structure for the expanded group
  // The overrides are stored on the collapsed group node, but when expanded,
  // we need to track them separately or reconstruct them
  // For now, we'll use an empty overrides object and store them on the container
  const parentGroupNode = containerNode ? {
    id: parentGroupNodeId,
    data: {
      blockType: 'group' as const,
      groupDefinitionId: groupDefinitionId,
      instanceConfigOverrides: (containerNode.data as any)._instanceConfigOverrides || {}
    }
  } : null
  
  const internalNodeDef = groupDef?.internalNodes.find(n => n.id === internalNodeId)

  if (!groupDef || !parentGroupNode || !internalNodeDef) {
    // Debug logging
    console.log('InternalNodeConfigPanel - Debug Info:', {
      groupDefinitionId,
      hasGroupDef: !!groupDef,
      parentGroupNodeId,
      hasParentNode: !!parentGroupNode,
      hasContainerNode: !!containerNode,
      internalNodeId,
      hasInternalNodeDef: !!internalNodeDef,
      availableInternalNodeIds: groupDef?.internalNodes.map(n => n.id)
    })

    return (
      <div className="w-80 bg-card border-l border-border h-full flex items-center justify-center">
        <div className="text-center text-muted-foreground p-6">
          <p className="text-sm">Configuration not available</p>
          <p className="text-xs mt-2">
            {!groupDef && 'Group definition not found'}
            {!parentGroupNode && 'Parent group node not found'}
            {!internalNodeDef && 'Internal node not found'}
          </p>
        </div>
      </div>
    )
  }

  const nodeDef = getNodeDefinition(internalNodeDef.data.blockType as BlockType, BackendFramework.PyTorch)
  if (!nodeDef) {
    return (
      <div className="w-80 bg-card border-l border-border h-full flex items-center justify-center">
        <div className="text-center text-muted-foreground p-6">
          <p className="text-sm">Node type not supported</p>
        </div>
      </div>
    )
  }

  const effectiveConfig = getEffectiveInternalNodeConfig(parentGroupNodeId, internalNodeId) || {}
  const baseConfig = internalNodeDef.data.config
  const hasAnyOverrides = hasConfigOverrides(parentGroupNodeId, internalNodeId)

  const handleConfigChange = (fieldName: string, value: any) => {
    updateGroupInternalNodeConfig(parentGroupNodeId, internalNodeId, {
      [fieldName]: value
    })
  }

  const handleResetField = (fieldName: string) => {
    resetGroupInternalNodeConfig(parentGroupNodeId, internalNodeId, fieldName)
    toast.success('Field reset to default')
  }

  const handleResetAll = () => {
    resetGroupInternalNodeConfig(parentGroupNodeId, internalNodeId)
    toast.success('All fields reset to defaults')
  }

  const isFieldOverridden = (fieldName: string): boolean => {
    const parentGroupData = parentGroupNode.data as any
    const overrides = parentGroupData.instanceConfigOverrides?.[internalNodeId]
    return overrides?.[fieldName] !== undefined
  }

  const getDefaultValue = (fieldName: string): any => {
    return baseConfig[fieldName]
  }

  const selectedNode = nodes.find(n => n.id === selectedNodeId)

  return (
    <div className="w-80 bg-card border-l border-border h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-border shrink-0">
        <div className="flex items-start justify-between mb-2">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6"
                onClick={onClose}
              >
                <ArrowLeft size={14} />
              </Button>
              <h2 className="font-semibold text-base truncate">
                {internalNodeDef.data.label}
              </h2>
            </div>
            <div className="text-xs text-muted-foreground ml-8">
              Inside: {groupDef.name}
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
          >
            <X size={18} />
          </Button>
        </div>

        {hasAnyOverrides && (
          <div className="flex items-center gap-2 mt-3">
            <Badge variant="secondary" className="text-xs">
              <PencilSimple size={12} className="mr-1" />
              Customized
            </Badge>
            <Button
              variant="outline"
              size="sm"
              className="h-7 text-xs"
              onClick={handleResetAll}
            >
              <ArrowCounterClockwise size={12} className="mr-1" />
              Reset All
            </Button>
          </div>
        )}
      </div>

      {/* Configuration Fields */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-4 space-y-6">
          {nodeDef.configSchema.length > 0 ? (
            nodeDef.configSchema.map((field) => {
              const isOverridden = isFieldOverridden(field.name)
              const defaultValue = getDefaultValue(field.name)
              const currentValue = effectiveConfig[field.name]

              return (
                <div key={field.name} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label className="text-sm font-medium">
                      {field.label}
                      {field.required && <span className="text-destructive ml-1">*</span>}
                    </Label>
                    {isOverridden && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 text-xs"
                        onClick={() => handleResetField(field.name)}
                      >
                        <ArrowCounterClockwise size={12} className="mr-1" />
                        Reset
                      </Button>
                    )}
                  </div>

                  {field.description && (
                    <p className="text-xs text-muted-foreground">{field.description}</p>
                  )}

                  {isOverridden && (
                    <div className="flex items-center gap-2 text-xs">
                      <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                        <PencilSimple size={10} className="mr-1" />
                        Override
                      </Badge>
                      <span className="text-muted-foreground">
                        Default: {String(defaultValue ?? 'none')}
                      </span>
                    </div>
                  )}

                  {field.type === 'text' && (
                    <Input
                      type="text"
                      value={String(currentValue ?? field.default ?? '')}
                      onChange={(e) => handleConfigChange(field.name, e.target.value)}
                      placeholder={field.placeholder || `Enter ${field.label.toLowerCase()}`}
                      className={`font-mono text-sm ${isOverridden ? 'border-blue-500' : ''}`}
                    />
                  )}

                  {field.type === 'number' && (
                    <Input
                      type="number"
                      min={field.min}
                      max={field.max}
                      value={Number(currentValue ?? field.default ?? 0)}
                      onChange={(e) => handleConfigChange(field.name, parseFloat(e.target.value) || 0)}
                      placeholder={`Enter ${field.label.toLowerCase()}`}
                      className={isOverridden ? 'border-blue-500' : ''}
                    />
                  )}

                  {field.type === 'boolean' && (
                    <div className="flex items-center gap-2">
                      <Switch
                        checked={currentValue as boolean ?? field.default}
                        onCheckedChange={(checked) => handleConfigChange(field.name, checked)}
                        className={isOverridden ? 'border-blue-500' : ''}
                      />
                      <span className="text-sm">
                        {currentValue ? 'Enabled' : 'Disabled'}
                      </span>
                    </div>
                  )}

                  {field.type === 'select' && field.options && (
                    <Select
                      value={String(currentValue ?? field.default ?? '')}
                      onValueChange={(value) => handleConfigChange(field.name, value)}
                    >
                      <SelectTrigger className={isOverridden ? 'border-blue-500' : ''}>
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

                  {field.type === 'multiselect' && field.options && (
                    <div className={`space-y-3 border border-input rounded-md p-3 bg-muted/30 ${isOverridden ? 'border-blue-500' : ''}`}>
                      {field.options.map((opt) => {
                        const currentValues = currentValue ?? field.default ?? []
                        const isChecked = Array.isArray(currentValues) && currentValues.includes(opt.value)

                        return (
                          <div key={opt.value} className="flex items-center gap-2">
                            <Checkbox
                              id={`${field.name}-${opt.value}`}
                              checked={isChecked}
                              onCheckedChange={(checked) => {
                                const newValues = Array.isArray(currentValues) ? [...currentValues] : []
                                if (checked) {
                                  if (!newValues.includes(opt.value)) {
                                    newValues.push(opt.value)
                                  }
                                } else {
                                  const index = newValues.indexOf(opt.value)
                                  if (index > -1) {
                                    newValues.splice(index, 1)
                                  }
                                }
                                handleConfigChange(field.name, newValues)
                              }}
                            />
                            <Label htmlFor={`${field.name}-${opt.value}`} className="font-normal cursor-pointer">
                              {opt.label}
                            </Label>
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>
              )
            })
          ) : (
            <div className="text-sm text-muted-foreground">
              No configuration required for this block.
            </div>
          )}

          {/* Shape Information */}
          {selectedNode?.data.inputShape && (
            <Card className="p-3 bg-muted/50">
              <div className="text-xs font-medium mb-1">Input Shape</div>
              <div className="font-mono text-sm">
                [{selectedNode.data.inputShape.dims.join(', ')}]
              </div>
            </Card>
          )}

          {selectedNode?.data.outputShape && (
            <Card className="p-3 bg-accent/10 border-accent/30">
              <div className="text-xs font-medium mb-1">Output Shape</div>
              <div className="font-mono text-sm font-semibold">
                [{selectedNode.data.outputShape.dims.join(', ')}]
              </div>
            </Card>
          )}
        </div>
      </div>

      {/* Info Footer */}
      <div className="p-4 border-t border-border shrink-0 bg-muted/30">
        <div className="text-xs text-muted-foreground">
          <p className="mb-1">
            Editing internal node configuration for this group instance.
          </p>
          <p>
            Changes only affect this instance, not the group definition.
          </p>
        </div>
      </div>
    </div>
  )
}
