import { useState, useEffect } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { BlockCategory, PortMapping } from '@/lib/types'
import { useModelBuilderStore } from '@/lib/store'
import { getNodeDefinition, BackendFramework } from '@/lib/nodes/registry'
import { validateConnectivity, detectCycles, validateBlockName } from '@/lib/blockValidation'
import { toast } from 'sonner'
import * as Icons from '@phosphor-icons/react'

interface GroupCreationDialogProps {
  isOpen: boolean
  onClose: () => void
  onSave: (config: {
    name: string
    description: string
    category: BlockCategory
    color: string
    portMappings: PortMapping[]
  }) => void
  selectedNodeIds: string[]
}

const COLOR_OPTIONS = [
  { value: '#9333ea', label: 'Purple', color: '#9333ea' },
  { value: '#ec4899', label: 'Pink', color: '#ec4899' },
  { value: '#f59e0b', label: 'Orange', color: '#f59e0b' },
  { value: '#10b981', label: 'Green', color: '#10b981' },
  { value: '#3b82f6', label: 'Blue', color: '#3b82f6' },
  { value: '#ef4444', label: 'Red', color: '#ef4444' },
  { value: '#8b5cf6', label: 'Violet', color: '#8b5cf6' },
  { value: '#06b6d4', label: 'Cyan', color: '#06b6d4' },
]

interface PortInfo {
  nodeId: string
  nodeName: string
  portId: string
  portLabel: string
  type: 'input' | 'output'
  semantic: string
  isExternal: boolean
}

export default function GroupCreationDialog({
  isOpen,
  onClose,
  onSave,
  selectedNodeIds
}: GroupCreationDialogProps) {
  const [step, setStep] = useState(1)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [category, setCategory] = useState<BlockCategory>('utility')
  const [color, setColor] = useState('#9333ea')
  const [nameError, setNameError] = useState('')
  const [validationErrors, setValidationErrors] = useState<string[]>([])
  const [selectedPorts, setSelectedPorts] = useState<Set<string>>(new Set())
  const [portLabels, setPortLabels] = useState<Map<string, string>>(new Map())

  const nodes = useModelBuilderStore((state) => state.nodes)
  const edges = useModelBuilderStore((state) => state.edges)
  const currentProject = useModelBuilderStore((state) => state.currentProject)
  const groupDefinitions = useModelBuilderStore((state) => state.groupDefinitions)

  // Discover available ports from selected nodes
  const availablePorts: PortInfo[] = []
  const selectedNodes = nodes.filter(n => selectedNodeIds.includes(n.id))

  selectedNodes.forEach(node => {
    const nodeDef = getNodeDefinition(node.data.blockType, currentProject?.framework as any || BackendFramework.PyTorch)
    if (!nodeDef) return

    const inputPorts = nodeDef.getInputPorts ? nodeDef.getInputPorts(node.data.config) : []
    const outputPorts = nodeDef.getOutputPorts ? nodeDef.getOutputPorts(node.data.config) : []

    // Check which ports have external connections
    inputPorts.forEach(port => {
      const hasExternalConnection = edges.some(e =>
        e.target === node.id &&
        (e.targetHandle || 'default') === port.id &&
        !selectedNodeIds.includes(e.source)
      )
      availablePorts.push({
        nodeId: node.id,
        nodeName: node.data.label || node.data.blockType,
        portId: port.id,
        portLabel: port.label,
        type: 'input',
        semantic: port.semantic,
        isExternal: hasExternalConnection
      })
    })

    outputPorts.forEach(port => {
      const hasExternalConnection = edges.some(e =>
        e.source === node.id &&
        (e.sourceHandle || 'default') === port.id &&
        !selectedNodeIds.includes(e.target)
      )
      availablePorts.push({
        nodeId: node.id,
        nodeName: node.data.label || node.data.blockType,
        portId: port.id,
        portLabel: port.label,
        type: 'output',
        semantic: port.semantic,
        isExternal: hasExternalConnection
      })
    })
  })

  useEffect(() => {
    if (isOpen) {
      setStep(1)
      setName('')
      setDescription('')
      setCategory('utility')
      setColor('#9333ea')
      setNameError('')
      setValidationErrors([])
      setSelectedPorts(new Set())
      setPortLabels(new Map())

      // Validate selection on open
      const errors: string[] = []

      // Check connectivity
      const connectivityResult = validateConnectivity(selectedNodeIds, edges)
      if (!connectivityResult.isValid) {
        errors.push(...connectivityResult.errors)
      }

      // Check for cycles
      const cycleResult = detectCycles(selectedNodeIds, edges)
      if (!cycleResult.isValid) {
        errors.push(...cycleResult.errors)
      }

      setValidationErrors(errors)

      // Auto-select external ports
      const autoSelected = new Set<string>()
      const autoLabels = new Map<string, string>()
      availablePorts.forEach(port => {
        if (port.isExternal) {
          const portKey = `${port.nodeId}-${port.portId}-${port.type}`
          autoSelected.add(portKey)
          autoLabels.set(portKey, `${port.type === 'input' ? 'Input' : 'Output'} ${autoLabels.size + 1}`)
        }
      })
      setSelectedPorts(autoSelected)
      setPortLabels(autoLabels)
    }
  }, [isOpen, selectedNodeIds, edges])

  const validateName = (value: string) => {
    // Get existing block names
    const existingNames = Array.from(groupDefinitions.values()).map(def => def.name)
    
    const result = validateBlockName(value, existingNames)
    
    if (!result.isValid && result.errors.length > 0) {
      setNameError(result.errors[0])
      return false
    }
    
    setNameError('')
    return true
  }

  const togglePort = (portKey: string) => {
    const newSelected = new Set(selectedPorts)
    if (newSelected.has(portKey)) {
      newSelected.delete(portKey)
      const newLabels = new Map(portLabels)
      newLabels.delete(portKey)
      setPortLabels(newLabels)
    } else {
      newSelected.add(portKey)
      // Auto-generate label if not exists
      if (!portLabels.has(portKey)) {
        const port = availablePorts.find(p => `${p.nodeId}-${p.portId}-${p.type}` === portKey)
        if (port) {
          const newLabels = new Map(portLabels)
          const count = Array.from(selectedPorts).filter(k => k.endsWith(port.type)).length + 1
          newLabels.set(portKey, `${port.type === 'input' ? 'Input' : 'Output'} ${count}`)
          setPortLabels(newLabels)
        }
      }
    }
    setSelectedPorts(newSelected)
  }

  const updatePortLabel = (portKey: string, label: string) => {
    const newLabels = new Map(portLabels)
    newLabels.set(portKey, label)
    setPortLabels(newLabels)
  }

  const handleNext = () => {
    // Check for structural validation errors first
    if (validationErrors.length > 0) {
      // Show toast with first error for better UX
      toast.error('Cannot proceed', {
        description: validationErrors[0]
      })
      return
    }
    
    if (!validateName(name)) {
      toast.error('Invalid block name', {
        description: nameError
      })
      return
    }
    setStep(2)
  }

  const handleBack = () => {
    setStep(1)
  }

  const handleSave = () => {
    // Check for structural validation errors
    if (validationErrors.length > 0) {
      toast.error('Cannot create block', {
        description: validationErrors[0]
      })
      return
    }
    
    if (!validateName(name)) {
      toast.error('Invalid block name', {
        description: nameError
      })
      return
    }

    // Validate port selection
    if (selectedPorts.size === 0) {
      setValidationErrors(['At least one port must be exposed'])
      toast.error('No ports selected', {
        description: 'At least one port must be exposed on the block'
      })
      return
    }

    // Build port mappings from selections
    const portMappings: PortMapping[] = []
    let inputIndex = 0
    let outputIndex = 0

    selectedPorts.forEach(portKey => {
      const port = availablePorts.find(p => `${p.nodeId}-${p.portId}-${p.type}` === portKey)
      if (!port) return

      const externalPortId = port.type === 'input'
        ? `group-input-${inputIndex++}`
        : `group-output-${outputIndex++}`

      portMappings.push({
        internalNodeId: port.nodeId,
        internalPortId: port.portId,
        externalPortId,
        externalPortLabel: portLabels.get(portKey) || port.portLabel,
        type: port.type,
        semantic: port.semantic as any
      })
    })

    onSave({
      name: name.trim(),
      description: description.trim(),
      category,
      color,
      portMappings
    })
    onClose()
  }

  const inputPorts = availablePorts.filter(p => p.type === 'input')
  const outputPorts = availablePorts.filter(p => p.type === 'output')

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl max-h-[90vh]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Icons.SquaresFour size={24} weight="bold" style={{ color }} />
            Create Block from Selection
            <Badge variant="outline" className="ml-auto">
              Step {step} of 2
            </Badge>
          </DialogTitle>
          <DialogDescription>
            Group {selectedNodeIds.length} nodes into a reusable block
          </DialogDescription>
        </DialogHeader>

        {step === 1 && (
          <div className="space-y-4 py-4">
          {/* Validation Errors */}
          {validationErrors.length > 0 && (
            <Alert variant="destructive">
              <Icons.Warning size={16} className="h-4 w-4" />
              <AlertDescription>
                <ul className="list-disc list-inside space-y-1">
                  {validationErrors.map((error, index) => (
                    <li key={index}>{error}</li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          )}
          
          {/* Name Input */}
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
              className={nameError ? 'border-red-500' : ''}
            />
            {nameError && (
              <p className="text-sm text-red-500">{nameError}</p>
            )}
          </div>

          {/* Description Input */}
          <div className="space-y-2">
            <Label htmlFor="description">Description (Optional)</Label>
            <Textarea
              id="description"
              placeholder="Describe what this block does..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
            />
          </div>

          {/* Category Selection */}
          <div className="space-y-2">
            <Label htmlFor="category">Category</Label>
            <Select value={category} onValueChange={(value) => setCategory(value as BlockCategory)}>
              <SelectTrigger id="category">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="utility">Utility</SelectItem>
                <SelectItem value="basic">Basic</SelectItem>
                <SelectItem value="advanced">Advanced</SelectItem>
                <SelectItem value="activation">Activation</SelectItem>
                <SelectItem value="merge">Merge</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Color Selection */}
          <div className="space-y-2">
            <Label>Color</Label>
            <div className="flex gap-2 flex-wrap">
              {COLOR_OPTIONS.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  className={`w-10 h-10 rounded-md transition-all ${
                    color === option.value
                      ? 'ring-2 ring-offset-2 ring-accent scale-110'
                      : 'hover:scale-105'
                  }`}
                  style={{ backgroundColor: option.color }}
                  onClick={() => setColor(option.value)}
                  title={option.label}
                />
              ))}
            </div>
          </div>

          {/* Preview */}
          <div className="space-y-2">
            <Label>Preview</Label>
            <div
              className="p-3 rounded-md border-2 border-dashed"
              style={{ borderColor: color }}
            >
              <div className="flex items-center gap-2">
                <div
                  className="p-1.5 rounded"
                  style={{ backgroundColor: color, color: 'white' }}
                >
                  <Icons.SquaresFour size={16} weight="bold" />
                </div>
                <div>
                  <div className="font-semibold text-sm">
                    {name || 'Block Name'}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {category} • {selectedNodeIds.length} nodes
                  </div>
                </div>
              </div>
              {description && (
                <div className="mt-2 text-xs text-muted-foreground">
                  {description}
                </div>
              )}
            </div>
          </div>
        </div>
        )}

        {step === 2 && (
          <ScrollArea className="max-h-[500px] py-4">
            <div className="space-y-6 pr-4">
              {/* Port Selection Validation Error */}
              {validationErrors.length > 0 && (
                <Alert variant="destructive">
                  <Icons.Warning size={16} className="h-4 w-4" />
                  <AlertDescription>
                    <ul className="list-disc list-inside space-y-1">
                      {validationErrors.map((error, index) => (
                        <li key={index}>{error}</li>
                      ))}
                    </ul>
                  </AlertDescription>
                </Alert>
              )}
              
              <div className="text-sm text-muted-foreground">
                Select which ports to expose on the group block. Ports with external connections are auto-selected.
              </div>

              {/* Input Ports */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Icons.ArrowsIn size={18} className="text-blue-500" />
                  <Label className="text-base font-semibold">Input Ports ({inputPorts.length})</Label>
                </div>
                {inputPorts.length === 0 ? (
                  <div className="text-sm text-muted-foreground italic">No input ports available</div>
                ) : (
                  <div className="space-y-2">
                    {inputPorts.map(port => {
                      const portKey = `${port.nodeId}-${port.portId}-${port.type}`
                      const isSelected = selectedPorts.has(portKey)
                      return (
                        <div key={portKey} className="flex items-start gap-3 p-3 rounded-lg border bg-card">
                          <Checkbox
                            checked={isSelected}
                            onCheckedChange={() => togglePort(portKey)}
                            className="mt-1"
                          />
                          <div className="flex-1 space-y-2">
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-sm">{port.nodeName}</span>
                              <Icons.ArrowRight size={12} className="text-muted-foreground" />
                              <span className="text-sm text-muted-foreground">{port.portLabel}</span>
                              {port.isExternal && (
                                <Badge variant="secondary" className="text-xs">External</Badge>
                              )}
                              <Badge variant="outline" className="text-xs">{port.semantic}</Badge>
                            </div>
                            {isSelected && (
                              <Input
                                placeholder="External port label..."
                                value={portLabels.get(portKey) || ''}
                                onChange={(e) => updatePortLabel(portKey, e.target.value)}
                                className="h-8 text-xs"
                              />
                            )}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>

              {/* Output Ports */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Icons.ArrowsOut size={18} className="text-green-500" />
                  <Label className="text-base font-semibold">Output Ports ({outputPorts.length})</Label>
                </div>
                {outputPorts.length === 0 ? (
                  <div className="text-sm text-muted-foreground italic">No output ports available</div>
                ) : (
                  <div className="space-y-2">
                    {outputPorts.map(port => {
                      const portKey = `${port.nodeId}-${port.portId}-${port.type}`
                      const isSelected = selectedPorts.has(portKey)
                      return (
                        <div key={portKey} className="flex items-start gap-3 p-3 rounded-lg border bg-card">
                          <Checkbox
                            checked={isSelected}
                            onCheckedChange={() => togglePort(portKey)}
                            className="mt-1"
                          />
                          <div className="flex-1 space-y-2">
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-sm">{port.nodeName}</span>
                              <Icons.ArrowRight size={12} className="text-muted-foreground" />
                              <span className="text-sm text-muted-foreground">{port.portLabel}</span>
                              {port.isExternal && (
                                <Badge variant="secondary" className="text-xs">External</Badge>
                              )}
                              <Badge variant="outline" className="text-xs">{port.semantic}</Badge>
                            </div>
                            {isSelected && (
                              <Input
                                placeholder="External port label..."
                                value={portLabels.get(portKey) || ''}
                                onChange={(e) => updatePortLabel(portKey, e.target.value)}
                                className="h-8 text-xs"
                              />
                            )}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>

              {/* Summary */}
              <div className="flex items-center gap-2 p-3 bg-muted rounded-lg">
                <Icons.Info size={16} />
                <span className="text-sm">
                  {selectedPorts.size} port(s) selected
                </span>
              </div>
            </div>
          </ScrollArea>
        )}

        <DialogFooter className="flex items-center gap-2">
          {step === 1 ? (
            <>
              <Button variant="outline" onClick={onClose}>
                Cancel
              </Button>
              <Button
                onClick={handleNext}
                disabled={!name.trim() || validationErrors.length > 0}
                style={{
                  backgroundColor: color,
                  color: 'white'
                }}
              >
                Next: Select Ports
                <Icons.ArrowRight size={16} className="ml-2" />
              </Button>
            </>
          ) : (
            <>
              <Button variant="outline" onClick={handleBack}>
                <Icons.ArrowLeft size={16} className="mr-2" />
                Back
              </Button>
              <Button
                onClick={handleSave}
                disabled={selectedPorts.size === 0}
                style={{
                  backgroundColor: color,
                  color: 'white'
                }}
              >
                <Icons.SquaresFour size={16} weight="bold" className="mr-2" />
                Create Block
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
