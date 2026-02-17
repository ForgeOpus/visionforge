import { memo, Fragment } from 'react'
import { Handle, Position, NodeProps } from '@xyflow/react'
import { BlockData, BlockType } from '@/lib/types'
import { getNodeDefinition, BackendFramework } from '@/lib/nodes/registry'
import { useModelBuilderStore } from '@/lib/store'
import * as Icons from '@phosphor-icons/react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'

interface BlockNodeProps {
  data: BlockData & {
    onViewCode?: (nodeId: string) => void
    onReplicate?: (nodeId: string) => void
  }
  selected?: boolean
  id: string
}

const BlockNode = memo(({ data, selected, id }: BlockNodeProps) => {
  const nodeDef = getNodeDefinition(data.blockType as BlockType, BackendFramework.PyTorch)
  const validationErrors = useModelBuilderStore((state) => state.validationErrors)
  const edges = useModelBuilderStore((state) => state.edges)
  const hasConfigOverrides = useModelBuilderStore((state) => state.hasConfigOverrides)

  if (!nodeDef) return null

  const definition = nodeDef.metadata
  const IconComponent = (Icons as any)[definition.icon] || Icons.Cube

  // Check if this node has any validation errors
  const nodeErrors = validationErrors.filter((error) => error.nodeId === id && error.type === 'error')
  const hasErrors = nodeErrors.length > 0
  
  // Check if this is an expanded internal node with overrides
  const isExpandedInternal = (data as any)._isExpandedInternal === true
  const parentGroupNodeId = (data as any)._expandedFrom as string | undefined
  const internalNodeId = id.split('-expanded-')[0]
  const hasOverrides = isExpandedInternal && parentGroupNodeId 
    ? hasConfigOverrides(parentGroupNodeId, internalNodeId)
    : false
  
  // Helper to check if a handle is already connected
  const isHandleConnected = (handleId: string, isTarget: boolean) => {
    return edges.some(edge => {
      if (isTarget) {
        return edge.target === id && (edge.targetHandle || 'default') === handleId
      } else {
        return edge.source === id && (edge.sourceHandle || 'default') === handleId
      }
    })
  }

  const formatShape = (dims?: (number | string)[]) => {
    if (!dims) return '?'
    return `[${dims.join(', ')}]`
  }

  return (
    <Card
      className="min-w-[200px] w-[220px] transition-all duration-200 relative"
      style={{
        borderColor: selected ? 'var(--color-accent)' : definition.color,
        borderWidth: selected ? 3 : 2,
        boxShadow: selected ? '0 0 20px rgba(0, 188, 212, 0.3)' : 'none'
      }}
    >
      {/* Error Badge */}
      {hasErrors && (
        <div className="absolute -top-2 -right-2 z-20">
          <div className="bg-red-500 rounded-full p-1 shadow-lg">
            <Icons.Warning size={16} weight="fill" className="text-white" />
          </div>
        </div>
      )}

      {/* Override Badge - for expanded internal nodes with customizations */}
      {hasOverrides && !hasErrors && (
        <div className="absolute -top-2 -right-2 z-20">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="bg-blue-500 rounded-full p-1 shadow-lg cursor-help">
                  <Icons.PencilSimple size={14} weight="fill" className="text-white" />
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <div className="text-xs">
                  <div className="font-semibold">Customized</div>
                  <div>This node has custom configuration</div>
                </div>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      )}

      {/* Action Buttons - Only shown when selected */}
      {selected && (
        <div className="absolute top-2 right-2 flex gap-1 z-30 animate-in fade-in duration-200">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7 bg-background/80 backdrop-blur-sm hover:bg-accent shadow-sm"
                  onClick={(e) => {
                    e.stopPropagation()
                    data.onViewCode?.(id)
                  }}
                >
                  <Icons.Eye size={14} />
                </Button>
              </TooltipTrigger>
              <TooltipContent>View Code</TooltipContent>
            </Tooltip>

            {data.blockType !== 'custom' && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 bg-background/80 backdrop-blur-sm hover:bg-accent shadow-sm"
                    onClick={(e) => {
                      e.stopPropagation()
                      data.onReplicate?.(id)
                    }}
                  >
                    <Icons.Copy size={14} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Replicate as Custom</TooltipContent>
              </Tooltip>
            )}
          </TooltipProvider>
        </div>
      )}

      {data.blockType !== 'dataloader' && data.blockType !== 'loss' && data.blockType !== 'metrics' && (
        <>
          {/* Get input port ID from node definition */}
          {(() => {
            const inputPorts = nodeDef.getInputPorts ? nodeDef.getInputPorts(data.config) : []
            const inputPort = inputPorts.length > 0 ? inputPorts[0] : null
            const handleId = inputPort?.id || 'default'
            const isConnected = isHandleConnected(handleId, true)

            return (
              <>
                <Handle
                  type="target"
                  position={Position.Left}
                  id={handleId}
                  className={`w-3 h-3 !bg-accent transition-all ${isConnected ? 'ring-2 ring-offset-1 ring-green-400' : ''}`}
                  style={{
                    left: -6,
                    zIndex: 10,
                    opacity: isConnected ? 1 : 0.8
                  }}
                />
                {selected && (
                  <div
                    className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-1/2 w-6 h-6 rounded-full border-2 border-accent bg-accent/20 animate-pulse pointer-events-none"
                    style={{ left: -6 }}
                  />
                )}
              </>
            )
          })()}
        </>
      )}

      <div className="p-2 space-y-1.5">
        <div className="flex items-center gap-2">
          <div
            className="p-1 rounded"
            style={{
              backgroundColor: definition.color,
              color: 'white'
            }}
          >
            <IconComponent size={14} weight="bold" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="font-medium text-xs truncate">
              {data.config?.label || definition.label}
            </div>
            <Badge
              variant="secondary"
              className="text-[9px] px-1 py-0 h-3.5"
            >
              {data.category}
            </Badge>
          </div>
        </div>

        {data.inputShape && (
          <div className="text-[10px] font-mono text-muted-foreground leading-tight">
            In: {formatShape(data.inputShape.dims)}
          </div>
        )}

        {/* Always show output shape if available */}
        {data.outputShape && (
          <div className="text-[10px] font-mono text-foreground font-medium leading-tight">
            Out: {formatShape(data.outputShape.dims)}
          </div>
        )}

        {/* For DataLoader with multiple outlets - show all output shapes */}
        {data.blockType === 'dataloader' && (() => {
          const numInputOutlets = Number(data.config?.num_input_outlets || 1)
          const hasGT = data.config?.has_ground_truth
          const shapes: React.ReactElement[] = []

          // Add additional input outlet shapes if configured
          if (numInputOutlets > 1 && data.config?.input_shapes) {
            try {
              const additionalShapes = JSON.parse(String(data.config.input_shapes))
              if (Array.isArray(additionalShapes)) {
                additionalShapes.forEach((shape: any, idx: number) => {
                  if (Array.isArray(shape)) {
                    shapes.push(
                      <div key={`out-${idx + 2}`} className="text-[10px] font-mono text-blue-600 font-medium leading-tight">
                        Out {idx + 2}: {formatShape(shape)}
                      </div>
                    )
                  }
                })
              }
            } catch (e) {
              // Invalid JSON, ignore
            }
          }

          // Add GT shape
          if (hasGT && data.config?.ground_truth_shape) {
            try {
              const gtShape = typeof data.config.ground_truth_shape === 'string'
                ? JSON.parse(data.config.ground_truth_shape)
                : data.config.ground_truth_shape

              if (Array.isArray(gtShape)) {
                shapes.push(
                  <div key="gt-shape" className="text-[10px] font-mono text-green-600 font-medium leading-tight">
                    GT: {formatShape(gtShape)}
                  </div>
                )
              }
            } catch (e) {
              // Invalid format, ignore
            }
          }

          return shapes
        })()}

        {/* Loss node input ports display */}
        {data.blockType === 'loss' && (() => {
          const lossNodeDef = nodeDef as any
          const inputPorts = lossNodeDef.getInputPorts ? lossNodeDef.getInputPorts(data.config) : []

          if (inputPorts.length === 0) return null

          return (
            <div className="space-y-1 mt-2">
              <div className="text-[9px] font-semibold text-muted-foreground uppercase tracking-wide">Inputs</div>
              {inputPorts.map((port: any, i: number) => (
                <div
                  key={`port-label-${i}`}
                  className="text-[10px] flex items-center gap-1.5 relative"
                  id={`loss-port-row-${i}`}
                >
                  <div
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ backgroundColor: ['#ef4444', '#f59e0b'][i % 2] }}
                  />
                  <span className="text-muted-foreground">{port.label}</span>
                </div>
              ))}
            </div>
          )
        })()}

        {/* Metrics node input ports display */}
        {data.blockType === 'metrics' && (() => {
          const metricsNodeDef = nodeDef as any
          const inputPorts = metricsNodeDef.getInputPorts ? metricsNodeDef.getInputPorts(data.config) : []

          if (inputPorts.length === 0) return null

          return (
            <div className="space-y-1 mt-2">
              <div className="text-[9px] font-semibold text-muted-foreground uppercase tracking-wide">Inputs</div>
              {inputPorts.map((port: any, i: number) => (
                <div
                  key={`port-label-${i}`}
                  className="text-[10px] flex items-center gap-1.5 relative"
                  id={`metrics-port-row-${i}`}
                >
                  <div
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ backgroundColor: ['#3b82f6', '#8b5cf6'][i % 2] }}
                  />
                  <span className="text-muted-foreground">{port.label}</span>
                </div>
              ))}
            </div>
          )
        })()}

        {!data.outputShape && data.blockType !== 'input' && data.blockType !== 'dataloader' && data.blockType !== 'groundtruth' && data.blockType !== 'empty' && data.blockType !== 'output' &&  data.blockType !== 'loss' && (
          <div className="text-[10px] text-orange-600">
            Configure params
          </div>
        )}
      </div>

      {/* Multiple output handles for DataLoader node */}
      {data.blockType === 'dataloader' ? (
        <>
          {(() => {
            const numInputOutlets = Number(data.config?.num_input_outlets || 1)
            const hasGT = data.config?.has_ground_truth
            const totalOutlets = numInputOutlets + (hasGT ? 1 : 0)
            const spacing = 100 / (totalOutlets + 1)

            const outlets: React.ReactElement[] = []
            const colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981']

            // Add input outlets
            for (let i = 0; i < numInputOutlets; i++) {
              const topPercent = spacing * (i + 1)
              const color = colors[i % colors.length]
              const handleId = numInputOutlets > 1 ? `input-output-${i}` : 'input-output'
              const isConnected = isHandleConnected(handleId, false)

              outlets.push(
                <div key={`input-${i}`} className="absolute right-0 flex items-center" style={{ top: `${topPercent}%`, transform: 'translateY(-50%)' }}>
                  <span 
                    className={`text-[10px] font-medium mr-2 bg-card px-1.5 py-0.5 rounded border ${isConnected ? 'opacity-60' : ''}`}
                    style={{ color: isConnected ? '#10b981' : color, borderColor: isConnected ? '#10b981' : color }}
                  >
                    In{numInputOutlets > 1 ? ` ${i + 1}` : ''} {isConnected && '✓'}
                  </span>
                  <Handle
                    type="source"
                    position={Position.Right}
                    id={handleId}
                    className={`w-3 h-3 transition-all border-2 border-card ${isConnected ? 'ring-2 ring-offset-1 ring-green-400' : ''}`}
                    style={{
                      position: 'relative',
                      right: -6,
                      zIndex: 10,
                      backgroundColor: isConnected ? '#10b981' : color,
                      opacity: isConnected ? 1 : 0.8
                    }}
                  />
                  {selected && (
                    <div
                      className="absolute right-0 w-6 h-6 rounded-full border-2 animate-pulse pointer-events-none"
                      style={{
                        top: 0,
                        right: -6,
                        transform: 'translate(50%, -50%)',
                        borderColor: isConnected ? '#10b981' : color,
                        backgroundColor: `${isConnected ? '#10b981' : color}33`
                      }}
                    />
                  )}
                </div>
              )
            }

            // Add ground truth outlet
            if (hasGT) {
              const topPercent = spacing * (numInputOutlets + 1)
              const gtColor = '#10b981'
              const handleId = 'ground-truth-output'
              const isConnected = isHandleConnected(handleId, false)

              outlets.push(
                <div key="gt" className="absolute right-0 flex items-center" style={{ top: `${topPercent}%`, transform: 'translateY(-50%)' }}>
                  <span 
                    className={`text-[10px] text-green-600 font-medium mr-2 bg-card px-1.5 py-0.5 rounded border border-green-200 ${isConnected ? 'opacity-60' : ''}`}
                  >
                    GT {isConnected && '✓'}
                  </span>
                  <Handle
                    type="source"
                    position={Position.Right}
                    id={handleId}
                    className={`w-3 h-3 !bg-green-500 transition-all border-2 border-card ${isConnected ? 'ring-2 ring-offset-1 ring-green-400' : ''}`}
                    style={{
                      position: 'relative',
                      right: -6,
                      zIndex: 10,
                      opacity: isConnected ? 1 : 0.8
                    }}
                  />
                  {selected && (
                    <div
                      className="absolute right-0 w-6 h-6 rounded-full border-2 border-green-500 bg-green-500/20 animate-pulse pointer-events-none"
                      style={{ top: 0, right: -6, transform: 'translate(50%, -50%)' }}
                    />
                  )}
                </div>
              )
            }

            return outlets
          })()}
        </>
      ) : data.blockType === 'loss' ? (
        <>
          {/* Multiple input handles for Loss node - aligned with labels */}
          {(() => {
            const lossNodeDef = nodeDef as any
            const inputPorts = lossNodeDef.getInputPorts ? lossNodeDef.getInputPorts(data.config) : []

            if (inputPorts.length === 0) {
              // Fallback to default single input
              const isConnected = isHandleConnected('default', true)
              return (
                <>
                  <Handle
                    type="target"
                    position={Position.Left}
                    className={`w-3 h-3 !bg-red-400 transition-all ${isConnected ? 'ring-2 ring-offset-1 ring-green-400' : ''}`}
                    style={{
                      left: -6,
                      zIndex: 10,
                      opacity: isConnected ? 1 : 0.8
                    }}
                  />
                  {selected && (
                    <div
                      className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-1/2 w-6 h-6 rounded-full border-2 border-red-400 bg-red-400/20 animate-pulse pointer-events-none"
                      style={{ left: -6 }}
                    />
                  )}
                </>
              )
            }

            // Calculate positions to align with label rows
            // Estimated positions based on card layout:
            // - padding-top: 8px
            // - header row: ~28px
            // - space before inputs: ~6px
            // - "INPUTS" header: ~14px
            // - space: ~4px
            // For typical card height of ~110px:
            // First label: ~60px from top (54.5%)
            // Second label: ~82px from top (74.5%)
            const positions = inputPorts.length === 2
              ? [60, 82] // pixel positions for 2 ports
              : inputPorts.length === 3
              ? [56, 72, 88] // pixel positions for 3 ports
              : [70] // single port fallback

            const colors = ['#ef4444', '#f59e0b', '#10b981']

            return inputPorts.map((port: any, i: number) => {
              const topPx = positions[i] || 70
              const color = colors[i % colors.length]
              const handleId = port.id
              const isConnected = isHandleConnected(handleId, true)

              return (
                <Fragment key={`loss-input-${i}`}>
                  <Handle
                    type="target"
                    position={Position.Left}
                    id={handleId}
                    className={`w-3 h-3 transition-all ${isConnected ? 'ring-2 ring-offset-1 ring-green-400' : ''}`}
                    style={{
                      top: `${topPx}px`,
                      left: -6,
                      zIndex: 10,
                      backgroundColor: isConnected ? '#10b981' : color,
                      opacity: isConnected ? 1 : 0.8
                    }}
                  />
                  {selected && (
                    <div
                      className="absolute left-0 w-6 h-6 rounded-full border-2 animate-pulse pointer-events-none"
                      style={{
                        top: `${topPx}px`,
                        left: -6,
                        transform: 'translate(-50%, -50%)',
                        borderColor: isConnected ? '#10b981' : color,
                        backgroundColor: `${isConnected ? '#10b981' : color}33`
                      }}
                    />
                  )}
                </Fragment>
              )
            })
          })()}

        </>
      ) : data.blockType === 'metrics' ? (
        <>
          {/* Multiple input handles for Metrics node - aligned with labels */}
          {(() => {
            const metricsNodeDef = nodeDef as any
            const inputPorts = metricsNodeDef.getInputPorts ? metricsNodeDef.getInputPorts(data.config) : []

            if (inputPorts.length === 0) {
              // Fallback to default single input
              const isConnected = isHandleConnected('default', true)
              return (
                <>
                  <Handle
                    type="target"
                    position={Position.Left}
                    className={`w-3 h-3 !bg-blue-400 transition-all ${isConnected ? 'ring-2 ring-offset-1 ring-green-400' : ''}`}
                    style={{
                      left: -6,
                      zIndex: 10,
                      opacity: isConnected ? 1 : 0.8
                    }}
                  />
                  {selected && (
                    <div
                      className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-1/2 w-6 h-6 rounded-full border-2 border-blue-400 bg-blue-400/20 animate-pulse pointer-events-none"
                      style={{ left: -6 }}
                    />
                  )}
                </>
              )
            }

            const positions = inputPorts.length === 2
              ? [60, 82]
              : inputPorts.length === 3
              ? [56, 72, 88]
              : [70]

            const colors = ['#3b82f6', '#8b5cf6']

            return inputPorts.map((port: any, i: number) => {
              const topPx = positions[i] || 70
              const color = colors[i % colors.length]
              const handleId = port.id
              const isConnected = isHandleConnected(handleId, true)

              return (
                <Fragment key={`metrics-input-${i}`}>
                  <Handle
                    type="target"
                    position={Position.Left}
                    id={handleId}
                    className={`w-3 h-3 transition-all ${isConnected ? 'ring-2 ring-offset-1 ring-green-400' : ''}`}
                    style={{
                      top: `${topPx}px`,
                      left: -6,
                      zIndex: 10,
                      backgroundColor: isConnected ? '#10b981' : color,
                      opacity: isConnected ? 1 : 0.8
                    }}
                  />
                  {selected && (
                    <div
                      className="absolute left-0 w-6 h-6 rounded-full border-2 animate-pulse pointer-events-none"
                      style={{
                        top: `${topPx}px`,
                        left: -6,
                        transform: 'translate(-50%, -50%)',
                        borderColor: isConnected ? '#10b981' : color,
                        backgroundColor: `${isConnected ? '#10b981' : color}33`
                      }}
                    />
                  )}
                </Fragment>
              )
            })
          })()}
        </>
      ) : (
        <>
          {/* Get output port ID from node definition */}
          {(() => {
            const outputPorts = nodeDef.getOutputPorts ? nodeDef.getOutputPorts(data.config) : []
            const outputPort = outputPorts.length > 0 ? outputPorts[0] : null
            const handleId = outputPort?.id || 'default'
            const isConnected = isHandleConnected(handleId, false)
            
            return (
              <>
                <Handle
                  type="source"
                  position={Position.Right}
                  id={handleId}
                  className={`w-3 h-3 !bg-accent transition-all ${isConnected ? 'ring-2 ring-offset-1 ring-green-400' : ''}`}
                  style={{
                    right: -6,
                    zIndex: 10,
                    opacity: isConnected ? 1 : 0.8
                  }}
                />
                {selected && (
                  <div
                    className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2 w-6 h-6 rounded-full border-2 border-accent bg-accent/20 animate-pulse pointer-events-none"
                    style={{ right: -6 }}
                  />
                )}
              </>
            )
          })()}
        </>
      )}
    </Card>
  )
})

BlockNode.displayName = 'BlockNode'

export default BlockNode
