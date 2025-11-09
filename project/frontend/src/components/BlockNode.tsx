import { memo } from 'react'
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

  if (!nodeDef) return null

  const definition = nodeDef.metadata
  const IconComponent = (Icons as any)[definition.icon] || Icons.Cube

  // Check if this node has any validation errors
  const nodeErrors = validationErrors.filter((error) => error.nodeId === id && error.type === 'error')
  const hasErrors = nodeErrors.length > 0
  
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

      {data.blockType !== 'dataloader' && data.blockType !== 'loss' && (
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

        {!data.outputShape && data.blockType !== 'input' && data.blockType !== 'dataloader' && data.blockType !== 'empty' && (
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
          {/* Multiple input handles for Loss node based on loss type */}
          {(() => {
            // Get input ports from the node definition
            const lossNodeDef = nodeDef as any
            const inputPorts = lossNodeDef.getInputPorts ? lossNodeDef.getInputPorts(data.config) : []
            
            if (inputPorts.length === 0) {
              // Fallback to default single input
              return (
                <>
                  <Handle
                    type="target"
                    position={Position.Left}
                    className="w-3 h-3 !bg-accent transition-all"
                    style={{
                      left: -6,
                      zIndex: 10
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
            }

            const spacing = 100 / (inputPorts.length + 1)
            const colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6']

            return inputPorts.map((port: any, i: number) => {
              const topPercent = spacing * (i + 1)
              const color = colors[i % colors.length]
              const handleId = port.id  // Port ID already includes 'loss-input-' prefix
              const isConnected = isHandleConnected(handleId, true)

              return (
                <div key={`loss-input-${i}`} className="absolute left-0 flex items-center" style={{ top: `${topPercent}%`, transform: 'translateY(-50%)' }}>
                  <Handle
                    type="target"
                    position={Position.Left}
                    id={handleId}
                    className={`w-3 h-3 transition-all border-2 border-card ${isConnected ? 'ring-2 ring-offset-1 ring-green-400' : ''}`}
                    style={{
                      position: 'relative',
                      left: -6,
                      zIndex: 10,
                      backgroundColor: isConnected ? '#10b981' : color,
                      opacity: isConnected ? 1 : 0.8
                    }}
                  />
                  <span 
                    className={`text-[10px] font-medium ml-2 bg-card px-1.5 py-0.5 rounded border ${isConnected ? 'opacity-60' : ''}`}
                    style={{ color: isConnected ? '#10b981' : color, borderColor: isConnected ? '#10b981' : color }}
                  >
                    {port.label} {isConnected && '✓'}
                  </span>
                  {selected && (
                    <div
                      className="absolute left-0 w-6 h-6 rounded-full border-2 animate-pulse pointer-events-none"
                      style={{
                        top: 0,
                        left: -6,
                        transform: 'translate(-50%, -50%)',
                        borderColor: isConnected ? '#10b981' : color,
                        backgroundColor: `${isConnected ? '#10b981' : color}33`
                      }}
                    />
                  )}
                </div>
              )
            })
          })()}
          
          {/* Single output handle for loss value */}
          <Handle
            type="source"
            position={Position.Right}
            className="w-3 h-3 !bg-red-500 transition-all"
            style={{
              right: -6,
              zIndex: 10
            }}
          />
          {selected && (
            <div
              className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2 w-6 h-6 rounded-full border-2 border-red-500 bg-red-500/20 animate-pulse pointer-events-none"
              style={{ right: -6 }}
            />
          )}
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
