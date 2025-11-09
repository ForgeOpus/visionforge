import { memo } from 'react'
import { Handle, Position, NodeProps } from '@xyflow/react'
import { BlockData, BlockType } from '@/lib/types'
import { getNodeDefinition, BackendFramework } from '@/lib/nodes/registry'
import { useModelBuilderStore } from '@/lib/store'
import * as Icons from '@phosphor-icons/react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface BlockNodeProps {
  data: BlockData
  selected?: boolean
  id: string
}

const BlockNode = memo(({ data, selected, id }: BlockNodeProps) => {
  const nodeDef = getNodeDefinition(data.blockType as BlockType, BackendFramework.PyTorch)
  const validationErrors = useModelBuilderStore((state) => state.validationErrors)

  if (!nodeDef) return null

  const definition = nodeDef.metadata
  const IconComponent = (Icons as any)[definition.icon] || Icons.Cube

  // Check if this node has any validation errors
  const nodeErrors = validationErrors.filter((error) => error.nodeId === id && error.type === 'error')
  const hasErrors = nodeErrors.length > 0

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

      {data.blockType !== 'dataloader' && (
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
          const shapes = []

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

            const outlets = []
            const colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981']

            // Add input outlets
            for (let i = 0; i < numInputOutlets; i++) {
              const topPercent = spacing * (i + 1)
              const color = colors[i % colors.length]

              outlets.push(
                <div key={`input-${i}`} className="absolute right-0 flex items-center" style={{ top: `${topPercent}%`, transform: 'translateY(-50%)' }}>
                  <span className="text-[10px] font-medium mr-2 bg-card px-1.5 py-0.5 rounded border" style={{ color, borderColor: color }}>
                    In{numInputOutlets > 1 ? ` ${i + 1}` : ''}
                  </span>
                  <Handle
                    type="source"
                    position={Position.Right}
                    id={numInputOutlets > 1 ? `input-output-${i}` : 'input-output'}
                    className="w-3 h-3 transition-all border-2 border-card"
                    style={{
                      position: 'relative',
                      right: -6,
                      zIndex: 10,
                      backgroundColor: color
                    }}
                  />
                  {selected && (
                    <div
                      className="absolute right-0 w-6 h-6 rounded-full border-2 animate-pulse pointer-events-none"
                      style={{
                        top: 0,
                        right: -6,
                        transform: 'translate(50%, -50%)',
                        borderColor: color,
                        backgroundColor: `${color}33`
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

              outlets.push(
                <div key="gt" className="absolute right-0 flex items-center" style={{ top: `${topPercent}%`, transform: 'translateY(-50%)' }}>
                  <span className="text-[10px] text-green-600 font-medium mr-2 bg-card px-1.5 py-0.5 rounded border border-green-200">
                    GT
                  </span>
                  <Handle
                    type="source"
                    position={Position.Right}
                    id="ground-truth-output"
                    className="w-3 h-3 !bg-green-500 transition-all border-2 border-card"
                    style={{
                      position: 'relative',
                      right: -6,
                      zIndex: 10
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
      ) : (
        <>
          <Handle
            type="source"
            position={Position.Right}
            className="w-3 h-3 !bg-accent transition-all"
            style={{
              right: -6,
              zIndex: 10
            }}
          />
          {selected && (
            <div
              className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2 w-6 h-6 rounded-full border-2 border-accent bg-accent/20 animate-pulse pointer-events-none"
              style={{ right: -6 }}
            />
          )}
        </>
      )}
    </Card>
  )
})

BlockNode.displayName = 'BlockNode'

export default BlockNode
