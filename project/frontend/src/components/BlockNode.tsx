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

      <div className="p-3 space-y-2">
        <div className="flex items-center gap-2">
          <div
            className="p-1.5 rounded"
            style={{
              backgroundColor: definition.color,
              color: 'white'
            }}
          >
            <IconComponent size={16} weight="bold" />
          </div>
          <div className="flex-1">
            <div className="font-medium text-sm">
              {data.config?.label || definition.label}
            </div>
            <Badge
              variant="secondary"
              className="text-[10px] px-1 py-0 h-4"
            >
              {data.category}
            </Badge>
          </div>
        </div>

        {data.inputShape && (
          <div className="text-xs font-mono text-muted-foreground">
            In: {formatShape(data.inputShape.dims)}
          </div>
        )}

        {data.outputShape && (
          <div className="text-xs font-mono text-foreground font-medium">
            Out: {formatShape(data.outputShape.dims)}
          </div>
        )}

        {!data.outputShape && data.blockType !== 'input' && data.blockType !== 'dataloader' && data.blockType !== 'empty' && (
          <div className="text-xs text-orange-600">
            Configure parameters
          </div>
        )}
      </div>

      {/* Multiple output handles for DataLoader node with ground truth */}
      {data.blockType === 'dataloader' && data.config?.has_ground_truth ? (
        <>
          {/* Input Data Output */}
          <div className="absolute right-0 flex items-center" style={{ top: '33%', transform: 'translateY(-50%)' }}>
            <span className="text-[10px] text-blue-600 font-medium mr-2 bg-white/90 px-1.5 py-0.5 rounded border border-blue-200">
              Input
            </span>
            <Handle
              type="source"
              position={Position.Right}
              id="input-output"
              className="w-3 h-3 !bg-blue-500 transition-all border-2 border-white"
              style={{
                position: 'relative',
                right: -6,
                zIndex: 10
              }}
            />
          </div>

          {/* Ground Truth Output */}
          <div className="absolute right-0 flex items-center" style={{ top: '66%', transform: 'translateY(-50%)' }}>
            <span className="text-[10px] text-green-600 font-medium mr-2 bg-white/90 px-1.5 py-0.5 rounded border border-green-200">
              GT
            </span>
            <Handle
              type="source"
              position={Position.Right}
              id="ground-truth-output"
              className="w-3 h-3 !bg-green-500 transition-all border-2 border-white"
              style={{
                position: 'relative',
                right: -6,
                zIndex: 10
              }}
            />
          </div>

          {selected && (
            <>
              <div
                className="absolute right-0 w-6 h-6 rounded-full border-2 border-blue-500 bg-blue-500/20 animate-pulse pointer-events-none"
                style={{ top: '33%', right: -6, transform: 'translate(50%, -50%)' }}
              />
              <div
                className="absolute right-0 w-6 h-6 rounded-full border-2 border-green-500 bg-green-500/20 animate-pulse pointer-events-none"
                style={{ top: '66%', right: -6, transform: 'translate(50%, -50%)' }}
              />
            </>
          )}
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
