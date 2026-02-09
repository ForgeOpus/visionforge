import { memo } from 'react'
import { Handle, Position, NodeProps } from '@xyflow/react'
import { GroupBlockData, PortMapping } from '@/lib/types'
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

interface GroupBlockNodeProps {
  data: GroupBlockData
  selected?: boolean
  id: string
}

const GroupBlockNode = memo(({ data, selected, id }: GroupBlockNodeProps) => {
  const validationErrors = useModelBuilderStore((state) => state.validationErrors)
  const edges = useModelBuilderStore((state) => state.edges)
  const groupDefinitions = useModelBuilderStore((state) => state.groupDefinitions)
  const toggleGroupExpansion = useModelBuilderStore((state) => state.toggleGroupExpansion)

  const groupDef = groupDefinitions.get(data.groupDefinitionId)
  if (!groupDef) return null

  const nodeErrors = validationErrors.filter((error) => error.nodeId === id && error.type === 'error')
  const hasErrors = nodeErrors.length > 0

  // Check if this group instance has any configuration overrides
  const hasCustomizations = data.instanceConfigOverrides && Object.keys(data.instanceConfigOverrides).length > 0
  const customizationCount = hasCustomizations ? Object.keys(data.instanceConfigOverrides!).length : 0

  const isHandleConnected = (handleId: string, isTarget: boolean) => {
    return edges.some(edge => {
      if (isTarget) {
        return edge.target === id && (edge.targetHandle || 'default') === handleId
      } else {
        return edge.source === id && (edge.sourceHandle || 'default') === handleId
      }
    })
  }

  const inputPorts = groupDef.portMappings.filter(p => p.type === 'input')
  const outputPorts = groupDef.portMappings.filter(p => p.type === 'output')

  const getPortColor = (semantic: string) => {
    const colors: Record<string, string> = {
      'data': '#3b82f6',
      'labels': '#10b981',
      'loss': '#ef4444',
      'predictions': '#8b5cf6',
      'anchor': '#ec4899',
      'positive': '#f59e0b',
      'negative': '#f43f5e',
      'input1': '#06b6d4',
      'input2': '#8b5cf6',
      'weights': '#6366f1'
    }
    return colors[semantic] || '#3b82f6'
  }

  return (
    <Card
      className="min-w-[260px] w-[280px] transition-all duration-200 relative"
      style={{
        borderColor: selected ? 'var(--color-accent)' : groupDef.color,
        borderWidth: 3,
        borderStyle: 'dashed',
        boxShadow: selected ? '0 0 20px rgba(147, 51, 234, 0.4)' : '0 4px 6px rgba(0, 0, 0, 0.1)'
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

      {/* Repetition Badge */}
      {data.repetitionMetadata && (
        <div className="absolute -top-2 -left-2 z-20">
          <Badge
            variant="secondary"
            className="text-[10px] px-1.5 py-0.5 shadow-md"
            style={{
              backgroundColor: groupDef.color,
              color: 'white'
            }}
          >
            {data.repetitionMetadata.index + 1}/{data.repetitionMetadata.totalCount}
          </Badge>
        </div>
      )}

      {/* Customization Badge */}
      {hasCustomizations && !hasErrors && (
        <div className="absolute -top-2 -left-2 z-20" style={{ left: data.repetitionMetadata ? 'auto' : '-0.5rem', right: data.repetitionMetadata ? '-0.5rem' : 'auto' }}>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Badge
                  variant="secondary"
                  className="text-[10px] px-1.5 py-0.5 shadow-md bg-blue-500 hover:bg-blue-600 text-white cursor-help"
                >
                  <Icons.PencilSimple size={10} weight="fill" className="mr-0.5" />
                  {customizationCount}
                </Badge>
              </TooltipTrigger>
              <TooltipContent>
                <div className="text-xs">
                  <div className="font-semibold">Customized Instance</div>
                  <div>{customizationCount} internal node{customizationCount > 1 ? 's' : ''} customized</div>
                </div>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      )}

      {/* Action Buttons */}
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
                    toggleGroupExpansion(id)
                  }}
                >
                  {data.isExpanded ? (
                    <Icons.ArrowsIn size={14} />
                  ) : (
                    <Icons.ArrowsOut size={14} />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent>{data.isExpanded ? 'Collapse' : 'Expand'} (Space)</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      )}

      {/* Render input handles */}
      {inputPorts.map((port, index) => {
        const rangeStart = 70
        const rangeEnd = 90
        const spacing = (rangeEnd - rangeStart) / (inputPorts.length + 1)
        const topPercent = rangeStart + spacing * (index + 1)
        const color = getPortColor(port.semantic)
        const isConnected = isHandleConnected(port.externalPortId, true)

        return (
          <div key={port.externalPortId} className="absolute left-0 flex items-center" style={{ top: `${topPercent}%`, transform: 'translateY(-50%)' }}>
            <Handle
              type="target"
              position={Position.Left}
              id={port.externalPortId}
              className={`w-3 h-3 transition-all border-2 border-card ${isConnected ? 'ring-2 ring-offset-1 ring-green-400' : ''}`}
              style={{
                position: 'relative',
                left: -6,
                zIndex: 10,
                backgroundColor: isConnected ? '#10b981' : color,
                opacity: isConnected ? 1 : 0.8
              }}
            />
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span
                    className={`text-[10px] font-medium ml-2 bg-card px-1.5 py-0.5 rounded border ${isConnected ? 'opacity-60' : ''}`}
                    style={{
                      color: isConnected ? '#10b981' : color,
                      borderColor: isConnected ? '#10b981' : color
                    }}
                  >
                    {port.externalPortLabel} {isConnected && '✓'}
                  </span>
                </TooltipTrigger>
                <TooltipContent>
                  <div className="text-xs">
                    <div className="font-semibold">Internal Mapping:</div>
                    <div>Node: {port.internalNodeId.split('-')[0]}</div>
                    <div>Port: {port.internalPortId}</div>
                  </div>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
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
      })}

      <div className="p-3">
        <div className="flex items-center gap-2">
          <div
            className="p-1 rounded shrink-0"
            style={{
              backgroundColor: groupDef.color,
              color: 'white'
            }}
          >
            <Icons.SquaresFour size={14} weight="bold" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="font-semibold text-sm truncate leading-tight">
              {groupDef.name}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-1 mt-1">
          <Badge
            variant="secondary"
            className="text-[9px] px-1 py-0 h-3.5"
          >
            {groupDef.category}
          </Badge>
          <Badge
            variant="outline"
            className="text-[9px] px-1 py-0 h-3.5"
          >
            {groupDef.internalNodes.length} nodes
          </Badge>
        </div>

        {groupDef.description && (
          <div className="text-[10px] text-muted-foreground line-clamp-2 mt-1">
            {groupDef.description}
          </div>
        )}

        <div className="flex items-center gap-1 text-[10px] text-muted-foreground mt-1">
          <Icons.ArrowsIn size={12} />
          <span>{inputPorts.length} in</span>
          <span className="mx-1">•</span>
          <Icons.ArrowsOut size={12} />
          <span>{outputPorts.length} out</span>
        </div>
      </div>

      {/* Render output handles */}
      {outputPorts.map((port, index) => {
        const rangeStart = 70
        const rangeEnd = 90
        const spacing = (rangeEnd - rangeStart) / (outputPorts.length + 1)
        const topPercent = rangeStart + spacing * (index + 1)
        const color = getPortColor(port.semantic)
        const isConnected = isHandleConnected(port.externalPortId, false)

        return (
          <div key={port.externalPortId} className="absolute right-0 flex items-center" style={{ top: `${topPercent}%`, transform: 'translateY(-50%)' }}>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span
                    className={`text-[10px] font-medium mr-2 bg-card px-1.5 py-0.5 rounded border ${isConnected ? 'opacity-60' : ''}`}
                    style={{
                      color: isConnected ? '#10b981' : color,
                      borderColor: isConnected ? '#10b981' : color
                    }}
                  >
                    {port.externalPortLabel} {isConnected && '✓'}
                  </span>
                </TooltipTrigger>
                <TooltipContent>
                  <div className="text-xs">
                    <div className="font-semibold">Internal Mapping:</div>
                    <div>Node: {port.internalNodeId.split('-')[0]}</div>
                    <div>Port: {port.internalPortId}</div>
                  </div>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <Handle
              type="source"
              position={Position.Right}
              id={port.externalPortId}
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
      })}
    </Card>
  )
})

GroupBlockNode.displayName = 'GroupBlockNode'

export default GroupBlockNode
