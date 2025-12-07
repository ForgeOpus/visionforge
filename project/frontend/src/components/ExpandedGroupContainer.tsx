import { memo } from 'react'
import { NodeProps } from '@xyflow/react'
import { useModelBuilderStore } from '@/lib/store'
import * as Icons from '@phosphor-icons/react'
import { Button } from '@/components/ui/button'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'

interface ExpandedGroupContainerData {
  _expandedFrom: string
  _groupDefinitionId: string
  groupName: string
  groupColor: string
}

const ExpandedGroupContainer = memo(({ data, id }: NodeProps<ExpandedGroupContainerData>) => {
  const toggleGroupExpansion = useModelBuilderStore((state) => state.toggleGroupExpansion)

  const handleCollapse = (e: React.MouseEvent) => {
    e.stopPropagation()
    // Use the _expandedFrom ID to collapse the group
    toggleGroupExpansion(data._expandedFrom)
  }

  return (
    <div
      className="w-full h-full rounded-lg pointer-events-none"
      style={{
        border: `2px dashed ${data.groupColor}`,
        opacity: 0.6,
      }}
    >
      {/* Collapse button in top right */}
      <div className="absolute -top-3 -right-3 pointer-events-auto z-50">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="default"
                size="icon"
                className="h-8 w-8 shadow-lg"
                style={{
                  backgroundColor: data.groupColor,
                  color: 'white',
                }}
                onClick={handleCollapse}
              >
                <Icons.ArrowsIn size={16} weight="bold" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Collapse {data.groupName}</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      {/* Group name label in top left */}
      <div
        className="absolute -top-3 -left-3 px-2 py-1 rounded text-xs font-semibold shadow-md"
        style={{
          backgroundColor: data.groupColor,
          color: 'white',
        }}
      >
        {data.groupName}
      </div>
    </div>
  )
})

ExpandedGroupContainer.displayName = 'ExpandedGroupContainer'

export default ExpandedGroupContainer
