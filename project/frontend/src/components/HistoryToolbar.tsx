import { ArrowCounterClockwise, ArrowClockwise } from '@phosphor-icons/react'
import { Button } from './ui/button'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip'
import { useModelBuilderStore } from '../lib/store'

export function HistoryToolbar() {
  const { undo, redo, canUndo, canRedo } = useModelBuilderStore()
  
  return (
    <div className="fixed top-20 right-4 z-50 flex items-center gap-2 bg-white/95 backdrop-blur-sm border border-gray-200 rounded-lg shadow-lg p-2">
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              onClick={undo}
              disabled={!canUndo()}
              className="h-8 w-8"
            >
              <ArrowCounterClockwise className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Undo (Ctrl+Z)</p>
          </TooltipContent>
        </Tooltip>
        
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              onClick={redo}
              disabled={!canRedo()}
              className="h-8 w-8"
            >
              <ArrowClockwise className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Redo (Ctrl+Y)</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
  )
}
