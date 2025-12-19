import { useState, useEffect } from 'react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useModelBuilderStore } from '@/lib/store'
import * as Icons from '@phosphor-icons/react'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'

export default function ValidationErrorsPanel() {
  const [isOpen, setIsOpen] = useState(true)
  const [isVisible, setIsVisible] = useState(true)
  const validationErrors = useModelBuilderStore((state) => state.validationErrors)
  const setSelectedNodeId = useModelBuilderStore((state) => state.setSelectedNodeId)
  const nodes = useModelBuilderStore((state) => state.nodes)

  const errors = validationErrors.filter(e => e.type === 'error')
  const warnings = validationErrors.filter(e => e.type === 'warning')

  // Show panel when validation runs and there are errors/warnings
  // Reset visibility when new validation results come in
  useEffect(() => {
    if (validationErrors.length > 0) {
      setIsVisible(true)
    }
  }, [validationErrors])

  // Don't show panel if no errors/warnings or if user closed it
  if (validationErrors.length === 0 || !isVisible) {
    return null
  }

  const handleErrorClick = (nodeId?: string) => {
    if (nodeId) {
      setSelectedNodeId(nodeId)
      // Scroll to node (handled by Canvas component)
      const node = nodes.find(n => n.id === nodeId)
      if (node) {
        // Trigger a custom event to center the node
        window.dispatchEvent(new CustomEvent('centerNode', { detail: { nodeId } }))
      }
    }
  }

  return (
    <div className="absolute bottom-4 right-4 z-50 w-96 max-w-[calc(100vw-2rem)]">
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <Card className="shadow-lg border-2">
          <CollapsibleTrigger asChild>
            <Button
              variant="ghost"
              className="w-full flex items-center justify-between p-4 hover:bg-accent/50"
            >
              <div className="flex items-center gap-2">
                {errors.length > 0 ? (
                  <Icons.Warning size={20} weight="fill" className="text-red-500" />
                ) : (
                  <Icons.WarningCircle size={20} weight="fill" className="text-yellow-500" />
                )}
                <span className="font-semibold">
                  Validation Issues
                </span>
                <div className="flex gap-1">
                  {errors.length > 0 && (
                    <Badge variant="destructive" className="text-xs">
                      {errors.length} {errors.length === 1 ? 'Error' : 'Errors'}
                    </Badge>
                  )}
                  {warnings.length > 0 && (
                    <Badge variant="secondary" className="text-xs bg-yellow-500/20 text-yellow-700 dark:text-yellow-400">
                      {warnings.length} {warnings.length === 1 ? 'Warning' : 'Warnings'}
                    </Badge>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-1">
                <Icons.CaretDown
                  size={16}
                  className={`transition-transform ${isOpen ? 'rotate-180' : ''}`}
                />
                <div
                  role="button"
                  tabIndex={0}
                  className="h-6 w-6 hover:bg-accent ml-1 rounded inline-flex items-center justify-center cursor-pointer transition-colors"
                  onClick={(e) => {
                    e.stopPropagation()
                    setIsVisible(false)
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      e.stopPropagation()
                      setIsVisible(false)
                    }
                  }}
                >
                  <Icons.X size={14} />
                </div>
              </div>
            </Button>
          </CollapsibleTrigger>

          <CollapsibleContent>
            <ScrollArea className="max-h-64 px-4 pb-4">
              <div className="space-y-2">
                {/* Errors */}
                {errors.length > 0 && (
                  <div className="space-y-2">
                    <div className="text-xs font-semibold text-red-500 flex items-center gap-1">
                      <Icons.XCircle size={14} weight="fill" />
                      ERRORS
                    </div>
                    {errors.map((error, index) => (
                      <div
                        key={`error-${index}`}
                        className={`p-2 rounded-md bg-red-500/10 border border-red-500/20 text-sm ${
                          error.nodeId ? 'cursor-pointer hover:bg-red-500/20' : ''
                        }`}
                        onClick={() => handleErrorClick(error.nodeId)}
                      >
                        <div className="flex items-start gap-2">
                          <Icons.XCircle
                            size={16}
                            weight="fill"
                            className="text-red-500 mt-0.5 flex-shrink-0"
                          />
                          <div className="flex-1 min-w-0">
                            {(error.blockName || error.layerName) && (
                              <p className="text-xs font-semibold text-red-600 dark:text-red-400 mb-1">
                                {error.blockName && `Block: ${error.blockName}`}
                                {error.blockName && error.layerName && ' • '}
                                {error.layerName && `Layer: ${error.layerName}`}
                              </p>
                            )}
                            <p className="text-red-700 dark:text-red-400 break-words">
                              {error.message}
                            </p>
                            {error.nodeId && (
                              <p className="text-xs text-red-600/70 dark:text-red-400/70 mt-1">
                                Click to select node
                              </p>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Warnings */}
                {warnings.length > 0 && (
                  <div className="space-y-2 mt-3">
                    <div className="text-xs font-semibold text-yellow-600 dark:text-yellow-500 flex items-center gap-1">
                      <Icons.WarningCircle size={14} weight="fill" />
                      WARNINGS
                    </div>
                    {warnings.map((warning, index) => (
                      <div
                        key={`warning-${index}`}
                        className={`p-2 rounded-md bg-yellow-500/10 border border-yellow-500/20 text-sm ${
                          warning.nodeId ? 'cursor-pointer hover:bg-yellow-500/20' : ''
                        }`}
                        onClick={() => handleErrorClick(warning.nodeId)}
                      >
                        <div className="flex items-start gap-2">
                          <Icons.WarningCircle
                            size={16}
                            weight="fill"
                            className="text-yellow-600 dark:text-yellow-500 mt-0.5 flex-shrink-0"
                          />
                          <div className="flex-1 min-w-0">
                            <p className="text-yellow-700 dark:text-yellow-400 break-words">
                              {warning.message}
                            </p>
                            {warning.nodeId && (
                              <p className="text-xs text-yellow-600/70 dark:text-yellow-400/70 mt-1">
                                Click to select node
                              </p>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </ScrollArea>
          </CollapsibleContent>
        </Card>
      </Collapsible>
    </div>
  )
}
