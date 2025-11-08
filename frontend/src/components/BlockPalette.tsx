import { ScrollArea } from '@/components/ui/scroll-area'
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion'
import { Card } from '@/components/ui/card'
import { getBlocksByCategory } from '@/lib/blockDefinitions'
import * as Icons from '@phosphor-icons/react'

interface BlockPaletteProps {
  onDragStart: (blockType: string) => void
}

export default function BlockPalette({ onDragStart }: BlockPaletteProps) {
  const categories = [
    { key: 'input', label: 'Input Layers', icon: Icons.ArrowDown },
    { key: 'basic', label: 'Basic Layers', icon: Icons.Lightning },
    { key: 'advanced', label: 'Advanced Layers', icon: Icons.Brain },
    { key: 'merge', label: 'Merge/Split', icon: Icons.GitMerge }
  ]

  const handleDragStart = (type: string) => {
    (window as any).draggedBlockTypeGlobal = type
    onDragStart(type)
  }

  return (
    <div className="w-64 bg-card border-r border-border h-full flex flex-col">
      <div className="p-4 border-b border-border">
        <h2 className="font-semibold text-lg">Block Palette</h2>
        <p className="text-xs text-muted-foreground mt-1">
          Drag blocks onto the canvas
        </p>
      </div>

      <ScrollArea className="flex-1 overflow-y-auto">
        <div className="h-full">
          <Accordion type="multiple" defaultValue={['input', 'basic']} className="px-2 py-2">
            {categories.map((category) => {
              const blocks = getBlocksByCategory(category.key)
              const CategoryIcon = category.icon

              return (
                <AccordionItem key={category.key} value={category.key}>
                  <AccordionTrigger className="text-sm font-medium">
                    <div className="flex items-center gap-2">
                      <CategoryIcon size={16} />
                      {category.label}
                    </div>
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-2 pt-2">
                      {blocks.map((block) => {
                        const IconComponent = (Icons as any)[block.icon] || Icons.Cube

                        return (
                          <Card
                            key={block.type}
                            className="p-3 cursor-grab active:cursor-grabbing hover:shadow-md transition-all"
                            draggable
                            onDragStart={(e) => {
                              e.dataTransfer.effectAllowed = 'move'
                              handleDragStart(block.type)
                            }}
                          >
                            <div className="flex items-center gap-2">
                              <div
                                className="p-1.5 rounded flex-shrink-0"
                                style={{
                                  backgroundColor: block.color,
                                  color: 'white'
                                }}
                              >
                                <IconComponent size={14} weight="bold" />
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="text-sm font-medium truncate">
                                  {block.label}
                                </div>
                                <div className="text-[10px] text-muted-foreground truncate">
                                  {block.description}
                                </div>
                              </div>
                            </div>
                          </Card>
                        )
                      })}
                    </div>
                  </AccordionContent>
                </AccordionItem>
              )
            })}
          </Accordion>
        </div>
      </ScrollArea>
    </div>
  )
}
