import { useState, useMemo } from 'react'
import { ScrollArea } from '@visionforge/core/components/ui/scroll-area'
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@visionforge/core/components/ui/accordion'
import { Card } from '@visionforge/core/components/ui/card'
import { Input } from '@visionforge/core/components/ui/input'
import { getAllNodeDefinitions, getNodeDefinitionsByCategory, BackendFramework } from '@visionforge/core/nodes'
import * as Icons from '@phosphor-icons/react'
import Fuse from 'fuse.js'

interface BlockPaletteProps {
  onDragStart: (blockType: string) => void
  onBlockClick: (blockType: string) => void
  isCollapsed: boolean
}

export default function BlockPalette({ onDragStart, onBlockClick, isCollapsed }: BlockPaletteProps) {
  const [searchQuery, setSearchQuery] = useState('')

  const categories = [
    { key: 'input', label: 'Input & Data', icon: Icons.DownloadSimple },
    { key: 'basic', label: 'Base Layers', icon: Icons.SquaresFour },
    { key: 'activation', label: 'Activation Functions', icon: Icons.Lightning },
    { key: 'advanced', label: 'Advanced Layers', icon: Icons.CubeFocus },
    { key: 'merge', label: 'Operations', icon: Icons.Unite },
    { key: 'output', label: 'Output & Loss', icon: Icons.UploadSimple },
    { key: 'utility', label: 'Utility', icon: Icons.Wrench }
  ]

  // Prepare all blocks for fuzzy search - maintain category order
  const allBlocks = useMemo(() => {
    const categoryOrder = ['input', 'basic', 'activation', 'advanced', 'merge', 'output', 'utility']
    const nodes = getAllNodeDefinitions(BackendFramework.PyTorch)

    // Group by category
    const nodesByCategory = new Map<string, typeof nodes>()
    nodes.forEach(node => {
      const cat = node.metadata.category
      if (!nodesByCategory.has(cat)) {
        nodesByCategory.set(cat, [])
      }
      nodesByCategory.get(cat)!.push(node)
    })

    // Build ordered list
    const orderedNodes: typeof nodes = []
    categoryOrder.forEach(category => {
      const categoryNodes = nodesByCategory.get(category) || []
      orderedNodes.push(...categoryNodes)
    })

    // Add any remaining categories not in the order list
    nodesByCategory.forEach((nodes, category) => {
      if (!categoryOrder.includes(category)) {
        orderedNodes.push(...nodes)
      }
    })

    const blocks = orderedNodes.map(node => ({
      type: node.metadata.type,
      label: node.metadata.label,
      category: node.metadata.category,
      color: node.metadata.color,
      icon: node.metadata.icon,
      description: node.metadata.description
    }))

    // Debug: log all icons
    console.log('Block icons loaded:', blocks.map(b => `${b.label}: ${b.icon}`))

    return blocks
  }, [])

  // Setup fuzzy search
  const fuse = useMemo(() => {
    return new Fuse(allBlocks, {
      keys: ['label', 'description', 'type'],
      threshold: 0.3,
      includeScore: true
    })
  }, [allBlocks])

  // Filter blocks based on search
  const filteredBlocks = useMemo(() => {
    if (!searchQuery.trim()) {
      return null // Return null to show categorized view
    }

    const results = fuse.search(searchQuery)
    return results.map(result => result.item)
  }, [searchQuery, fuse])

  const handleDragStart = (type: string) => {
    (window as any).draggedBlockTypeGlobal = type
    onDragStart(type)
  }

  const renderBlockCard = (block: {
    type: string
    label: string
    category: string
    color: string
    icon: string
    description: string
  }) => {
    const IconComponent = (Icons as any)[block.icon]

    // Debug: log if icon is missing
    if (!IconComponent && block.icon) {
      console.warn(`Icon "${block.icon}" not found for block "${block.label}" (${block.type})`)
    }

    const FinalIcon = IconComponent || Icons.Cube

    return (
      <Card
        key={block.type}
        className="p-2 cursor-pointer hover:shadow-md hover:scale-[1.02] transition-all overflow-hidden"
        draggable
        onDragStart={(e) => {
          e.dataTransfer.effectAllowed = 'move'
          handleDragStart(block.type)
        }}
        onClick={() => onBlockClick(block.type)}
      >
        <div className="flex items-center gap-2 min-w-0">
          <div
            className="p-1.5 rounded shrink-0"
            style={{
              backgroundColor: block.color,
              color: 'white'
            }}
          >
            <FinalIcon size={14} weight="bold" />
          </div>
          <div className="flex-1 min-w-0 overflow-hidden">
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
  }

  if (isCollapsed) {
    return (
      <div className="w-full bg-card h-full flex flex-col items-center relative overflow-hidden">
        {/* Scrollable Block Icons */}
        <ScrollArea className="flex-1 w-full min-h-0">
          <div className="py-2 space-y-1 flex flex-col items-center px-2">
            {allBlocks.map((block) => {
              const IconComponent = (Icons as any)[block.icon]

              // Debug: log if icon is missing
              if (!IconComponent && block.icon) {
                console.warn(`Icon "${block.icon}" not found for block "${block.label}" (${block.type})`)
              }

              const FinalIcon = IconComponent || Icons.Cube

              return (
                <button
                  key={block.type}
                  className="w-12 h-12 rounded flex items-center justify-center hover:bg-accent transition-colors cursor-pointer group relative flex-shrink-0"
                  draggable
                  onDragStart={(e) => {
                    e.dataTransfer.effectAllowed = 'move'
                    handleDragStart(block.type)
                  }}
                  onClick={() => onBlockClick(block.type)}
                  title={block.label}
                  style={{
                    backgroundColor: 'transparent'
                  }}
                >
                  <div
                    className="w-8 h-8 rounded flex items-center justify-center"
                    style={{
                      backgroundColor: block.color,
                      color: 'white'
                    }}
                  >
                    <FinalIcon size={16} weight="bold" />
                  </div>

                  {/* Tooltip on hover */}
                  <div className="absolute left-full ml-2 px-2 py-1 bg-popover text-popover-foreground text-xs rounded shadow-md border border-border whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50">
                    {block.label}
                  </div>
                </button>
              )
            })}
          </div>
        </ScrollArea>
      </div>
    )
  }

  return (
    <div className="w-full bg-card h-full flex flex-col relative">
      <div className="p-3 border-b border-border sticky top-0 bg-card z-10">
        <div className="relative">
          <Icons.MagnifyingGlass
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground"
          />
          <Input
            type="text"
            placeholder="Search blocks..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9 h-9"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-1 hover:bg-accent rounded transition-colors"
            >
              <Icons.X size={14} />
            </button>
          )}
        </div>
      </div>

      <ScrollArea className="flex-1 overflow-y-auto">
        <div className="h-full">
          {filteredBlocks !== null ? (
            // Search results view
            <div className="p-2 space-y-2">
              {filteredBlocks.length > 0 ? (
                filteredBlocks.map((block) => renderBlockCard(block))
              ) : (
                <div className="text-center text-muted-foreground p-6">
                  <Icons.MagnifyingGlass size={32} className="mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No blocks found</p>
                  <p className="text-xs mt-1">Try a different search term</p>
                </div>
              )}
            </div>
          ) : (
            // Categorized view
            <Accordion type="multiple" defaultValue={['input', 'basic', 'activation']} className="px-2 py-2">
              {categories.map((category) => {
                const blocks = allBlocks.filter(b => b.category === category.key)
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
                        {blocks.map((block) => renderBlockCard(block))}
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                )
              })}
            </Accordion>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
