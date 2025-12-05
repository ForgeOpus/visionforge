import { useState, useMemo } from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { getAllNodeDefinitions, getNodeDefinitionsByCategory, BackendFramework } from '@/lib/nodes/registry'
import { useModelBuilderStore } from '@/lib/store'
import { BlockDefinitionContextMenu } from './BlockDefinitionContextMenu'
import RenameBlockDialog from './RenameBlockDialog'
import DeleteBlockDialog from './DeleteBlockDialog'
import { toast } from 'sonner'
import * as Icons from '@phosphor-icons/react'
import Fuse from 'fuse.js'

interface BlockPaletteProps {
  onDragStart: (blockType: string) => void
  onBlockClick: (blockType: string) => void
  isCollapsed: boolean
}

export default function BlockPalette({ onDragStart, onBlockClick, isCollapsed }: BlockPaletteProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [contextMenu, setContextMenu] = useState<{
    x: number
    y: number
    definitionId: string
    definitionName: string
  } | null>(null)
  const [renameDialog, setRenameDialog] = useState<{
    definitionId: string
    currentName: string
  } | null>(null)
  const [deleteDialog, setDeleteDialog] = useState<{
    definitionId: string
    blockName: string
    instanceCount: number
  } | null>(null)

  const groupDefinitions = useModelBuilderStore((state) => state.groupDefinitions)
  const nodes = useModelBuilderStore((state) => state.nodes)
  const renameGroupDefinition = useModelBuilderStore((state) => state.renameGroupDefinition)
  const deleteGroupDefinition = useModelBuilderStore((state) => state.deleteGroupDefinition)
  const duplicateGroupDefinition = useModelBuilderStore((state) => state.duplicateGroupDefinition)

  const categories = [
    { key: 'input', label: 'Input & Data', icon: Icons.DownloadSimple },
    { key: 'basic', label: 'Base Layers', icon: Icons.SquaresFour },
    { key: 'activation', label: 'Activation Functions', icon: Icons.Lightning },
    { key: 'advanced', label: 'Advanced Layers', icon: Icons.CubeFocus },
    { key: 'merge', label: 'Operations', icon: Icons.Unite },
    { key: 'output', label: 'Output & Loss', icon: Icons.UploadSimple },
    { key: 'utility', label: 'Utility', icon: Icons.Wrench },
    { key: 'custom', label: 'Custom Blocks', icon: Icons.Package }
  ]

  // Prepare all blocks for fuzzy search - maintain category order
  const allBlocks = useMemo(() => {
    const categoryOrder = ['input', 'basic', 'activation', 'advanced', 'merge', 'output', 'utility', 'custom']
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
      description: node.metadata.description,
      isGroup: false
    }))

    // Add custom group blocks
    const groupBlocks = Array.from(groupDefinitions.values()).map(def => ({
      type: `group:${def.id}`,
      label: def.name,
      category: 'custom',
      color: def.color,
      icon: 'SquaresFour',
      description: def.description || `Custom block with ${def.internalNodes.length} nodes`,
      isGroup: true,
      groupDefinitionId: def.id
    }))

    blocks.push(...groupBlocks)

    return blocks
  }, [groupDefinitions])

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

  const handleContextMenu = (e: React.MouseEvent, block: any) => {
    if (!block.isGroup) return
    
    e.preventDefault()
    e.stopPropagation()
    
    setContextMenu({
      x: e.clientX,
      y: e.clientY,
      definitionId: block.groupDefinitionId,
      definitionName: block.label
    })
  }

  const handleRename = (definitionId: string) => {
    const definition = groupDefinitions.get(definitionId)
    if (!definition) return
    
    setRenameDialog({
      definitionId,
      currentName: definition.name
    })
  }

  const handleDuplicate = (definitionId: string) => {
    const definition = groupDefinitions.get(definitionId)
    const newId = duplicateGroupDefinition(definitionId)
    if (newId && definition) {
      toast.success('Block duplicated', {
        description: `Created copy of "${definition.name}"`
      })
    }
  }

  const handleDelete = (definitionId: string) => {
    const definition = groupDefinitions.get(definitionId)
    if (!definition) return
    
    // Count instances on canvas
    const instanceCount = nodes.filter(node => {
      if (node.data.blockType === 'group') {
        const groupData = node.data as any
        return groupData.groupDefinitionId === definitionId
      }
      return false
    }).length
    
    setDeleteDialog({
      definitionId,
      blockName: definition.name,
      instanceCount
    })
  }

  const renderBlockCard = (block: {
    type: string
    label: string
    category: string
    color: string
    icon: string
    description: string
    isGroup?: boolean
    groupDefinitionId?: string
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
        onContextMenu={(e) => handleContextMenu(e, block)}
      >
        <div className="flex items-center gap-2 min-w-0">
          <div
            className="p-1.5 rounded shrink-0"
            style={{
              backgroundColor: block.color,
              color: 'white',
              borderStyle: block.isGroup ? 'dashed' : 'solid',
              borderWidth: block.isGroup ? '2px' : '0'
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
      <>
        <div className="w-full bg-card h-full flex flex-col items-center relative overflow-hidden">
          {/* Scrollable Block Icons */}
          <ScrollArea className="flex-1 w-full min-h-0">
            <div className="py-2 space-y-1 flex flex-col items-center px-2">
              {allBlocks.map((block: any) => {
                const IconComponent = (Icons as any)[block.icon]

                // Debug: log if icon is missing
                if (!IconComponent && block.icon) {
                  console.warn(`Icon "${block.icon}" not found for block "${block.label}" (${block.type})`)
                }

                const FinalIcon = IconComponent || Icons.Cube

                return (
                  <button
                    key={block.type}
                    className="w-12 h-12 rounded flex items-center justify-center hover:bg-accent transition-colors cursor-pointer group relative shrink-0"
                    draggable
                    onDragStart={(e) => {
                      e.dataTransfer.effectAllowed = 'move'
                      handleDragStart(block.type)
                    }}
                    onClick={() => onBlockClick(block.type)}
                    onContextMenu={(e) => handleContextMenu(e, block)}
                    title={block.label}
                    style={{
                      backgroundColor: 'transparent'
                    }}
                  >
                    <div
                      className="w-8 h-8 rounded flex items-center justify-center"
                      style={{
                        backgroundColor: block.color,
                        color: 'white',
                        borderStyle: block.isGroup ? 'dashed' : 'solid',
                        borderWidth: block.isGroup ? '2px' : '0'
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

        {/* Context Menu */}
        {contextMenu && (
          <BlockDefinitionContextMenu
            x={contextMenu.x}
            y={contextMenu.y}
            definitionId={contextMenu.definitionId}
            definitionName={contextMenu.definitionName}
            instanceCount={nodes.filter(node => {
              if (node.data.blockType === 'group') {
                const groupData = node.data as any
                return groupData.groupDefinitionId === contextMenu.definitionId
              }
              return false
            }).length}
            onClose={() => setContextMenu(null)}
            onRename={handleRename}
            onDuplicate={handleDuplicate}
            onDelete={handleDelete}
          />
        )}

        {/* Rename Dialog */}
        {renameDialog && (
          <RenameBlockDialog
            isOpen={true}
            onClose={() => setRenameDialog(null)}
            onSave={(newName) => {
              renameGroupDefinition(renameDialog.definitionId, newName)
              toast.success('Block renamed', {
                description: `Renamed "${renameDialog.currentName}" to "${newName}"`
              })
              setRenameDialog(null)
            }}
            currentName={renameDialog.currentName}
            existingNames={Array.from(groupDefinitions.values()).map(def => def.name)}
          />
        )}

        {/* Delete Dialog */}
        {deleteDialog && (
          <DeleteBlockDialog
            isOpen={true}
            onClose={() => setDeleteDialog(null)}
            onConfirm={(cascade) => {
              deleteGroupDefinition(deleteDialog.definitionId, cascade)
              if (cascade && deleteDialog.instanceCount > 0) {
                toast.success('Block deleted', {
                  description: `Deleted "${deleteDialog.blockName}" and ${deleteDialog.instanceCount} instance(s) from canvas`
                })
              } else if (deleteDialog.instanceCount > 0) {
                toast.warning('Definition deleted', {
                  description: `"${deleteDialog.blockName}" deleted but ${deleteDialog.instanceCount} instance(s) remain on canvas with errors`
                })
              } else {
                toast.success('Block deleted', {
                  description: `Deleted "${deleteDialog.blockName}"`
                })
              }
              setDeleteDialog(null)
            }}
            blockName={deleteDialog.blockName}
            instanceCount={deleteDialog.instanceCount}
          />
        )}
      </>
    )
  }

  return (
    <>
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

      {/* Context Menu */}
      {contextMenu && (
        <BlockDefinitionContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          definitionId={contextMenu.definitionId}
          definitionName={contextMenu.definitionName}
          instanceCount={nodes.filter(node => {
            if (node.data.blockType === 'group') {
              const groupData = node.data as any
              return groupData.groupDefinitionId === contextMenu.definitionId
            }
            return false
          }).length}
          onClose={() => setContextMenu(null)}
          onRename={handleRename}
          onDuplicate={handleDuplicate}
          onDelete={handleDelete}
        />
      )}

      {/* Rename Dialog */}
      {renameDialog && (
        <RenameBlockDialog
          isOpen={true}
          onClose={() => setRenameDialog(null)}
          onSave={(newName) => {
            renameGroupDefinition(renameDialog.definitionId, newName)
            toast.success('Block renamed', {
              description: `Renamed "${renameDialog.currentName}" to "${newName}"`
            })
            setRenameDialog(null)
          }}
          currentName={renameDialog.currentName}
          existingNames={Array.from(groupDefinitions.values()).map(def => def.name)}
        />
      )}

      {/* Delete Dialog */}
      {deleteDialog && (
        <DeleteBlockDialog
          isOpen={true}
          onClose={() => setDeleteDialog(null)}
          onConfirm={(cascade) => {
            deleteGroupDefinition(deleteDialog.definitionId, cascade)
            if (cascade && deleteDialog.instanceCount > 0) {
              toast.success('Block deleted', {
                description: `Deleted "${deleteDialog.blockName}" and ${deleteDialog.instanceCount} instance(s) from canvas`
              })
            } else if (deleteDialog.instanceCount > 0) {
              toast.warning('Definition deleted', {
                description: `"${deleteDialog.blockName}" deleted but ${deleteDialog.instanceCount} instance(s) remain on canvas with errors`
              })
            } else {
              toast.success('Block deleted', {
                description: `Deleted "${deleteDialog.blockName}"`
              })
            }
            setDeleteDialog(null)
          }}
          blockName={deleteDialog.blockName}
          instanceCount={deleteDialog.instanceCount}
        />
      )}
    </>
  )
}
