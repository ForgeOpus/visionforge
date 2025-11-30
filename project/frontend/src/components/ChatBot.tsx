import { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Card } from '@/components/ui/card'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import * as Icons from '@phosphor-icons/react'
import ReactMarkdown from 'react-markdown'
import { sendChatMessage } from '@/lib/api'
import { toast } from 'sonner'
import { useModelBuilderStore } from '@/lib/store'
import { getNodeDefinition, BackendFramework } from '@/lib/nodes/registry'
import { BlockType } from '@/lib/types'
import { useApiKeys } from '@/contexts/ApiKeyContext'
import ApiKeyModal from './ApiKeyModal'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  modifications?: any[]
  attachedFile?: {
    name: string
    type: string
    geminiUri?: string
  }
}

export default function ChatBot() {
  const [isOpen, setIsOpen] = useState(false)
  const [modificationMode, setModificationMode] = useState(false)
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: 'Hello! I\'m your VisionForge assistant. How can I help you build your neural network today?\n\nToggle **Modification Mode** above to allow me to suggest changes to your workflow.',
      timestamp: new Date()
    }
  ])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [isUploadingFile, setIsUploadingFile] = useState(false)
  const [showApiKeyModal, setShowApiKeyModal] = useState(false)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // API Key management
  const {
    geminiApiKey,
    anthropicApiKey,
    requiresApiKey,
    hasRequiredKey,
    provider
  } = useApiKeys()

  // Get workflow state from store
  const { nodes, edges, addNode, updateNode, removeNode, duplicateNode, addEdge, removeEdge } = useModelBuilderStore()

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]')
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight
      }
    }
  }, [messages])

  // Show API key modal when chat opens if keys are required
  useEffect(() => {
    if (isOpen && requiresApiKey && !hasRequiredKey()) {
      setShowApiKeyModal(true)
    }
  }, [isOpen, requiresApiKey, hasRequiredKey])

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      // Validate file type
      const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'application/pdf', 'text/plain']
      if (!validTypes.includes(file.type)) {
        toast.error('Invalid file type', {
          description: 'Only images (PNG, JPG, WEBP), PDFs, and text files are supported'
        })
        return
      }

      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        toast.error('File too large', {
          description: 'Maximum file size is 10MB'
        })
        return
      }

      setUploadedFile(file)
      toast.success('File attached', {
        description: file.name
      })
    }
  }

  const removeAttachedFile = () => {
    setUploadedFile(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleSendMessage = async () => {
    if ((!inputValue.trim() && !uploadedFile) || isLoading) return

    const currentFile = uploadedFile
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue || (currentFile ? `[Attached file: ${currentFile.name}]` : ''),
      timestamp: new Date(),
      attachedFile: currentFile ? {
        name: currentFile.name,
        type: currentFile.type
      } : undefined
    }

    setMessages(prev => [...prev, userMessage])
    const currentInput = inputValue
    setInputValue('')
    setUploadedFile(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
    setIsLoading(true)

    try {
      // Prepare workflow state for context
      const workflowState = {
        nodes: nodes.map(node => ({
          id: node.id,
          type: node.type,
          data: node.data,
          position: node.position
        })),
        edges: edges.map(edge => ({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          sourceHandle: edge.sourceHandle,
          targetHandle: edge.targetHandle
        }))
      }

      // Send message to backend API with workflow context
      const response = await sendChatMessage(
        currentInput,
        messages,
        modificationMode,
        workflowState,
        currentFile || undefined,
        { geminiApiKey, anthropicApiKey }
      )

      if (response.success && response.data) {
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: response.data.response,
          timestamp: new Date(),
          modifications: response.data.modifications || undefined
        }
        setMessages(prev => [...prev, assistantMessage])

        // If modifications were suggested, show notification
        if (response.data.modifications && response.data.modifications.length > 0) {
          toast.info('Workflow modifications suggested', {
            description: 'Check the chat for suggested changes to your workflow'
          })
        }
      } else {
        // Show error message
        const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `I apologize, but I encountered an error: ${response.error || 'Unknown error'}. Please try again.`,
          timestamp: new Date()
        }
        setMessages(prev => [...prev, errorMessage])
        toast.error('Failed to get response', {
          description: response.error
        })
      }
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'I apologize, but I\'m having trouble connecting to the server. Please check your connection and try again.',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
      toast.error('Connection error', {
        description: error instanceof Error ? error.message : 'Failed to send message'
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const applyModification = (modification: any) => {
    try {
      const { action, details } = modification

      switch (action) {
        case 'add_node': {
          const { nodeType, config, position } = details

          // Get the node definition to populate proper metadata
          const nodeDef = getNodeDefinition(nodeType as BlockType, BackendFramework.PyTorch)
          if (!nodeDef) {
            toast.error('Invalid node type', {
              description: `Node type '${nodeType}' is not recognized`
            })
            return
          }

          // Create properly structured node with all metadata (matching drag-and-drop format)
          const newNode = {
            id: `${nodeType}-${Date.now()}`,
            type: 'custom',
            position: position || { x: 100, y: 100 },
            data: {
              blockType: nodeDef.metadata.type,
              label: nodeDef.metadata.label,
              category: nodeDef.metadata.category,
              config: config || {},
              inputShape: undefined,
              outputShape: undefined
            }
          }

          // Apply default config values from schema
          nodeDef.configSchema.forEach((field) => {
            if (field.default !== undefined && !config?.[field.name]) {
              newNode.data.config[field.name] = field.default
            }
          })

          // Merge provided config
          if (config) {
            newNode.data.config = { ...newNode.data.config, ...config }
          }

          addNode(newNode)

          // Infer dimensions after adding node
          setTimeout(() => {
            useModelBuilderStore.getState().inferDimensions()
          }, 0)
          toast.success('Node added', {
            description: `Added ${nodeDef.metadata.label} to the workflow`
          })
          break
        }

        case 'remove_node': {
          const { id, nodeId } = details
          const targetNodeId = id || nodeId
          if (!targetNodeId) {
            toast.error('Invalid node ID', {
              description: 'Node ID is required to remove a node'
            })
            return
          }
          removeNode(targetNodeId)
          toast.success('Node removed', {
            description: 'Removed node from workflow'
          })
          break
        }

        case 'duplicate_node': {
          const { id, nodeId } = details
          const targetNodeId = id || nodeId
          if (!targetNodeId) {
            toast.error('Invalid node ID', {
              description: 'Node ID is required to duplicate a node'
            })
            return
          }
          duplicateNode(targetNodeId)
          toast.success('Node duplicated', {
            description: 'Created a copy of the node'
          })
          break
        }

        case 'modify_node': {
          const { id, nodeId, config, position, label } = details
          const targetNodeId = id || nodeId
          const node = nodes.find(n => n.id === targetNodeId)
          if (!node) {
            toast.error('Node not found', {
              description: `Node with ID '${targetNodeId}' does not exist`
            })
            return
          }
          
          let updated = false
          
          // If position is being updated, we need to update the node directly in the store
          if (position) {
            useModelBuilderStore.setState((state) => ({
              nodes: state.nodes.map((n) =>
                n.id === targetNodeId
                  ? { ...n, position }
                  : n
              )
            }))
            toast.success('Node moved', {
              description: 'Node position updated'
            })
            updated = true
          }
          
          // If config or label is being updated, use the updateNode function
          if (config || label) {
            const updates: any = { ...node.data }
            if (config) {
              updates.config = { ...node.data.config, ...config }
            }
            if (label) {
              updates.label = label
            }
            updateNode(targetNodeId, updates)
            const description = label ? 'Node label updated' : 'Node configuration updated'
            toast.success('Node updated', {
              description
            })
            updated = true
          }
          
          if (!updated) {
            toast.warning('No changes made', {
              description: 'No valid fields provided for modification'
            })
          }
          break
        }

        case 'add_connection': {
          const { source, target, sourceHandle, targetHandle } = details
          if (!source || !target) {
            toast.error('Invalid connection', {
              description: 'Both source and target are required'
            })
            return
          }
          addEdge({
            id: `edge-${Date.now()}`,
            source,
            target,
            sourceHandle: sourceHandle || null,
            targetHandle: targetHandle || null
          })
          toast.success('Connection added', {
            description: 'Added new connection between nodes'
          })
          break
        }

        case 'remove_connection': {
          const { id, edgeId, source, target } = details
          
          // Support both direct ID removal and source/target based removal
          if (id || edgeId) {
            const targetEdgeId = id || edgeId
            removeEdge(targetEdgeId)
            toast.success('Connection removed', {
              description: 'Removed connection from workflow'
            })
          } else if (source && target) {
            // Find edge by source and target
            const edge = edges.find(e => e.source === source && e.target === target)
            if (edge) {
              removeEdge(edge.id)
              toast.success('Connection removed', {
                description: 'Removed connection from workflow'
              })
            } else {
              toast.error('Connection not found', {
                description: `No connection found between ${source} and ${target}`
              })
            }
          } else {
            toast.error('Invalid connection removal', {
              description: 'Either edge ID or source/target pair is required'
            })
          }
          break
        }

        default:
          toast.warning('Unknown modification type', {
            description: `Action '${action}' is not supported`
          })
      }
    } catch (error) {
      toast.error('Failed to apply modification', {
        description: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }

  return (
    <>
      {/* Floating Chat Button */}
      {!isOpen && (
        <Button
          size="icon"
          className="fixed bottom-6 right-6 h-14 w-14 rounded-full shadow-lg z-50 hover:scale-110 transition-transform"
          onClick={() => setIsOpen(true)}
        >
          <Icons.ChatCircleDots size={24} />
        </Button>
      )}

      {/* Chat Panel - Right Side */}
      {isOpen && (
        <Card className="fixed right-0 top-0 h-full w-[400px] z-40 flex flex-col shadow-2xl border-l rounded-none animate-in slide-in-from-right duration-300 overflow-hidden">
          {/* Header */}
          <div className="p-2 border-b bg-card flex-shrink-0">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Icons.Robot size={24} className="text-primary" />
                <h3 className="font-semibold text-lg">VisionForge Assistant</h3>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsOpen(false)}
                className="h-8 w-8 hover:bg-destructive/10"
              >
                <Icons.X size={20} />
              </Button>
            </div>

            {/* Modification Mode Toggle */}
            <div className="flex items-center justify-between p-2 rounded-md bg-muted/50">
              <div className="flex items-center gap-2">
                <Icons.Wrench size={16} className={modificationMode ? 'text-primary' : 'text-muted-foreground'} />
                <Label htmlFor="modification-mode" className="text-sm cursor-pointer">
                  Modification Mode
                </Label>
              </div>
              <Switch
                id="modification-mode"
                checked={modificationMode}
                onCheckedChange={setModificationMode}
              />
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              {modificationMode
                ? 'AI can suggest changes to your workflow'
                : 'AI will only answer questions (no workflow changes)'}
            </p>
          </div>

          {/* Messages Area */}
          <ScrollArea className="flex-1 min-h-0 px-4" ref={scrollAreaRef}>
            <div className="space-y-4 py-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[85%] rounded-lg p-3 ${
                      message.role === 'user'
                        ? 'bg-primary text-primary-foreground ml-auto'
                        : 'bg-muted'
                    }`}
                  >
                    <div className="flex items-start gap-2 min-w-0">
                      {message.role === 'assistant' && (
                        <Icons.Robot size={18} className="shrink-0 mt-0.5" />
                      )}
                      <div className="flex-1 min-w-0 overflow-hidden">
                        {/* Show attached file if present */}
                        {message.attachedFile && (
                          <div className="mb-2 p-2 bg-background/50 border rounded-md flex items-center gap-2">
                            <Icons.Paperclip size={14} className="shrink-0" />
                            <span className="text-xs font-medium truncate">{message.attachedFile.name}</span>
                          </div>
                        )}

                        <div className="prose prose-sm dark:prose-invert max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0 text-sm break-words [&_pre]:overflow-x-auto [&_pre]:whitespace-pre-wrap [&_code]:break-words [&_code]:whitespace-pre-wrap [&_p]:break-words">
                          <ReactMarkdown>{message.content}</ReactMarkdown>
                        </div>

                        {/* Modification Actions */}
                        {message.role === 'assistant' && message.modifications && message.modifications.length > 0 && (
                          <div className="mt-3 space-y-2 min-w-0">
                            <div className="text-xs font-semibold text-muted-foreground flex items-center gap-1">
                              <Icons.Wrench size={12} className="shrink-0" />
                              <span className="truncate">Suggested Modifications ({message.modifications.length}):</span>
                            </div>
                            {message.modifications.map((mod, idx) => (
                              <div key={idx} className="bg-background/50 border rounded p-2 space-y-1 min-w-0">
                                <div className="text-xs font-medium break-words">
                                  {mod.action.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
                                </div>
                                {mod.explanation && (
                                  <div className="text-xs text-muted-foreground break-words">
                                    {mod.explanation}
                                  </div>
                                )}
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="w-full text-xs h-7"
                                  onClick={() => applyModification(mod)}
                                >
                                  <Icons.Play size={12} className="mr-1" />
                                  Apply Change
                                </Button>
                              </div>
                            ))}
                            
                            {/* Apply All Button - shown when multiple modifications exist */}
                            {message.modifications.length > 1 && (
                              <Button
                                size="sm"
                                className="w-full text-xs h-8 bg-primary hover:bg-primary/90"
                                onClick={async () => {
                                  let successCount = 0
                                  let failCount = 0
                                  
                                  for (const mod of message.modifications || []) {
                                    try {
                                      applyModification(mod)
                                      successCount++
                                      // Small delay between modifications to avoid race conditions
                                      await new Promise(resolve => setTimeout(resolve, 50))
                                    } catch (error) {
                                      failCount++
                                      console.error('Failed to apply modification:', error)
                                    }
                                  }
                                  
                                  if (failCount === 0) {
                                    toast.success('All changes applied', {
                                      description: `Successfully applied ${successCount} modifications`
                                    })
                                  } else {
                                    toast.warning('Some changes failed', {
                                      description: `Applied ${successCount} modifications, ${failCount} failed`
                                    })
                                  }
                                }}
                              >
                                <Icons.CheckCircle size={14} className="mr-1.5" />
                                Apply All {message.modifications.length} Changes
                              </Button>
                            )}
                          </div>
                        )}
                      </div>
                      {message.role === 'user' && (
                        <Icons.User size={18} className="shrink-0 mt-0.5" />
                      )}
                    </div>
                    <div
                      className={`text-[10px] mt-1.5 ${
                        message.role === 'user'
                          ? 'text-primary-foreground/70'
                          : 'text-muted-foreground'
                      }`}
                    >
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-muted rounded-lg p-3 max-w-[85%]">
                    <div className="flex items-center gap-2">
                      <Icons.Robot size={18} />
                      <div className="flex gap-1">
                        <span className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                        <span className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                        <span className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>

          {/* Input Area */}
          <div className="p-4 border-t bg-card flex-shrink-0">
            {/* Attached file preview */}
            {uploadedFile && (
              <div className="mb-2 p-2 bg-muted rounded-md flex items-center justify-between">
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <Icons.Paperclip size={16} className="text-muted-foreground shrink-0" />
                  <span className="text-sm truncate">{uploadedFile.name}</span>
                  <span className="text-xs text-muted-foreground shrink-0">
                    ({(uploadedFile.size / 1024).toFixed(1)} KB)
                  </span>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 shrink-0"
                  onClick={removeAttachedFile}
                >
                  <Icons.X size={14} />
                </Button>
              </div>
            )}

            <div className="flex gap-2">
              {/* Hidden file input */}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/png,image/jpeg,image/jpg,image/webp,application/pdf,text/plain"
                onChange={handleFileUpload}
                className="hidden"
              />

              {/* File attachment button */}
              <Button
                variant="ghost"
                size="icon"
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading || !!uploadedFile}
                title="Attach file"
              >
                <Icons.Paperclip size={18} />
              </Button>

              <Input
                placeholder={uploadedFile ? "Add a message (optional)..." : "Ask me anything..."}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={isLoading}
                className="flex-1"
              />
              <Button
                onClick={handleSendMessage}
                disabled={(!inputValue.trim() && !uploadedFile) || isLoading}
                size="icon"
              >
                <Icons.PaperPlaneRight size={18} />
              </Button>
            </div>
          </div>
        </Card>
      )}

      {/* API Key Modal */}
      <ApiKeyModal
        open={showApiKeyModal}
        onOpenChange={setShowApiKeyModal}
        required={requiresApiKey && !hasRequiredKey()}
      />
    </>
  )
}
