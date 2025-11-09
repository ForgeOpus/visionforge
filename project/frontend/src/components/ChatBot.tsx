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

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  modifications?: any[]
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
  const scrollAreaRef = useRef<HTMLDivElement>(null)

  // Get workflow state from store
  const { nodes, edges, addNode, updateNode, removeNode, addEdge, removeEdge } = useModelBuilderStore()

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]')
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight
      }
    }
  }, [messages])

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    const currentInput = inputValue
    setInputValue('')
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
        workflowState
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
          const newNode = {
            id: `node-${Date.now()}`,
            type: 'block',
            position: position || { x: 100, y: 100 },
            data: {
              label: nodeType,
              nodeType,
              config: config || {},
              inputShape: null,
              outputShape: null
            }
          }
          addNode(newNode)
          toast.success('Node added', {
            description: `Added ${nodeType} to the workflow`
          })
          break
        }

        case 'remove_node': {
          const { nodeId } = details
          removeNode(nodeId)
          toast.success('Node removed', {
            description: 'Removed node from workflow'
          })
          break
        }

        case 'modify_node': {
          const { nodeId, config } = details
          const node = nodes.find(n => n.id === nodeId)
          if (node) {
            updateNode(nodeId, {
              ...node.data,
              config: { ...node.data.config, ...config }
            })
            toast.success('Node updated', {
              description: 'Node configuration updated'
            })
          }
          break
        }

        case 'add_connection': {
          const { source, target, sourceHandle, targetHandle } = details
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
          const { edgeId } = details
          removeEdge(edgeId)
          toast.success('Connection removed', {
            description: 'Removed connection from workflow'
          })
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
          <div className="p-4 border-b bg-card flex-shrink-0">
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
                        <div className="prose prose-sm dark:prose-invert max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0 text-sm break-words [&_pre]:overflow-x-auto [&_code]:break-words [&_p]:break-words">
                          <ReactMarkdown>{message.content}</ReactMarkdown>
                        </div>

                        {/* Modification Actions */}
                        {message.role === 'assistant' && message.modifications && message.modifications.length > 0 && (
                          <div className="mt-3 space-y-2 min-w-0">
                            <div className="text-xs font-semibold text-muted-foreground flex items-center gap-1">
                              <Icons.Wrench size={12} className="shrink-0" />
                              <span className="truncate">Suggested Modifications:</span>
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
            <div className="flex gap-2">
              <Input
                placeholder="Ask me anything..."
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={isLoading}
                className="flex-1"
              />
              <Button
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isLoading}
                size="icon"
              >
                <Icons.PaperPlaneRight size={18} />
              </Button>
            </div>
          </div>
        </Card>
      )}
    </>
  )
}
