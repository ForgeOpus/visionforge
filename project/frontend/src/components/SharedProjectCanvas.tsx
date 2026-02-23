import { useState, useEffect, useRef, Suspense, lazy } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { Toaster } from 'sonner'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ThemeToggle } from '@/components/ThemeToggle'
import { CopySimple, Download, CaretDown, FilePy, Code, EyeSlash } from '@phosphor-icons/react'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useModelBuilderStore } from '@/lib/store'
import {
  fetchSharedProject,
  fetchSharedArchitecture,
  createProject,
  saveArchitecture,
  SharedProjectResponse,
} from '@/lib/projectApi'
import ConfigPanel from './ConfigPanel'
import { exportModel as apiExportModel } from '@/lib/api'
import { useAuth } from '@/contexts/AuthContext'
import LoginModal from './LoginModal'

const Canvas = lazy(() => import('./Canvas'))

function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary" />
    </div>
  )
}

export default function SharedProjectCanvas() {
  const { shareToken } = useParams<{ shareToken: string }>()
  const navigate = useNavigate()
  const { user } = useAuth()

  const {
    nodes,
    edges,
    groupDefinitions,
    selectedNodeId,
    setNodes,
    setEdges,
    loadGroupDefinitions,
    reset,
  } = useModelBuilderStore()

  const [projectMeta, setProjectMeta] = useState<SharedProjectResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [notFound, setNotFound] = useState(false)

  // Login gate state
  const [showLoginModal, setShowLoginModal] = useState(false)
  const [pendingAction, setPendingAction] = useState<'copy' | 'export' | null>(null)

  // Export dialog state (mirrors Header.tsx export dialog)
  const [isExportOpen, setIsExportOpen] = useState(false)
  const [exportCode, setExportCode] = useState<{
    model: string
    train: string
    dataset: string
    config: string
    zip: string
    filename: string
  } | null>(null)

  // A no-op ref for the palette drag handler (not used in read-only mode)
  const noopRegister = useRef((_: (blockType: string) => void) => {})

  // Load shared project on mount
  useEffect(() => {
    if (!shareToken) return

    const load = async () => {
      setIsLoading(true)
      try {
        const [meta, arch] = await Promise.all([
          fetchSharedProject(shareToken),
          fetchSharedArchitecture(shareToken),
        ])
        setProjectMeta(meta)
        setNodes(arch.nodes || [])
        setEdges(arch.edges || [])
        if (arch.groupDefinitions && arch.groupDefinitions.length > 0) {
          loadGroupDefinitions(arch.groupDefinitions)
        }
      } catch (error: unknown) {
        console.error('Failed to load shared project', error)

        const anyError = error as { status?: number; name?: string }
        const status = anyError?.status

        if (status === 404 || status === undefined) {
          setNotFound(true)
          toast.error('Shared link not found.')
        } else if (status === 401 || status === 403) {
          toast.error('You are not authorized to view this shared project.')
        } else if (typeof status === 'number' && status >= 500 && status < 600) {
          toast.error('Server error while loading shared project. Please try again later.')
        } else if (anyError?.name === 'TypeError') {
          toast.error('Network error while loading shared project. Check your connection and try again.')
        } else {
          toast.error('Unexpected error while loading shared project.')
        }
      } finally {
        setIsLoading(false)
      }
    }

    load()

    // Cleanup: reset store when leaving the page
    return () => {
      reset()
    }
  }, [shareToken])

  // After login completes, resume the pending action
  useEffect(() => {
    if (!user || !pendingAction) return
    setShowLoginModal(false)
    if (pendingAction === 'copy') {
      handleMakeCopy()
    } else if (pendingAction === 'export') {
      handleExport()
    }
    setPendingAction(null)
  }, [user, pendingAction, projectMeta, nodes, edges, groupDefinitions, navigate, handleMakeCopy, handleExport])

  const requireAuth = (action: 'copy' | 'export') => {
    if (!user) {
      setPendingAction(action)
      setShowLoginModal(true)
      return false
    }
    return true
  }

  const handleMakeCopy = async () => {
    if (!requireAuth('copy')) return
    if (!projectMeta) return

    const loadingToast = toast.loading('Copying project to your workspace...')
    try {
      const newProject = await createProject({
        name: `${projectMeta.name} (copy)`,
        description: projectMeta.description,
        framework: projectMeta.framework,
      })
      await saveArchitecture(newProject.id, nodes, edges, groupDefinitions)
      toast.dismiss(loadingToast)
      toast.success('Project copied!', {
        description: `"${newProject.name}" is now in your workspace`,
      })
      navigate(`/project/${newProject.id}`)
    } catch (error) {
      toast.dismiss(loadingToast)
      toast.error('Failed to copy project', {
        description: error instanceof Error ? error.message : 'Unknown error',
      })
    }
  }

  const handleExport = async () => {
    if (!requireAuth('export')) return
    if (!projectMeta) return

    if (nodes.length === 0) {
      toast.error('No blocks to export')
      return
    }

    const loadingToast = toast.loading('Generating code...')
    try {
      const result = await apiExportModel({
        nodes: nodes.map((node) => ({
          id: node.id,
          type: node.data.blockType,
          data: node.data,
          position: node.position,
        })),
        edges: edges.map((edge) => ({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          sourceHandle: edge.sourceHandle || '',
          targetHandle: edge.targetHandle || '',
        })),
        format: projectMeta.framework,
        projectName: projectMeta.name,
        groupDefinitions: Array.from(groupDefinitions.values()).map((def) => ({
          id: def.id,
          name: def.name,
          description: def.description,
          category: def.category,
          color: def.color,
          internal_structure: {
            nodes: def.internalNodes,
            edges: def.internalEdges,
            portMappings: def.portMappings,
          },
          createdAt: def.createdAt,
          updatedAt: def.updatedAt,
        })),
      })

      toast.dismiss(loadingToast)

      if (result.success && result.data) {
        setExportCode({
          model: result.data.files['model.py'],
          train: result.data.files['train.py'],
          dataset: result.data.files['dataset.py'],
          config: result.data.files['config.py'],
          zip: result.data.zip,
          filename: result.data.filename,
        })
        setIsExportOpen(true)
        toast.success(`${result.data.framework.toUpperCase()} code generated!`)
      } else {
        toast.error('Code generation failed', {
          description: typeof result.error === 'string' ? result.error : 'Unknown error',
        })
      }
    } catch (error) {
      toast.dismiss(loadingToast)
      toast.error('Code generation failed', {
        description: error instanceof Error ? error.message : 'Unknown error',
      })
    }
  }

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text)
    toast.success(`${label} copied to clipboard!`)
  }

  const downloadFile = (content: string, filename: string) => {
    const blob = new Blob([content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  if (isLoading) {
    return <LoadingSpinner />
  }

  if (notFound || !projectMeta) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen gap-4 bg-background text-foreground">
        <EyeSlash size={48} className="text-muted-foreground" />
        <h2 className="text-2xl font-semibold">Link not found</h2>
        <p className="text-muted-foreground max-w-sm text-center">
          This shared link is no longer active or does not exist. The owner may have disabled sharing.
        </p>
        <Button onClick={() => navigate('/')}>Go to VisionForge</Button>
      </div>
    )
  }

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-background">
      {/* Shared-view header */}
      <header className="h-16 border-b border-border bg-card px-6 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-4">
          <div
            className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => navigate('/')}
          >
            <img
              src={import.meta.env.VITE_LOGO_URL || '/logo_navbar.png'}
              alt="VisionForge Logo"
              className="h-10 w-auto"
            />
            <span className="text-xl font-semibold">VisionForge</span>
          </div>

          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span className="font-medium text-foreground">{projectMeta.name}</span>
            <Badge variant="secondary" className="text-xs">
              {projectMeta.framework}
            </Badge>
            {projectMeta.owner_display_name && (
              <span className="hidden sm:inline">
                · Shared by{' '}
                <span className="font-medium text-foreground">
                  {projectMeta.owner_display_name}
                </span>
              </span>
            )}
            <Badge variant="outline" className="text-xs text-muted-foreground">
              Read-only
            </Badge>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <ThemeToggle />

          <Button variant="outline" size="sm" onClick={handleMakeCopy}>
            <CopySimple size={16} className="mr-2" />
            Make a Copy
          </Button>

          {/* Export dropdown */}
          <Dialog open={isExportOpen} onOpenChange={setIsExportOpen}>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="default" size="sm" disabled={nodes.length === 0}>
                  <Download size={16} className="mr-2" />
                  Export
                  <CaretDown size={14} className="ml-1" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuLabel>Export Options</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={handleExport} className="gap-2 cursor-pointer">
                  <FilePy size={16} />
                  <div>
                    <div className="font-medium">
                      {projectMeta.framework.toUpperCase()} Code
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Generate model.py, train.py, dataset.py, config.py
                    </div>
                  </div>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Export dialog — identical to Header.tsx */}
            <DialogContent className="max-w-4xl max-h-[90vh] w-full overflow-hidden flex flex-col">
              <DialogHeader>
                <DialogTitle>
                  Export {projectMeta.framework.toUpperCase()} Code
                </DialogTitle>
                <DialogDescription>
                  Copy individual files or download all as ZIP
                </DialogDescription>
              </DialogHeader>
              {exportCode && (
                <div className="flex-1 flex flex-col min-h-0">
                  <div className="mb-3">
                    <Button
                      className="w-full"
                      onClick={() => {
                        const binaryString = atob(exportCode.zip)
                        const bytes = new Uint8Array(binaryString.length)
                        for (let i = 0; i < binaryString.length; i++) {
                          bytes[i] = binaryString.charCodeAt(i)
                        }
                        const blob = new Blob([bytes], { type: 'application/zip' })
                        const url = URL.createObjectURL(blob)
                        const a = document.createElement('a')
                        a.href = url
                        a.download = exportCode.filename
                        document.body.appendChild(a)
                        a.click()
                        document.body.removeChild(a)
                        URL.revokeObjectURL(url)
                        toast.success(`${exportCode.filename} downloaded!`)
                      }}
                    >
                      <Download size={16} className="mr-2" />
                      Download All Files (ZIP)
                    </Button>
                  </div>

                  <Tabs defaultValue="model" className="flex-1 flex flex-col min-h-0">
                    <TabsList className="w-full shrink-0">
                      <TabsTrigger value="model" className="flex-1">model.py</TabsTrigger>
                      <TabsTrigger value="train" className="flex-1">train.py</TabsTrigger>
                      <TabsTrigger value="dataset" className="flex-1">dataset.py</TabsTrigger>
                      <TabsTrigger value="config" className="flex-1">config.py</TabsTrigger>
                    </TabsList>
                    {(['model', 'train', 'dataset', 'config'] as const).map((key) => (
                      <TabsContent
                        key={key}
                        value={key}
                        className="mt-4 flex-1 flex flex-col min-h-0"
                      >
                        <div className="flex-1 w-full border rounded-md overflow-auto bg-muted">
                          <div className="min-w-max">
                            <pre className="text-xs font-mono p-4 whitespace-pre">
                              <code>{exportCode[key]}</code>
                            </pre>
                          </div>
                        </div>
                        <div className="flex gap-2 mt-3 shrink-0">
                          <Button
                            className="flex-1"
                            variant="outline"
                            onClick={() => copyToClipboard(exportCode[key], `${key}.py`)}
                          >
                            <Code size={16} className="mr-2" />
                            Copy
                          </Button>
                          <Button
                            className="flex-1"
                            variant="outline"
                            onClick={() => downloadFile(exportCode[key], `${key}.py`)}
                          >
                            <Download size={16} className="mr-2" />
                            Download
                          </Button>
                        </div>
                      </TabsContent>
                    ))}
                  </Tabs>
                </div>
              )}
            </DialogContent>
          </Dialog>
        </div>
      </header>

      {/* Read-only canvas + config panel */}
      <div className="flex-1 overflow-hidden flex">
        <Suspense fallback={<LoadingSpinner />}>
          <Canvas
            onDragStart={() => {}}
            onRegisterAddNode={noopRegister.current}
            readOnly={true}
          />
        </Suspense>
        {selectedNodeId && <ConfigPanel readOnly={true} />}
      </div>

      <Toaster position="bottom-right" richColors />

      {/* Login gate */}
      <LoginModal
        open={showLoginModal}
        onOpenChange={(open) => {
          setShowLoginModal(open)
          if (!open) setPendingAction(null)
        }}
        required={false}
      />
    </div>
  )
}
