import { useState, useRef, useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useModelBuilderStore } from '@visionforge/core/store'
import { Button } from '@visionforge/core/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle
} from '@visionforge/core/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@visionforge/core/components/ui/dropdown-menu'
import { Input } from '@visionforge/core/components/ui/input'
import { Label } from '@visionforge/core/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@visionforge/core/components/ui/tabs'
import { ScrollArea } from '@visionforge/core/components/ui/scroll-area'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@visionforge/core/components/ui/select'
import { Textarea } from '@visionforge/core/components/ui/textarea'
import { Plus, Download, FloppyDisk, CaretDown, Code, CheckCircle, GitBranch, Upload, FileCode, FilePy, GearSix, Trash, Info, PencilSimple, Warning } from '@phosphor-icons/react'
import { toast } from 'sonner'
import { ThemeToggle } from '@/components/ThemeToggle'
import api from '@/lib/api'
import { exportToJSON, importFromJSON, downloadJSON, readJSONFile } from '@/lib/exportImport'
import * as projectApi from '@/lib/projectApi'

export default function Header() {
  const navigate = useNavigate()
  const { projectId } = useParams<{ projectId: string }>()
  const { currentProject, nodes, edges, saveProject, loadProject, validateArchitecture, setNodes, setEdges } = useModelBuilderStore()

  const [projects, setProjects] = useState<projectApi.ProjectResponse[]>([])

  const [isNewProjectOpen, setIsNewProjectOpen] = useState(false)
  const [isExportOpen, setIsExportOpen] = useState(false)
  const [isManageProjectOpen, setIsManageProjectOpen] = useState(false)
  const [managingProject, setManagingProject] = useState<projectApi.ProjectResponse | null>(null)
  const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState(false)
  const [isSaveAsDialogOpen, setIsSaveAsDialogOpen] = useState(false)

  const [newProjectName, setNewProjectName] = useState('')
  const [newProjectDesc, setNewProjectDesc] = useState('')
  const [newProjectFramework, setNewProjectFramework] = useState<'pytorch' | 'tensorflow'>('pytorch')

  const [saveAsProjectName, setSaveAsProjectName] = useState('')
  const [saveAsProjectDesc, setSaveAsProjectDesc] = useState('')

  const [editProjectName, setEditProjectName] = useState('')
  const [editProjectDesc, setEditProjectDesc] = useState('')

  const [exportCode, setExportCode] = useState<{model: string, train: string, dataset: string, config: string, zip: string, filename: string} | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)

  // Load projects list on mount
  useEffect(() => {
    loadProjectsList()
  }, [])

  const loadProjectsList = async () => {
    // Loading projects
    try {
      const projectsList = await projectApi.fetchProjects()
      setProjects(projectsList)
    } catch (error) {
      console.error('Failed to load projects:', error)
      toast.error('Failed to load projects list')
    } finally {
      // Done loading
    }
  }

  const handleCreateProject = async () => {
    if (!newProjectName.trim()) {
      toast.error('Please enter a project name')
      return
    }

    try {
      const backendProject = await projectApi.createProject({
        name: newProjectName,
        description: newProjectDesc,
        framework: newProjectFramework
      })

      // Load the project in the store
      const projectData = projectApi.convertToFrontendProject(backendProject, [], [])
      loadProject(projectData)

      setIsNewProjectOpen(false)
      setNewProjectName('')
      setNewProjectDesc('')

      toast.success('Project created!')

      // Reload projects list
      await loadProjectsList()

      // Navigate to the new project
      navigate(`/project/${backendProject.id}`)
    } catch (error) {
      toast.error('Failed to create project', {
        description: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }

  const handleSaveAsProject = async () => {
    if (!saveAsProjectName.trim()) {
      toast.error('Please enter a project name')
      return
    }

    const loadingToast = toast.loading('Creating and saving project...')

    try {
      // Create project on backend
      const backendProject = await projectApi.createProject({
        name: saveAsProjectName,
        description: saveAsProjectDesc,
        framework: 'pytorch'
      })

      // Save architecture immediately to the newly created project
      await projectApi.saveArchitecture(backendProject.id, nodes, edges)

      // Fetch the saved project with architecture
      const { nodes: savedNodes, edges: savedEdges } = await projectApi.loadArchitecture(backendProject.id)
      const projectData = projectApi.convertToFrontendProject(backendProject, savedNodes, savedEdges)

      // Load into store
      loadProject(projectData)

      // Close dialog and reset fields
      setIsSaveAsDialogOpen(false)
      setSaveAsProjectName('')
      setSaveAsProjectDesc('')

      toast.dismiss(loadingToast)
      toast.success('Project saved to browser storage', {
        description: 'Your design is stored locally in your browser'
      })

      // Reload projects list
      await loadProjectsList()

      // Navigate to the new project
      navigate(`/project/${backendProject.id}`, { replace: true })
    } catch (error) {
      toast.dismiss(loadingToast)
      toast.error('Failed to create project', {
        description: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }

  const handleSaveProject = async () => {
    if (nodes.length === 0) {
      toast.error('No architecture to save')
      return
    }

    try {
      let project = currentProject
      let projectIdToSave = project?.id

      // Check if project ID is a timestamp (invalid) - treat as no project
      const isTimestampId = project?.id && /^\d{13}$/.test(project.id)

      // If no project exists or project has invalid timestamp ID, show dialog to create one
      if (!project || isTimestampId) {
        // Generate default name suggestion
        const timestamp = new Date().toLocaleString('en-US', {
          month: 'short',
          day: 'numeric',
          hour: 'numeric',
          minute: '2-digit',
          hour12: true
        })
        setSaveAsProjectName(`Untitled Project - ${timestamp}`)
        setSaveAsProjectDesc('')
        setIsSaveAsDialogOpen(true)
        return
      }

      // Save to local storage for existing project
      if (projectIdToSave) {
        await projectApi.saveArchitecture(projectIdToSave, nodes, edges)
      }

      // Save to local store
      saveProject()

      toast.success('Project saved to browser storage', {
        description: 'Your design is stored locally in your browser'
      })

      // Reload projects list
      loadProjectsList()
    } catch (error) {
      toast.dismiss()
      toast.error('Failed to save project', {
        description: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }

  const handleLoadProject = async (project: projectApi.ProjectResponse) => {
    try {
      // Fetch full project details (unused for now but may be needed later)
      await projectApi.fetchProject(project.id)

      // Navigate to project URL
      navigate(`/project/${project.id}`)

      toast.success(`Loaded "${project.name}"`)
    } catch (error) {
      toast.error('Failed to load project', {
        description: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }

  const handleExportCode = async () => {
    const errors = validateArchitecture()
    const criticalErrors = errors.filter((e) => e.type === 'error')

    if (criticalErrors.length > 0) {
      toast.error('Cannot export: Architecture has errors', {
        description: `Fix ${criticalErrors.length} error(s) first`
      })
      return
    }

    if (nodes.length === 0) {
      toast.error('Cannot export: No blocks in architecture')
      return
    }

    if (!currentProject) {
      toast.error('Cannot export: No active project')
      return
    }

    try {
      toast.loading('Generating code...')

      // Call backend API with framework selection
      const result = await api.exportModel({
        nodes: nodes.map(node => ({
          id: node.id,
          type: node.data.blockType,
          data: node.data,
          position: node.position
        })),
        edges: edges.map(edge => ({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          sourceHandle: edge.sourceHandle || '',
          targetHandle: edge.targetHandle || ''
        })),
        format: currentProject.framework as 'pytorch' | 'tensorflow',
        projectName: currentProject.name
      })

      toast.dismiss()

      if (result.success && result.data) {
        setExportCode({
          model: result.data.files['model.py'],
          train: result.data.files['train.py'],
          dataset: result.data.files['dataset.py'],
          config: result.data.files['config.py'],
          zip: result.data.zip || '',
          filename: result.data.filename || 'export.zip'
        })
        setIsExportOpen(true)
        toast.success(`${result.data.framework.toUpperCase()} code generated successfully!`)
      } else {
        toast.error('Code generation failed', {
          description: result.error || 'Unknown error occurred'
        })
      }
    } catch (error) {
      toast.dismiss()
      toast.error('Code generation failed', {
        description: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }

  const handleExportJSON = () => {
    if (nodes.length === 0) {
      toast.error('Cannot export: No blocks in architecture')
      return
    }

    try {
      const exportData = exportToJSON(nodes, edges, currentProject)
      downloadJSON(exportData)
      toast.success('JSON exported successfully!')
    } catch (error) {
      toast.error('JSON export failed', {
        description: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }

  const handleImportJSON = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    try {
      const jsonData = await readJSONFile(file)

      // Check if we're in an existing project or on the home page
      if (projectId && currentProject) {
        // CASE 1: Importing into an existing project
        // Only import nodes/edges, preserve project metadata
        // Pass existing nodes to handle ID conflicts
        const { nodes: importedNodes, edges: importedEdges } = importFromJSON(
          jsonData,
          nodes,
          edges
        )

        // Merge with existing nodes and edges
        const mergedNodes = [...nodes, ...importedNodes]
        const mergedEdges = [...edges, ...importedEdges]

        // Update the canvas
        setNodes(mergedNodes)
        setEdges(mergedEdges)

        // Save to backend
        await projectApi.saveArchitecture(projectId, mergedNodes, mergedEdges)

        toast.success('Architecture imported into current project!', {
          description: `Added ${importedNodes.length} blocks to "${currentProject.name}"`
        })
      } else {
        // CASE 2: No active project - create a new one from import
        const { nodes: importedNodes, edges: importedEdges, project } = importFromJSON(jsonData)

        if (project.name && project.description !== undefined) {
          const backendProject = await projectApi.createProject({
            name: project.name,
            description: project.description,
            framework: project.framework || 'pytorch'
          })

          // Save the architecture
          await projectApi.saveArchitecture(backendProject.id, importedNodes, importedEdges)

          // Navigate to new project
          navigate(`/project/${backendProject.id}`)

          toast.success('Project created from import!', {
            description: `Created project "${project.name}" with ${importedNodes.length} blocks`
          })

          // Reload projects list
          loadProjectsList()
        }
      }

      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    } catch (error) {
      toast.error('Import failed', {
        description: error instanceof Error ? error.message : 'Invalid JSON file'
      })

      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const triggerFileInput = () => {
    fileInputRef.current?.click()
  }

  const handleValidate = async () => {
    if (nodes.length === 0) {
      toast.error('Cannot validate: No architecture to validate')
      return
    }

    try {
      toast.loading('Validating architecture...')

      const result = await api.validateModel({
        nodes: nodes.map(node => ({
          id: node.id,
          type: node.data.blockType,
          data: node.data,
          position: node.position
        })),
        edges: edges.map(edge => ({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          sourceHandle: edge.sourceHandle || '',
          targetHandle: edge.targetHandle || ''
        }))
      })

      toast.dismiss()

      if (result.success && result.data) {
        if (result.data.isValid) {
          toast.success('Architecture is valid!', {
            description: result.data.warnings && result.data.warnings.length > 0
              ? `${result.data.warnings.length} warning(s) found`
              : 'No issues detected'
          })

          // Show warnings if any
          if (result.data.warnings && result.data.warnings.length > 0) {
            result.data.warnings.forEach((warning: any, index: number) => {
              setTimeout(() => {
                toast.warning(warning.message || `Warning ${index + 1}`, {
                  description: warning.suggestion || warning.nodeId ? `Node: ${warning.nodeId}` : undefined
                })
              }, index * 100)
            })
          }
        } else {
          toast.error('Architecture validation failed', {
            description: result.data.errors && result.data.errors.length > 0
              ? `${result.data.errors.length} error(s) found`
              : 'Invalid architecture'
          })

          // Show errors
          if (result.data.errors && result.data.errors.length > 0) {
            result.data.errors.forEach((error: any, index: number) => {
              setTimeout(() => {
                toast.error(error.message || `Error ${index + 1}`, {
                  description: error.suggestion || error.nodeId ? `Node: ${error.nodeId}` : undefined
                })
              }, index * 100)
            })
          }
        }
      } else {
        toast.error('Validation request failed', {
          description: result.error || 'Could not connect to validation service'
        })
      }
    } catch (error) {
      toast.dismiss()
      toast.error('Validation error', {
        description: error instanceof Error ? error.message : 'Unknown error occurred'
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
    toast.success(`${filename} downloaded!`)
  }

  const handleOpenProjectManagement = (project: projectApi.ProjectResponse, e: React.MouseEvent) => {
    e.stopPropagation()
    setManagingProject(project)
    setEditProjectName(project.name)
    setEditProjectDesc(project.description)
    setIsManageProjectOpen(true)
  }

  const handleUpdateProject = async () => {
    if (!managingProject) return

    if (!editProjectName.trim()) {
      toast.error('Please enter a project name')
      return
    }

    try {
      await projectApi.updateProject(managingProject.id, {
        name: editProjectName,
        description: editProjectDesc
      })

      toast.success('Project updated!')
      setIsManageProjectOpen(false)
      loadProjectsList()

      // If we're updating the current project, reload it
      if (currentProject?.id === managingProject.id) {
        navigate(`/project/${managingProject.id}`)
      }
    } catch (error) {
      toast.error('Failed to update project', {
        description: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }

  const handleDeleteProject = async () => {
    if (!managingProject) return

    try {
      await projectApi.deleteProject(managingProject.id)

      toast.success('Project deleted!')
      setIsDeleteConfirmOpen(false)
      setIsManageProjectOpen(false)
      loadProjectsList()

      // If we deleted the current project, clear the store and go to blank canvas
      if (currentProject?.id === managingProject.id) {
        const { reset } = useModelBuilderStore.getState()
        reset()
        navigate('/project')
      }
    } catch (error) {
      toast.error('Failed to delete project', {
        description: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }

  return (
    <div className="flex flex-col">
      {/* Demo Mode Banner */}
      <div className="bg-amber-500/10 border-b border-amber-500/20 px-4 py-1.5 flex items-center justify-center gap-2">
        <Warning size={14} className="text-amber-600" />
        <span className="text-xs text-amber-700 dark:text-amber-400">
          <strong>Demo Mode:</strong> Designs are saved locally in your browser. Provide your own Gemini API key to use AI features.
        </span>
      </div>

      <header className="h-16 border-b border-border bg-card px-6 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <img src="/logo_navbar.png" alt="VisionForge Logo" className="h-10 w-auto" />
            <h1 className="text-xl font-semibold">VisionForge</h1>
          </div>

        {/* Project Dropdown */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" className="h-8 gap-2">
              <GitBranch size={16} />
              <span className="font-medium">
                {currentProject ? currentProject.name : 'No Project'}
              </span>
              {currentProject && (
                <span className="text-xs text-muted-foreground">
                  ({currentProject.framework})
                </span>
              )}
              <CaretDown size={14} className="text-muted-foreground" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="w-xs">
            <DropdownMenuLabel className="text-xs text-muted-foreground font-normal">
              Switch project or create new
            </DropdownMenuLabel>
            <DropdownMenuSeparator />

            {/* New Project Option */}
            <DropdownMenuItem
              className="gap-2 cursor-pointer"
              onSelect={() => setIsNewProjectOpen(true)}
            >
              <Plus size={16} className="text-primary" />
              <div className="flex-1">
                <div className="font-medium">Create New Project</div>
                <div className="text-xs text-muted-foreground">
                  Start building a new architecture
                </div>
              </div>
            </DropdownMenuItem>

            {/* Saved Projects List */}
            {projects && projects.length > 0 && (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuLabel className="text-xs text-muted-foreground font-normal">
                  Recent Projects
                </DropdownMenuLabel>
                <ScrollArea className="max-h-[300px]">
                  {projects.slice(0, 10).map((project) => (
                    <DropdownMenuItem
                      key={project.id}
                      className="cursor-pointer flex-col items-start gap-1 py-2"
                      onSelect={() => handleLoadProject(project)}
                    >
                      <div className="flex items-center gap-2 w-full">
                        <GitBranch size={14} className="text-muted-foreground shrink-0" />
                        <span className="font-medium flex-1 truncate">{project.name}</span>
                        <div className="flex items-center gap-1 shrink-0">
                          {currentProject?.id === project.id && (
                            <CheckCircle size={14} weight="fill" className="text-primary" />
                          )}
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 hover:bg-accent"
                            onClick={(e) => handleOpenProjectManagement(project, e)}
                          >
                            <GearSix size={14} />
                          </Button>
                        </div>
                      </div>
                      <div className="text-xs text-muted-foreground pl-5">
                        {project.description || 'No description'} • {project.framework}
                      </div>
                      <div className="text-xs text-muted-foreground pl-5">
                        Updated {new Date(project.updated_at).toLocaleDateString()}
                      </div>
                    </DropdownMenuItem>
                  ))}
                </ScrollArea>
              </>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <div className="flex items-center gap-2">
        <ThemeToggle />
        
        {/* Hidden file input for JSON import */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          onChange={handleImportJSON}
          className="hidden"
        />

        {/* New Project Dialog */}
        <Dialog open={isNewProjectOpen} onOpenChange={setIsNewProjectOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create New Project</DialogTitle>
              <DialogDescription>
                Start building a new neural network architecture
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 pt-4">
              <div>
                <Label htmlFor="project-name">Project Name *</Label>
                <Input
                  id="project-name"
                  placeholder="My Model"
                  value={newProjectName}
                  onChange={(e) => setNewProjectName(e.target.value)}
                />
              </div>
              <div>
                <Label htmlFor="project-desc">Description</Label>
                <Textarea
                  id="project-desc"
                  placeholder="Describe your model architecture..."
                  value={newProjectDesc}
                  onChange={(e) => setNewProjectDesc(e.target.value)}
                />
              </div>
              <div>
                <Label htmlFor="framework">Framework</Label>
                <Select value={newProjectFramework} onValueChange={(v) => setNewProjectFramework(v as any)}>
                  <SelectTrigger id="framework">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="pytorch">PyTorch</SelectItem>
                    <SelectItem value="tensorflow">TensorFlow</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Button onClick={handleCreateProject} className="w-full">
                Create Project
              </Button>
            </div>
          </DialogContent>
        </Dialog>

        {/* Save As Dialog */}
        <Dialog open={isSaveAsDialogOpen} onOpenChange={setIsSaveAsDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Save Project</DialogTitle>
              <DialogDescription>
                Name your project to save your architecture
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 pt-4">
              <div>
                <Label htmlFor="save-as-name">Project Name *</Label>
                <Input
                  id="save-as-name"
                  placeholder="My Model"
                  value={saveAsProjectName}
                  onChange={(e) => setSaveAsProjectName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      handleSaveAsProject()
                    }
                  }}
                  autoFocus
                />
              </div>
              <div>
                <Label htmlFor="save-as-desc">Description (Optional)</Label>
                <Textarea
                  id="save-as-desc"
                  placeholder="Describe your model architecture..."
                  value={saveAsProjectDesc}
                  onChange={(e) => setSaveAsProjectDesc(e.target.value)}
                  rows={3}
                />
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  onClick={() => setIsSaveAsDialogOpen(false)}
                  className="flex-1"
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleSaveAsProject}
                  className="flex-1"
                >
                  <FloppyDisk size={16} className="mr-2" />
                  Save Project
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>

        {/* Import Button */}
        <Button
          variant="outline"
          size="sm"
          onClick={triggerFileInput}
        >
          <Upload size={16} className="mr-2" />
          Import
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={handleSaveProject}
          disabled={nodes.length === 0}
        >
          <FloppyDisk size={16} className="mr-2" />
          Save
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={handleValidate}
          disabled={nodes.length === 0}
        >
          <CheckCircle size={16} className="mr-2" />
          Validate
        </Button>

        {/* Export Dropdown */}
        <Dialog open={isExportOpen} onOpenChange={setIsExportOpen}>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="default"
                size="sm"
                disabled={nodes.length === 0}
              >
                <Download size={16} className="mr-2" />
                Export
                <CaretDown size={14} className="ml-1" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>Export Options</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={handleExportCode} className="gap-2 cursor-pointer">
                <FilePy size={16} />
                <div>
                  <div className="font-medium">
                    {currentProject ? `${currentProject.framework.toUpperCase()} Code` : 'Model Code'}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Generate model.py, train.py, dataset.py, config.py
                  </div>
                </div>
              </DropdownMenuItem>
              <DropdownMenuItem onClick={handleExportJSON} className="gap-2 cursor-pointer">
                <FileCode size={16} />
                <div>
                  <div className="font-medium">JSON Architecture</div>
                  <div className="text-xs text-muted-foreground">
                    Export as importable JSON file
                  </div>
                </div>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <DialogContent className="max-w-4xl max-h-[90vh] w-full overflow-hidden flex flex-col">
            <DialogHeader>
              <DialogTitle>
                Export {currentProject ? currentProject.framework.toUpperCase() : 'Model'} Code
              </DialogTitle>
              <DialogDescription>
                Copy individual files or download all as ZIP
              </DialogDescription>
            </DialogHeader>
            {exportCode && (
              <div className="flex-1 flex flex-col min-h-0">
                {/* Download All Button */}
                <div className="mb-3">
                  <Button
                    className="w-full"
                    onClick={() => {
                      // Decode base64 zip and download
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
                <TabsContent value="model" className="mt-4 flex-1 flex flex-col min-h-0">
                  <div className="flex-1 w-full border rounded-md overflow-auto bg-muted">
                    <div className="min-w-max">
                      <pre className="text-xs font-mono p-4 whitespace-pre">
                        <code>{exportCode.model}</code>
                      </pre>
                    </div>
                  </div>
                  <div className="flex gap-2 mt-3 shrink-0">
                    <Button
                      className="flex-1"
                      variant="outline"
                      onClick={() => copyToClipboard(exportCode.model, 'model.py')}
                    >
                      <Code size={16} className="mr-2" />
                      Copy
                    </Button>
                    <Button
                      className="flex-1"
                      variant="outline"
                      onClick={() => downloadFile(exportCode.model, 'model.py')}
                    >
                      <Download size={16} className="mr-2" />
                      Download
                    </Button>
                  </div>
                </TabsContent>
                <TabsContent value="train" className="mt-4 flex-1 flex flex-col min-h-0">
                  <div className="flex-1 w-full border rounded-md overflow-auto bg-muted">
                    <div className="min-w-max">
                      <pre className="text-xs font-mono p-4 whitespace-pre">
                        <code>{exportCode.train}</code>
                      </pre>
                    </div>
                  </div>
                  <div className="flex gap-2 mt-3 shrink-0">
                    <Button
                      className="flex-1"
                      variant="outline"
                      onClick={() => copyToClipboard(exportCode.train, 'train.py')}
                    >
                      <Code size={16} className="mr-2" />
                      Copy
                    </Button>
                    <Button
                      className="flex-1"
                      variant="outline"
                      onClick={() => downloadFile(exportCode.train, 'train.py')}
                    >
                      <Download size={16} className="mr-2" />
                      Download
                    </Button>
                  </div>
                </TabsContent>
                <TabsContent value="dataset" className="mt-4 flex-1 flex flex-col min-h-0">
                  <div className="flex-1 w-full border rounded-md overflow-auto bg-muted">
                    <div className="min-w-max">
                      <pre className="text-xs font-mono p-4 whitespace-pre">
                        <code>{exportCode.dataset}</code>
                      </pre>
                    </div>
                  </div>
                  <div className="flex gap-2 mt-3 shrink-0">
                    <Button
                      className="flex-1"
                      variant="outline"
                      onClick={() => copyToClipboard(exportCode.dataset, 'dataset.py')}
                    >
                      <Code size={16} className="mr-2" />
                      Copy
                    </Button>
                    <Button
                      className="flex-1"
                      variant="outline"
                      onClick={() => downloadFile(exportCode.dataset, 'dataset.py')}
                    >
                      <Download size={16} className="mr-2" />
                      Download
                    </Button>
                  </div>
                </TabsContent>
                <TabsContent value="config" className="mt-4 flex-1 flex flex-col min-h-0">
                  <div className="flex-1 w-full border rounded-md overflow-auto bg-muted">
                    <div className="min-w-max">
                      <pre className="text-xs font-mono p-4 whitespace-pre">
                        <code>{exportCode.config}</code>
                      </pre>
                    </div>
                  </div>
                  <div className="flex gap-2 mt-3 shrink-0">
                    <Button
                      className="flex-1"
                      variant="outline"
                      onClick={() => copyToClipboard(exportCode.config, 'config.py')}
                    >
                      <Code size={16} className="mr-2" />
                      Copy
                    </Button>
                    <Button
                      className="flex-1"
                      variant="outline"
                      onClick={() => downloadFile(exportCode.config, 'config.py')}
                    >
                      <Download size={16} className="mr-2" />
                      Download
                    </Button>
                  </div>
                </TabsContent>
                </Tabs>
              </div>
            )}
          </DialogContent>
        </Dialog>

        {/* Project Management Dialog */}
        <Dialog open={isManageProjectOpen} onOpenChange={setIsManageProjectOpen}>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>Manage Project</DialogTitle>
              <DialogDescription>
                {managingProject?.name}
              </DialogDescription>
            </DialogHeader>
            {managingProject && (
              <Tabs defaultValue="edit" className="w-full">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="edit">
                    <PencilSimple size={16} className="mr-2" />
                    Edit
                  </TabsTrigger>
                  <TabsTrigger value="info">
                    <Info size={16} className="mr-2" />
                    Info
                  </TabsTrigger>
                  <TabsTrigger value="delete">
                    <Trash size={16} className="mr-2" />
                    Delete
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="edit" className="space-y-4 pt-4">
                  <div>
                    <Label htmlFor="edit-project-name">Project Name *</Label>
                    <Input
                      id="edit-project-name"
                      placeholder="My Model"
                      value={editProjectName}
                      onChange={(e) => setEditProjectName(e.target.value)}
                    />
                  </div>
                  <div>
                    <Label htmlFor="edit-project-desc">Description</Label>
                    <Textarea
                      id="edit-project-desc"
                      placeholder="Describe your model architecture..."
                      value={editProjectDesc}
                      onChange={(e) => setEditProjectDesc(e.target.value)}
                      rows={4}
                    />
                  </div>
                  <div className="flex items-center justify-between p-3 bg-muted rounded-md">
                    <div>
                      <Label className="text-sm font-medium">Framework</Label>
                      <p className="text-sm text-muted-foreground">
                        {managingProject.framework.toUpperCase()}
                      </p>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Framework cannot be changed after creation
                    </p>
                  </div>
                  <Button onClick={handleUpdateProject} className="w-full">
                    <PencilSimple size={16} className="mr-2" />
                    Save Changes
                  </Button>
                </TabsContent>

                <TabsContent value="info" className="space-y-4 pt-4">
                  <div className="space-y-3">
                    <div className="p-3 bg-muted rounded-md">
                      <Label className="text-sm font-medium">Project ID</Label>
                      <p className="text-sm text-muted-foreground font-mono">{managingProject.id}</p>
                    </div>
                    <div className="p-3 bg-muted rounded-md">
                      <Label className="text-sm font-medium">Framework</Label>
                      <p className="text-sm text-muted-foreground">{managingProject.framework.toUpperCase()}</p>
                    </div>
                    <div className="p-3 bg-muted rounded-md">
                      <Label className="text-sm font-medium">Created</Label>
                      <p className="text-sm text-muted-foreground">
                        {new Date(managingProject.created_at).toLocaleString()}
                      </p>
                    </div>
                    <div className="p-3 bg-muted rounded-md">
                      <Label className="text-sm font-medium">Last Updated</Label>
                      <p className="text-sm text-muted-foreground">
                        {new Date(managingProject.updated_at).toLocaleString()}
                      </p>
                    </div>
                    <div className="p-3 bg-muted rounded-md">
                      <Label className="text-sm font-medium">Description</Label>
                      <p className="text-sm text-muted-foreground">
                        {managingProject.description || 'No description provided'}
                      </p>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="delete" className="space-y-4 pt-4">
                  <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-md">
                    <div className="flex items-start gap-3">
                      <Trash size={20} className="text-destructive shrink-0 mt-0.5" />
                      <div className="space-y-2">
                        <h4 className="font-semibold text-destructive">Delete Project</h4>
                        <p className="text-sm text-muted-foreground">
                          This will permanently delete the project "{managingProject.name}" and all its architecture data.
                          This action cannot be undone.
                        </p>
                      </div>
                    </div>
                  </div>

                  {!isDeleteConfirmOpen ? (
                    <Button
                      variant="destructive"
                      className="w-full"
                      onClick={() => setIsDeleteConfirmOpen(true)}
                    >
                      <Trash size={16} className="mr-2" />
                      Delete Project
                    </Button>
                  ) : (
                    <div className="space-y-3">
                      <p className="text-sm font-medium text-center">
                        Are you sure you want to delete this project?
                      </p>
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          className="flex-1"
                          onClick={() => setIsDeleteConfirmOpen(false)}
                        >
                          Cancel
                        </Button>
                        <Button
                          variant="destructive"
                          className="flex-1"
                          onClick={handleDeleteProject}
                        >
                          <Trash size={16} className="mr-2" />
                          Confirm Delete
                        </Button>
                      </div>
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            )}
          </DialogContent>
        </Dialog>
      </div>
      </header>
    </div>
  )
}
