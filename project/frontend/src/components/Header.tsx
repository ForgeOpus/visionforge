import { useState, useRef } from 'react'
import { useModelBuilderStore } from '@/lib/store'
import { useKV } from '@github/spark/hooks'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { Plus, Download, FloppyDisk, CaretDown, Code, Flask, CheckCircle, GitBranch, Upload, FileCode, FilePy } from '@phosphor-icons/react'
import { toast } from 'sonner'
import { generatePyTorchCode } from '@/lib/codeGenerator'
import { validateModel } from '@/lib/api'
import { Project } from '@/lib/types'
import { exportToJSON, importFromJSON, downloadJSON, readJSONFile } from '@/lib/exportImport'

export default function Header() {
  const { currentProject, nodes, edges, createProject, saveProject, loadProject, updateProjectInfo, validateArchitecture, setNodes, setEdges } = useModelBuilderStore()
  const [projects, setProjects] = useKV<Project[]>('model-builder-projects', [])

  const [isNewProjectOpen, setIsNewProjectOpen] = useState(false)
  const [isExportOpen, setIsExportOpen] = useState(false)

  const [newProjectName, setNewProjectName] = useState('')
  const [newProjectDesc, setNewProjectDesc] = useState('')
  const [newProjectFramework, setNewProjectFramework] = useState<'pytorch' | 'tensorflow'>('pytorch')

  const [exportCode, setExportCode] = useState<{model: string, train: string, config: string} | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleCreateProject = () => {
    if (!newProjectName.trim()) {
      toast.error('Please enter a project name')
      return
    }

    createProject(newProjectName, newProjectDesc, newProjectFramework)
    setIsNewProjectOpen(false)
    setNewProjectName('')
    setNewProjectDesc('')
    toast.success('Project created!')
  }

  const handleSaveProject = () => {
    if (nodes.length === 0) {
      toast.error('No architecture to save')
      return
    }

    // Ensure we have a project (auto-created when first node added)
    const project = currentProject
    if (!project) {
      toast.error('No active project')
      return
    }

    saveProject()

    setProjects((prevProjects) => {
      const projectList = prevProjects || []
      const existingIndex = projectList.findIndex((p) => p.id === project.id)
      const updatedProject = { ...project, nodes, edges, updatedAt: Date.now() }

      if (existingIndex >= 0) {
        const updated = [...projectList]
        updated[existingIndex] = updatedProject
        return updated
      } else {
        return [...projectList, updatedProject]
      }
    })

    toast.success('Project saved!')
  }

  const handleLoadProject = (project: Project) => {
    loadProject(project)
    toast.success(`Loaded "${project.name}"`)
  }

  const handleExportPyTorch = () => {
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

    try {
      const code = generatePyTorchCode(nodes, edges, currentProject?.name || 'CustomModel')
      setExportCode(code)
      setIsExportOpen(true)
      toast.success('Code generated successfully!')
    } catch (error) {
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
      const { nodes: importedNodes, edges: importedEdges, project } = importFromJSON(jsonData)

      // Create a new project from imported data or update current
      if (project.name && project.description !== undefined) {
        createProject(
          project.name,
          project.description,
          project.framework || 'pytorch'
        )
      }

      // Set the imported nodes and edges
      setNodes(importedNodes)
      setEdges(importedEdges)

      // Trigger validation to update error badges
      setTimeout(() => {
        validateArchitecture()
      }, 100)

      toast.success('Architecture imported successfully!', {
        description: `Loaded ${importedNodes.length} blocks and ${importedEdges.length} connections`
      })

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
      
      const result = await validateModel({
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

  return (
    <header className="h-16 border-b border-border bg-card px-6 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <Flask size={28} weight="fill" className="text-primary" />
          <h1 className="text-xl font-semibold">VisionForge</h1>
        </div>

        {/* Project Dropdown - GitHub Branch Style */}
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
                        <GitBranch size={14} className="text-muted-foreground" />
                        <span className="font-medium flex-1">{project.name}</span>
                        {currentProject?.id === project.id && (
                          <CheckCircle size={14} weight="fill" className="text-primary" />
                        )}
                      </div>
                      <div className="text-xs text-muted-foreground pl-5">
                        {project.description || 'No description'} • {project.framework} • {project.nodes.length} blocks
                      </div>
                      <div className="text-xs text-muted-foreground pl-5">
                        Updated {new Date(project.updatedAt).toLocaleDateString()}
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
              <DropdownMenuItem onClick={handleExportPyTorch} className="gap-2 cursor-pointer">
                <FilePy size={16} />
                <div>
                  <div className="font-medium">PyTorch Code</div>
                  <div className="text-xs text-muted-foreground">
                    Generate model.py, train.py, config.py
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
          <DialogContent className="max-w-4xl max-h-[80vh]">
            <DialogHeader>
              <DialogTitle>Export PyTorch Code</DialogTitle>
              <DialogDescription>
                Copy the generated code files or download them
              </DialogDescription>
            </DialogHeader>
            {exportCode && (
              <Tabs defaultValue="model" className="w-full">
                <TabsList>
                  <TabsTrigger value="model">model.py</TabsTrigger>
                  <TabsTrigger value="train">train.py</TabsTrigger>
                  <TabsTrigger value="config">config.py</TabsTrigger>
                </TabsList>
                <TabsContent value="model">
                  <ScrollArea className="h-[400px] w-full">
                    <pre className="text-xs font-mono bg-muted p-4 rounded">
                      <code>{exportCode.model}</code>
                    </pre>
                  </ScrollArea>
                  <Button
                    className="w-full mt-2"
                    variant="outline"
                    onClick={() => copyToClipboard(exportCode.model, 'model.py')}
                  >
                    <Code size={16} className="mr-2" />
                    Copy model.py
                  </Button>
                </TabsContent>
                <TabsContent value="train">
                  <ScrollArea className="h-[400px] w-full">
                    <pre className="text-xs font-mono bg-muted p-4 rounded">
                      <code>{exportCode.train}</code>
                    </pre>
                  </ScrollArea>
                  <Button
                    className="w-full mt-2"
                    variant="outline"
                    onClick={() => copyToClipboard(exportCode.train, 'train.py')}
                  >
                    <Code size={16} className="mr-2" />
                    Copy train.py
                  </Button>
                </TabsContent>
                <TabsContent value="config">
                  <ScrollArea className="h-[400px] w-full">
                    <pre className="text-xs font-mono bg-muted p-4 rounded">
                      <code>{exportCode.config}</code>
                    </pre>
                  </ScrollArea>
                  <Button
                    className="w-full mt-2"
                    variant="outline"
                    onClick={() => copyToClipboard(exportCode.config, 'config.py')}
                  >
                    <Code size={16} className="mr-2" />
                    Copy config.py
                  </Button>
                </TabsContent>
              </Tabs>
            )}
          </DialogContent>
        </Dialog>
      </div>
    </header>
  )
}
