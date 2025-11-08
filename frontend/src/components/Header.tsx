import { useState } from 'react'
import { useModelBuilderStore } from '@/lib/store'
import { useKV } from '@github/spark/hooks'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { Plus, Download, FloppyDisk, FolderOpen, Code, Flask } from '@phosphor-icons/react'
import { toast } from 'sonner'
import { generatePyTorchCode } from '@/lib/codeGenerator'
import { Project } from '@/lib/types'

export default function Header() {
  const { currentProject, nodes, edges, createProject, saveProject, loadProject, updateProjectInfo, validateArchitecture } = useModelBuilderStore()
  const [projects, setProjects] = useKV<Project[]>('model-builder-projects', [])

  const [isNewProjectOpen, setIsNewProjectOpen] = useState(false)
  const [isLoadProjectOpen, setIsLoadProjectOpen] = useState(false)
  const [isExportOpen, setIsExportOpen] = useState(false)

  const [newProjectName, setNewProjectName] = useState('')
  const [newProjectDesc, setNewProjectDesc] = useState('')
  const [newProjectFramework, setNewProjectFramework] = useState<'pytorch' | 'tensorflow'>('pytorch')

  const [exportCode, setExportCode] = useState<{model: string, train: string, config: string} | null>(null)

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
    if (!currentProject) {
      toast.error('No active project to save')
      return
    }

    saveProject()

    setProjects((prevProjects) => {
      const projectList = prevProjects || []
      const existingIndex = projectList.findIndex((p) => p.id === currentProject.id)
      const updatedProject = { ...currentProject, nodes, edges, updatedAt: Date.now() }

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
    setIsLoadProjectOpen(false)
    toast.success(`Loaded "${project.name}"`)
  }

  const handleExport = () => {
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

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text)
    toast.success(`${label} copied to clipboard!`)
  }

  return (
    <header className="h-16 border-b border-border bg-card px-6 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <Flask size={28} weight="fill" className="text-primary" />
          <h1 className="text-xl font-semibold">Visual AI Model Builder</h1>
        </div>

        {currentProject && (
          <div className="ml-4 px-3 py-1 bg-muted rounded text-sm">
            <span className="font-medium">{currentProject.name}</span>
            <span className="text-muted-foreground ml-2">({currentProject.framework})</span>
          </div>
        )}
      </div>

      <div className="flex items-center gap-2">
        <Dialog open={isNewProjectOpen} onOpenChange={setIsNewProjectOpen}>
          <DialogTrigger asChild>
            <Button variant="outline" size="sm">
              <Plus size={16} className="mr-2" />
              New Project
            </Button>
          </DialogTrigger>
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

        <Dialog open={isLoadProjectOpen} onOpenChange={setIsLoadProjectOpen}>
          <DialogTrigger asChild>
            <Button variant="outline" size="sm">
              <FolderOpen size={16} className="mr-2" />
              Load
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Load Project</DialogTitle>
              <DialogDescription>
                Select a previously saved project
              </DialogDescription>
            </DialogHeader>
            <ScrollArea className="max-h-[400px] pr-4">
              {!projects || projects.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  No saved projects yet
                </div>
              ) : (
                <div className="space-y-2 pt-4">
                  {projects.map((project) => (
                    <div
                      key={project.id}
                      className="p-4 border rounded-lg hover:bg-muted/50 cursor-pointer transition-colors"
                      onClick={() => handleLoadProject(project)}
                    >
                      <div className="font-medium">{project.name}</div>
                      <div className="text-sm text-muted-foreground mt-1">
                        {project.description || 'No description'}
                      </div>
                      <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
                        <span className="px-2 py-0.5 bg-primary/10 text-primary rounded">
                          {project.framework}
                        </span>
                        <span>
                          {project.nodes.length} blocks
                        </span>
                        <span>•</span>
                        <span>
                          {new Date(project.updatedAt).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>
          </DialogContent>
        </Dialog>

        <Button
          variant="outline"
          size="sm"
          onClick={handleSaveProject}
          disabled={!currentProject}
        >
          <FloppyDisk size={16} className="mr-2" />
          Save
        </Button>

        <Dialog open={isExportOpen} onOpenChange={setIsExportOpen}>
          <Button
            variant="default"
            size="sm"
            onClick={handleExport}
            disabled={!currentProject || nodes.length === 0}
          >
            <Download size={16} className="mr-2" />
            Export Code
          </Button>
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
