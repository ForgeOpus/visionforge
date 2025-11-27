import { Node, Edge } from '@xyflow/react'
import { BlockData, Project } from './types'
import {
  getLocalDesigns,
  saveDesignToLocal,
  loadDesignFromLocal,
  deleteLocalDesign,
  generateDesignId,
  LocalDesign,
} from './localStorageService'

export interface ProjectResponse {
  id: string
  name: string
  description: string
  framework: 'pytorch' | 'tensorflow'
  created_at: string
  updated_at: string
}

export interface ProjectDetailResponse extends ProjectResponse {
  architecture?: {
    id: string
    canvas_state: {
      nodes: any[]
      edges: any[]
    }
    is_valid: boolean
    validation_errors: any[]
    created_at: string
    updated_at: string
  }
}

export interface ProjectListResponse {
  projects: ProjectResponse[]
}

/**
 * Helper to convert LocalDesign to ProjectResponse format
 */
function localDesignToProjectResponse(design: LocalDesign): ProjectResponse {
  return {
    id: design.id,
    name: design.name,
    description: design.description,
    framework: design.framework,
    created_at: new Date(design.createdAt).toISOString(),
    updated_at: new Date(design.updatedAt).toISOString(),
  }
}

/**
 * Fetch all projects from local storage
 */
export async function fetchProjects(): Promise<ProjectResponse[]> {
  const designs = getLocalDesigns()
  return designs.map(localDesignToProjectResponse)
}

/**
 * Fetch a single project with full details from local storage
 */
export async function fetchProject(projectId: string): Promise<ProjectDetailResponse> {
  const design = loadDesignFromLocal(projectId)

  if (!design) {
    throw new Error(`Project not found: ${projectId}`)
  }

  return {
    ...localDesignToProjectResponse(design),
    architecture: {
      id: `arch-${design.id}`,
      canvas_state: {
        nodes: design.nodes,
        edges: design.edges,
      },
      is_valid: true,
      validation_errors: [],
      created_at: new Date(design.createdAt).toISOString(),
      updated_at: new Date(design.updatedAt).toISOString(),
    },
  }
}

/**
 * Create a new project in local storage
 */
export async function createProject(data: {
  name: string
  description: string
  framework: 'pytorch' | 'tensorflow'
}): Promise<ProjectResponse> {
  const newDesign: LocalDesign = {
    id: generateDesignId(),
    name: data.name,
    description: data.description,
    framework: data.framework,
    nodes: [],
    edges: [],
    createdAt: Date.now(),
    updatedAt: Date.now(),
  }

  saveDesignToLocal(newDesign)
  return localDesignToProjectResponse(newDesign)
}

/**
 * Update project metadata in local storage
 */
export async function updateProject(
  projectId: string,
  data: Partial<{
    name: string
    description: string
    framework: 'pytorch' | 'tensorflow'
  }>
): Promise<ProjectResponse> {
  const design = loadDesignFromLocal(projectId)

  if (!design) {
    throw new Error(`Project not found: ${projectId}`)
  }

  const updatedDesign: LocalDesign = {
    ...design,
    ...(data.name && { name: data.name }),
    ...(data.description !== undefined && { description: data.description }),
    ...(data.framework && { framework: data.framework }),
    updatedAt: Date.now(),
  }

  saveDesignToLocal(updatedDesign)
  return localDesignToProjectResponse(updatedDesign)
}

/**
 * Delete a project from local storage
 */
export async function deleteProject(projectId: string): Promise<void> {
  deleteLocalDesign(projectId)
}

/**
 * Save architecture (nodes and edges) to local storage
 */
export async function saveArchitecture(
  projectId: string,
  nodes: Node<BlockData>[],
  edges: Edge[]
): Promise<{ success: boolean; architecture_id: string }> {
  const design = loadDesignFromLocal(projectId)

  if (!design) {
    throw new Error(`Project not found: ${projectId}`)
  }

  const updatedDesign: LocalDesign = {
    ...design,
    nodes,
    edges,
    updatedAt: Date.now(),
  }

  saveDesignToLocal(updatedDesign)
  return { success: true, architecture_id: `arch-${projectId}` }
}

/**
 * Load architecture (nodes and edges) from local storage
 */
export async function loadArchitecture(projectId: string): Promise<{
  nodes: Node<BlockData>[]
  edges: Edge[]
}> {
  const design = loadDesignFromLocal(projectId)

  if (!design) {
    throw new Error(`Project not found: ${projectId}`)
  }

  return {
    nodes: design.nodes,
    edges: design.edges,
  }
}

/**
 * Convert backend project to frontend Project type
 */
export function convertToFrontendProject(
  backendProject: ProjectResponse | ProjectDetailResponse,
  nodes: Node<BlockData>[] = [],
  edges: Edge[] = []
): Project {
  return {
    id: backendProject.id,
    name: backendProject.name,
    description: backendProject.description,
    framework: backendProject.framework,
    nodes,
    edges,
    createdAt: new Date(backendProject.created_at).getTime(),
    updatedAt: new Date(backendProject.updated_at).getTime(),
  }
}
