import { Node, Edge } from '@xyflow/react'
import { BlockData, Project } from './types'
import { API_BASE_URL, createFetchOptions } from './apiUtils'

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
 * Fetch all projects
 */
export async function fetchProjects(): Promise<ProjectResponse[]> {
  const response = await fetch(`${API_BASE_URL}/projects/`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
  })

  if (!response.ok) {
    throw new Error(`Failed to fetch projects: ${response.statusText}`)
  }

  const data: ProjectListResponse = await response.json()
  return data.projects
}

/**
 * Fetch a single project with full details
 */
export async function fetchProject(projectId: string): Promise<ProjectDetailResponse> {
  const response = await fetch(`${API_BASE_URL}/projects/${projectId}/`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
  })

  if (!response.ok) {
    throw new Error(`Failed to fetch project: ${response.statusText}`)
  }

  return await response.json()
}

/**
 * Create a new project
 */
export async function createProject(data: {
  name: string
  description: string
  framework: 'pytorch' | 'tensorflow'
}): Promise<ProjectResponse> {
  const response = await fetch(
    `${API_BASE_URL}/projects/`,
    createFetchOptions('POST', data)
  )

  if (!response.ok) {
    throw new Error(`Failed to create project: ${response.statusText}`)
  }

  return await response.json()
}

/**
 * Update project metadata
 */
export async function updateProject(
  projectId: string,
  data: Partial<{
    name: string
    description: string
    framework: 'pytorch' | 'tensorflow'
  }>
): Promise<ProjectResponse> {
  const response = await fetch(
    `${API_BASE_URL}/projects/${projectId}/`,
    createFetchOptions('PATCH', data)
  )

  if (!response.ok) {
    throw new Error(`Failed to update project: ${response.statusText}`)
  }

  return await response.json()
}

/**
 * Delete a project
 */
export async function deleteProject(projectId: string): Promise<void> {
  const response = await fetch(
    `${API_BASE_URL}/projects/${projectId}/`,
    createFetchOptions('DELETE')
  )

  if (!response.ok) {
    throw new Error(`Failed to delete project: ${response.statusText}`)
  }
}

/**
 * Save architecture (nodes and edges) for a project
 */
export async function saveArchitecture(
  projectId: string,
  nodes: Node<BlockData>[],
  edges: Edge[]
): Promise<{ success: boolean; architecture_id: string }> {
  const response = await fetch(
    `${API_BASE_URL}/projects/${projectId}/save-architecture`,
    createFetchOptions('POST', { nodes, edges })
  )

  if (!response.ok) {
    throw new Error(`Failed to save architecture: ${response.statusText}`)
  }

  return await response.json()
}

/**
 * Load architecture (nodes and edges) for a project
 */
export async function loadArchitecture(projectId: string): Promise<{
  nodes: Node<BlockData>[]
  edges: Edge[]
}> {
  const response = await fetch(`${API_BASE_URL}/projects/${projectId}/load-architecture`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
  })

  if (!response.ok) {
    throw new Error(`Failed to load architecture: ${response.statusText}`)
  }

  return await response.json()
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
