import { Node, Edge } from '@xyflow/react'
import { BlockData, Project } from './types'
import { getAuthHeaders } from './auth'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'

export interface ProjectResponse {
  id: string
  name: string
  description: string
  framework: 'pytorch' | 'tensorflow'
  share_token: string | null
  is_shared: boolean
  created_at: string
  updated_at: string
}

export interface SharedProjectResponse {
  name: string
  description: string
  framework: 'pytorch' | 'tensorflow'
  owner_display_name: string | null
  share_token: string
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
  const headers = await getAuthHeaders()

  const response = await fetch(`${API_BASE_URL}/projects/`, {
    method: 'GET',
    headers,
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
  const headers = await getAuthHeaders()

  const response = await fetch(`${API_BASE_URL}/projects/${projectId}/`, {
    method: 'GET',
    headers,
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
  const headers = await getAuthHeaders()

  const response = await fetch(`${API_BASE_URL}/projects/`, {
    method: 'POST',
    headers,
    body: JSON.stringify(data),
  })

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
  const headers = await getAuthHeaders()

  const response = await fetch(`${API_BASE_URL}/projects/${projectId}/`, {
    method: 'PATCH',
    headers,
    body: JSON.stringify(data),
  })

  if (!response.ok) {
    throw new Error(`Failed to update project: ${response.statusText}`)
  }

  return await response.json()
}

/**
 * Delete a project
 */
export async function deleteProject(projectId: string): Promise<void> {
  const headers = await getAuthHeaders()

  const response = await fetch(`${API_BASE_URL}/projects/${projectId}/`, {
    method: 'DELETE',
    headers,
  })

  if (!response.ok) {
    throw new Error(`Failed to delete project: ${response.statusText}`)
  }
}

/**
 * Save architecture (nodes, edges, and group definitions) for a project
 */
export async function saveArchitecture(
  projectId: string,
  nodes: Node<BlockData>[],
  edges: Edge[],
  groupDefinitions?: Map<string, any>
): Promise<{ success: boolean; architecture_id: string }> {
  const headers = await getAuthHeaders()

  // Convert groupDefinitions Map to array for serialization
  const groupDefinitionsArray = groupDefinitions
    ? Array.from(groupDefinitions.values())
    : []

  const response = await fetch(`${API_BASE_URL}/projects/${projectId}/save-architecture`, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      nodes,
      edges,
      groupDefinitions: groupDefinitionsArray
    }),
  })

  if (!response.ok) {
    throw new Error(`Failed to save architecture: ${response.statusText}`)
  }

  return await response.json()
}

/**
 * Load architecture (nodes, edges, and group definitions) for a project
 */
export async function loadArchitecture(projectId: string): Promise<{
  nodes: Node<BlockData>[]
  edges: Edge[]
  groupDefinitions?: any[]
}> {
  const headers = await getAuthHeaders()

  const response = await fetch(`${API_BASE_URL}/projects/${projectId}/load-architecture`, {
    method: 'GET',
    headers,
  })

  if (!response.ok) {
    throw new Error(`Failed to load architecture: ${response.statusText}`)
  }

  return await response.json()
}

/**
 * Fetch a shared project's metadata (public — no auth required)
 */
export async function fetchSharedProject(shareToken: string): Promise<SharedProjectResponse> {
  const response = await fetch(`${API_BASE_URL}/shared/${shareToken}/`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  })

  if (!response.ok) {
    throw new Error(`Shared project not found: ${response.statusText}`)
  }

  return await response.json()
}

/**
 * Fetch a shared project's architecture (public — no auth required)
 */
export async function fetchSharedArchitecture(shareToken: string): Promise<{
  nodes: Node<BlockData>[]
  edges: Edge[]
  groupDefinitions?: any[]
}> {
  const response = await fetch(`${API_BASE_URL}/shared/${shareToken}/architecture/`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  })

  if (!response.ok) {
    throw new Error(`Failed to load shared architecture: ${response.statusText}`)
  }

  return await response.json()
}

/**
 * Enable public sharing for a project (owner only)
 */
export async function enableSharing(
  projectId: string
): Promise<{ share_token: string; is_shared: boolean }> {
  const headers = await getAuthHeaders()

  const response = await fetch(`${API_BASE_URL}/projects/${projectId}/share/`, {
    method: 'POST',
    headers,
  })

  if (!response.ok) {
    throw new Error(`Failed to enable sharing: ${response.statusText}`)
  }

  return await response.json()
}

/**
 * Disable public sharing for a project (owner only).
 * The share token is preserved so re-enabling restores the same URL.
 */
export async function disableSharing(projectId: string): Promise<void> {
  const headers = await getAuthHeaders()

  const response = await fetch(`${API_BASE_URL}/projects/${projectId}/unshare/`, {
    method: 'DELETE',
    headers,
  })

  if (!response.ok) {
    throw new Error(`Failed to disable sharing: ${response.statusText}`)
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
    share_token: backendProject.share_token ?? null,
    is_shared: backendProject.is_shared ?? false,
  }
}
