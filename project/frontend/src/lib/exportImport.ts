import { Node, Edge } from '@xyflow/react'
import { BlockData, Project } from './types'

/**
 * Sanitized project export format
 * This format excludes position data and internal IDs to keep the export clean
 * but includes all necessary configuration to rebuild the architecture
 */
export interface ExportData {
  version: string
  projectName: string
  projectDescription: string
  framework: 'pytorch' | 'tensorflow'
  architecture: {
    nodes: Array<{
      id: string
      type: string
      label: string
      category: string
      config: Record<string, any>
      inputShape?: { dims: (number | string)[] }
      outputShape?: { dims: (number | string)[] }
    }>
    connections: Array<{
      from: string
      to: string
    }>
  }
  metadata: {
    exportedAt: number
    nodeCount: number
    edgeCount: number
  }
}

/**
 * Export nodes and edges to a sanitized JSON format
 * Removes implementation details and internal React Flow state
 */
export function exportToJSON(
  nodes: Node<BlockData>[],
  edges: Edge[],
  project?: Project | null
): ExportData {
  return {
    version: '1.0.0',
    projectName: project?.name || 'Untitled Project',
    projectDescription: project?.description || '',
    framework: project?.framework || 'pytorch',
    architecture: {
      nodes: nodes.map((node) => ({
        id: node.id,
        type: node.data.blockType,
        label: node.data.label,
        category: node.data.category,
        config: node.data.config,
        inputShape: node.data.inputShape,
        outputShape: node.data.outputShape
      })),
      connections: edges.map((edge) => ({
        from: edge.source,
        to: edge.target
      }))
    },
    metadata: {
      exportedAt: Date.now(),
      nodeCount: nodes.length,
      edgeCount: edges.length
    }
  }
}

/**
 * Import from sanitized JSON format and reconstruct nodes and edges
 * Validates the data structure before importing
 */
export function importFromJSON(
  jsonData: ExportData
): {
  nodes: Node<BlockData>[]
  edges: Edge[]
  project: Partial<Project>
} {
  // Validate JSON structure
  if (!jsonData.version || !jsonData.architecture) {
    throw new Error('Invalid export file format')
  }

  if (jsonData.version !== '1.0.0') {
    throw new Error(`Unsupported export version: ${jsonData.version}`)
  }

  // Reconstruct nodes with proper React Flow format
  const nodes: Node<BlockData>[] = jsonData.architecture.nodes.map((nodeData, index) => ({
    id: nodeData.id,
    type: 'block',
    position: {
      // Create a grid layout for imported nodes
      x: (index % 4) * 250,
      y: Math.floor(index / 4) * 150
    },
    data: {
      blockType: nodeData.type as any,
      label: nodeData.label,
      category: nodeData.category as any,
      config: nodeData.config,
      inputShape: nodeData.inputShape,
      outputShape: nodeData.outputShape
    }
  }))

  // Reconstruct edges
  const edges: Edge[] = jsonData.architecture.connections.map((conn, index) => ({
    id: `e${conn.from}-${conn.to}-${index}`,
    source: conn.from,
    target: conn.to,
    animated: true
  }))

  // Create project metadata
  const project: Partial<Project> = {
    name: jsonData.projectName,
    description: jsonData.projectDescription,
    framework: jsonData.framework,
    nodes,
    edges
  }

  return { nodes, edges, project }
}

/**
 * Download JSON data as a file
 */
export function downloadJSON(data: ExportData, filename?: string): void {
  const json = JSON.stringify(data, null, 2)
  const blob = new Blob([json], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename || `${data.projectName.replace(/\s+/g, '_')}_architecture.json`
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

/**
 * Read and parse JSON file from user upload
 */
export function readJSONFile(file: File): Promise<ExportData> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()

    reader.onload = (event) => {
      try {
        const jsonData = JSON.parse(event.target?.result as string)
        resolve(jsonData)
      } catch (error) {
        reject(new Error('Invalid JSON file'))
      }
    }

    reader.onerror = () => {
      reject(new Error('Failed to read file'))
    }

    reader.readAsText(file)
  })
}
