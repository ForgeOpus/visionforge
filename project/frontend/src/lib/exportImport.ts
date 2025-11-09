import { Node, Edge } from '@xyflow/react'
import { BlockData, Project } from './types'

/**
 * Complete project export format
 * This format includes ALL data needed to perfectly restore the architecture
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
      position: { x: number; y: number }
      data: {
        blockType: string
        label: string
        category: string
        config: Record<string, any>
        inputShape?: { dims: (number | string)[]; description?: string }
        outputShape?: { dims: (number | string)[]; description?: string }
      }
    }>
    connections: Array<{
      id: string
      source: string
      target: string
      sourceHandle?: string | null
      targetHandle?: string | null
    }>
  }
  metadata: {
    exportedAt: number
    nodeCount: number
    edgeCount: number
  }
}

/**
 * Export nodes and edges to complete JSON format
 * Includes ALL data for perfect restoration
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
        type: node.type || 'block',
        position: node.position,
        data: {
          blockType: node.data.blockType,
          label: node.data.label,
          category: node.data.category,
          config: node.data.config,
          inputShape: node.data.inputShape,
          outputShape: node.data.outputShape
        }
      })),
      connections: edges.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        sourceHandle: edge.sourceHandle,
        targetHandle: edge.targetHandle
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
 * Import from complete JSON format and reconstruct nodes and edges
 * Validates the data structure before importing
 *
 * @param jsonData - The exported JSON data
 * @param existingNodes - Optional array of existing nodes to check for ID conflicts
 * @param existingEdges - Optional array of existing edges
 * @returns Reconstructed nodes, edges, and project metadata
 */
export function importFromJSON(
  jsonData: ExportData,
  existingNodes?: Node<BlockData>[],
  existingEdges?: Edge[]
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

  // Build a set of existing node IDs to detect conflicts
  const existingNodeIds = new Set(existingNodes?.map(n => n.id) || [])
  const idMapping = new Map<string, string>()

  // Reconstruct nodes with complete data restoration
  const nodes: Node<BlockData>[] = jsonData.architecture.nodes.map((nodeData) => {
    let nodeId = nodeData.id

    // If there's an ID conflict, generate a new unique ID
    if (existingNodeIds.has(nodeId)) {
      let counter = 1
      let newId = `${nodeId}_${counter}`
      while (existingNodeIds.has(newId)) {
        counter++
        newId = `${nodeId}_${counter}`
      }
      idMapping.set(nodeId, newId)
      nodeId = newId
      existingNodeIds.add(nodeId)
    } else {
      existingNodeIds.add(nodeId)
    }

    return {
      id: nodeId,
      type: nodeData.type || 'block',
      position: nodeData.position || { x: 0, y: 0 },
      data: {
        blockType: nodeData.data?.blockType || (nodeData as any).type,
        label: nodeData.data?.label || (nodeData as any).label || 'Node',
        category: nodeData.data?.category || (nodeData as any).category || 'basic',
        config: nodeData.data?.config || (nodeData as any).config || {},
        inputShape: nodeData.data?.inputShape || (nodeData as any).inputShape,
        outputShape: nodeData.data?.outputShape || (nodeData as any).outputShape
      }
    }
  })

  // Reconstruct edges with complete data
  const edges: Edge[] = jsonData.architecture.connections.map((conn) => {
    const sourceId = idMapping.get(conn.source || (conn as any).from) || conn.source || (conn as any).from
    const targetId = idMapping.get(conn.target || (conn as any).to) || conn.target || (conn as any).to

    return {
      id: conn.id || `e${sourceId}-${targetId}`,
      source: sourceId,
      target: targetId,
      sourceHandle: conn.sourceHandle,
      targetHandle: conn.targetHandle,
      animated: true
    }
  })

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
