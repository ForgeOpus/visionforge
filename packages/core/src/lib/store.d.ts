import { Node, Edge, Connection } from '@xyflow/react';
import { BlockData, Project, ValidationError, BlockType } from './types';
interface HistoryState {
    nodes: Node<BlockData>[];
    edges: Edge[];
}
interface ModelBuilderState {
    nodes: Node<BlockData>[];
    edges: Edge[];
    selectedNodeId: string | null;
    selectedEdgeId: string | null;
    recentlyUsedNodes: BlockType[];
    validationErrors: ValidationError[];
    currentProject: Project | null;
    past: HistoryState[];
    future: HistoryState[];
    setNodes: (nodes: Node<BlockData>[]) => void;
    setEdges: (edges: Edge[]) => void;
    addNode: (node: Node<BlockData>) => void;
    updateNode: (id: string, data: Partial<BlockData>) => void;
    removeNode: (id: string) => void;
    duplicateNode: (id: string) => void;
    addEdge: (edge: Edge) => void;
    removeEdge: (id: string) => void;
    setSelectedNodeId: (id: string | null) => void;
    setSelectedEdgeId: (id: string | null) => void;
    trackRecentlyUsedNode: (nodeType: BlockType) => void;
    validateConnection: (connection: Connection) => boolean;
    validateArchitecture: () => ValidationError[];
    inferDimensions: () => void;
    undo: () => void;
    redo: () => void;
    canUndo: () => boolean;
    canRedo: () => boolean;
    createProject: (name: string, description: string, framework: 'pytorch' | 'tensorflow') => void;
    saveProject: () => void;
    loadProject: (project: Project) => void;
    updateProjectInfo: (name: string, description: string) => void;
    reset: () => void;
}
export declare const useModelBuilderStore: import("zustand").UseBoundStore<import("zustand").StoreApi<ModelBuilderState>>;
export {};
//# sourceMappingURL=store.d.ts.map