import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useCallback, useRef, useEffect, useState, useMemo } from 'react';
import { ReactFlow, Background, Controls, MiniMap, useReactFlow, ReactFlowProvider, applyNodeChanges, applyEdgeChanges } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useModelBuilderStore } from '../lib/store';
import { getNodeDefinition, BackendFramework } from '../lib/nodes/registry';
import BlockNode from './BlockNode';
import CustomConnectionLine from './CustomConnectionLine';
import { HistoryToolbar } from './HistoryToolbar';
import { ContextMenu } from './ContextMenu';
import ViewCodeModal from './ViewCodeModal';
import { renderNodeCode } from '../lib/api';
import { toast } from 'sonner';
const nodeTypes = {
    custom: BlockNode
};
function FlowCanvas({ onRegisterAddNode }) {
    const { nodes, edges, setNodes, setEdges, addNode, addEdge, removeEdge, removeNode, selectedNodeId, setSelectedNodeId, selectedEdgeId, setSelectedEdgeId, duplicateNode, recentlyUsedNodes, validateConnection, undo, redo } = useModelBuilderStore();
    const { screenToFlowPosition, getViewport } = useReactFlow();
    const nextPositionOffset = useRef({ x: 0, y: 0 });
    const [contextMenu, setContextMenu] = useState(null);
    // ViewCodeModal state
    const [isViewCodeModalOpen, setIsViewCodeModalOpen] = useState(false);
    const [viewCodeData, setViewCodeData] = useState({
        code: '',
        nodeType: '',
        framework: 'pytorch'
    });
    const [isLoadingCode, setIsLoadingCode] = useState(false);
    const currentProject = useModelBuilderStore((state) => state.currentProject);
    // Keyboard shortcuts for undo/redo/delete
    useEffect(() => {
        const handleKeyDown = (e) => {
            // Check for Ctrl (Windows/Linux) or Cmd (Mac)
            const isMod = e.ctrlKey || e.metaKey;
            if (isMod && e.key === 'z' && !e.shiftKey) {
                e.preventDefault();
                undo();
            }
            else if (isMod && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
                e.preventDefault();
                redo();
            }
            else if ((e.key === 'Delete' || e.key === 'Backspace')) {
                // Only delete if not typing in an input field
                const target = e.target;
                if (target.tagName !== 'INPUT' && target.tagName !== 'TEXTAREA' && !target.isContentEditable) {
                    e.preventDefault();
                    if (selectedNodeId) {
                        removeNode(selectedNodeId);
                    }
                    else if (selectedEdgeId) {
                        removeEdge(selectedEdgeId);
                        setSelectedEdgeId(null);
                    }
                }
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [undo, redo, removeNode, removeEdge, selectedNodeId, selectedEdgeId, setSelectedEdgeId]);
    // Find a suitable position for a new node
    const findAvailablePosition = useCallback(() => {
        const viewport = getViewport();
        const GRID_SIZE = 50;
        // Start from center of viewport
        const centerX = -viewport.x / viewport.zoom + (window.innerWidth / 2) / viewport.zoom;
        const centerY = -viewport.y / viewport.zoom + (window.innerHeight / 2) / viewport.zoom;
        // Use offset for new nodes (allows overlapping)
        const x = centerX + nextPositionOffset.current.x;
        const y = centerY + nextPositionOffset.current.y;
        // Update offset for next node
        nextPositionOffset.current = {
            x: (nextPositionOffset.current.x + GRID_SIZE) % (GRID_SIZE * 4),
            y: nextPositionOffset.current.y
        };
        if (nextPositionOffset.current.x === 0) {
            nextPositionOffset.current.y = (nextPositionOffset.current.y + GRID_SIZE) % (GRID_SIZE * 4);
        }
        return { x, y };
    }, [getViewport]);
    // Handle block click from palette
    useEffect(() => {
        const handleBlockClickInternal = (blockType) => {
            const nodeDef = getNodeDefinition(blockType, BackendFramework.PyTorch);
            if (!nodeDef)
                return;
            const position = findAvailablePosition();
            const newNode = {
                id: `${blockType}-${Date.now()}`,
                type: 'custom',
                position,
                data: {
                    blockType: nodeDef.metadata.type,
                    label: nodeDef.metadata.label,
                    config: {},
                    category: nodeDef.metadata.category
                }
            };
            nodeDef.configSchema.forEach((field) => {
                if (field.default !== undefined) {
                    newNode.data.config[field.name] = field.default;
                }
            });
            addNode(newNode);
            setTimeout(() => {
                useModelBuilderStore.getState().inferDimensions();
            }, 0);
            toast.success(`Added ${nodeDef.metadata.label}`, {
                description: 'Block added to canvas'
            });
        };
        // Register the handler with parent
        onRegisterAddNode(handleBlockClickInternal);
    }, [addNode, findAvailablePosition, onRegisterAddNode]);
    const onDragOver = useCallback((event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);
    const onDrop = useCallback((event) => {
        event.preventDefault();
        const type = window.draggedBlockTypeGlobal;
        if (!type)
            return;
        const nodeDef = getNodeDefinition(type, BackendFramework.PyTorch);
        if (!nodeDef)
            return;
        const position = screenToFlowPosition({
            x: event.clientX,
            y: event.clientY
        });
        const newNode = {
            id: `${type}-${Date.now()}`,
            type: 'custom',
            position,
            data: {
                blockType: nodeDef.metadata.type,
                label: nodeDef.metadata.label,
                config: {},
                category: nodeDef.metadata.category
            }
        };
        nodeDef.configSchema.forEach((field) => {
            if (field.default !== undefined) {
                newNode.data.config[field.name] = field.default;
            }
        });
        addNode(newNode);
        setTimeout(() => {
            useModelBuilderStore.getState().inferDimensions();
        }, 0);
        window.draggedBlockTypeGlobal = null;
    }, [addNode, screenToFlowPosition]);
    const onConnect = useCallback((connection) => {
        const isValid = validateConnection(connection);
        if (!isValid) {
            const sourceNode = nodes.find((n) => n.id === connection.source);
            const targetNode = nodes.find((n) => n.id === connection.target);
            // Use the validation function to get specific error message
            if (sourceNode && targetNode) {
                const targetNodeDef = getNodeDefinition(targetNode.data.blockType, BackendFramework.PyTorch);
                if (!targetNodeDef) {
                    toast.error('Connection Invalid', {
                        description: 'Invalid target node type'
                    });
                    return;
                }
                const errorMessage = targetNodeDef.validateIncomingConnection(sourceNode.data.blockType, sourceNode.data.outputShape, targetNode.data.config);
                if (errorMessage) {
                    toast.error('Connection Invalid', {
                        description: errorMessage
                    });
                    return;
                }
            }
            // Fallback for other validation errors
            if (targetNode?.data.blockType !== 'concat' && targetNode?.data.blockType !== 'add' && targetNode?.data.blockType !== 'loss') {
                const hasInput = edges.some((e) => e.target === connection.target);
                if (hasInput) {
                    toast.error('Block already has an input connection', {
                        description: 'Use a Concatenate or Add block for multiple inputs'
                    });
                    return;
                }
            }
            toast.error('Connection not allowed');
            return;
        }
        const edge = {
            ...connection,
            id: `e${connection.source}-${connection.target}-${Date.now()}`,
            animated: true,
            style: { stroke: '#6366f1', strokeWidth: 2 }
        };
        addEdge(edge);
        toast.success('Connection created', {
            description: 'Input shape automatically configured'
        });
    }, [validateConnection, addEdge, nodes, edges]);
    const onNodesChange = useCallback((changes) => {
        const updatedNodes = applyNodeChanges(changes, nodes);
        setNodes(updatedNodes);
    }, [nodes, setNodes]);
    const onEdgesChange = useCallback((changes) => {
        setEdges(applyEdgeChanges(changes, edges));
        // Handle edge removal for store cleanup
        changes.forEach((change) => {
            if (change.type === 'remove') {
                removeEdge(change.id);
            }
        });
    }, [edges, setEdges, removeEdge]);
    const onNodeClick = useCallback((_event, node) => {
        setSelectedNodeId(node.id);
    }, [setSelectedNodeId]);
    const onPaneClick = useCallback(() => {
        setSelectedNodeId(null);
        setSelectedEdgeId(null);
    }, [setSelectedNodeId, setSelectedEdgeId]);
    const onEdgeClick = useCallback((_event, edge) => {
        setSelectedEdgeId(edge.id);
    }, [setSelectedEdgeId]);
    const onPaneContextMenu = useCallback((event) => {
        event.preventDefault();
        setContextMenu({
            x: event.clientX,
            y: event.clientY,
            type: 'canvas'
        });
    }, []);
    const onNodeContextMenu = useCallback((event, node) => {
        event.preventDefault();
        setContextMenu({
            x: event.clientX,
            y: event.clientY,
            type: 'node',
            nodeId: node.id
        });
    }, []);
    const onReconnect = useCallback((oldEdge, newConnection) => {
        const isValid = validateConnection(newConnection);
        if (!isValid) {
            toast.error('Connection not allowed', {
                description: 'This reconnection is not valid'
            });
            return;
        }
        // Remove old edge
        removeEdge(oldEdge.id);
        // Add new edge
        const edge = {
            ...newConnection,
            id: `e${newConnection.source}-${newConnection.target}-${Date.now()}`,
            animated: true,
            style: { stroke: '#6366f1', strokeWidth: 2 }
        };
        addEdge(edge);
        toast.success('Connection reconnected', {
            description: 'Edge successfully reconnected'
        });
    }, [validateConnection, removeEdge, addEdge]);
    const handleAddNodeFromContextMenu = useCallback((nodeType, x, y) => {
        const nodeDef = getNodeDefinition(nodeType, BackendFramework.PyTorch);
        if (!nodeDef)
            return;
        const position = screenToFlowPosition({ x, y });
        const newNode = {
            id: `${nodeType}-${Date.now()}`,
            type: 'custom',
            position,
            data: {
                blockType: nodeDef.metadata.type,
                label: nodeDef.metadata.label,
                config: {},
                category: nodeDef.metadata.category
            }
        };
        nodeDef.configSchema.forEach((field) => {
            if (field.default !== undefined) {
                newNode.data.config[field.name] = field.default;
            }
        });
        addNode(newNode);
        setTimeout(() => {
            useModelBuilderStore.getState().inferDimensions();
        }, 0);
        toast.success(`Added ${nodeDef.metadata.label}`, {
            description: 'Block added to canvas'
        });
    }, [addNode, screenToFlowPosition]);
    const handleViewCode = useCallback(async (nodeId) => {
        const node = nodes.find(n => n.id === nodeId);
        if (!node)
            return;
        setIsLoadingCode(true);
        setIsViewCodeModalOpen(true);
        try {
            const response = await renderNodeCode(node.data.blockType, currentProject?.framework || 'pytorch', node.data.config, {
                node_id: nodeId,
                inputShape: node.data.inputShape,
                outputShape: node.data.outputShape
            });
            if (response.success && response.data) {
                setViewCodeData({
                    code: response.data.code,
                    nodeType: node.data.label,
                    framework: currentProject?.framework || 'pytorch'
                });
            }
            else {
                throw new Error(response.error || 'Failed to fetch code');
            }
        }
        catch (error) {
            toast.error('Failed to fetch code', {
                description: error instanceof Error ? error.message : 'Network error - check backend connection'
            });
            setIsViewCodeModalOpen(false);
        }
        finally {
            setIsLoadingCode(false);
        }
    }, [nodes, currentProject]);
    const handleReplicateAsCustom = useCallback(async (nodeId) => {
        const sourceNode = nodes.find(n => n.id === nodeId);
        if (!sourceNode || sourceNode.data.blockType === 'custom') {
            toast.error('Cannot replicate custom blocks');
            return;
        }
        const toastId = toast.loading('Replicating block...');
        try {
            // 1. Fetch code from backend
            const response = await renderNodeCode(sourceNode.data.blockType, currentProject?.framework || 'pytorch', sourceNode.data.config, {
                node_id: nodeId,
                inputShape: sourceNode.data.inputShape,
                outputShape: sourceNode.data.outputShape
            });
            if (!response.success || !response.data) {
                throw new Error(response.error || 'Failed to fetch code');
            }
            // 2. Create new custom block positioned near original
            const newPosition = {
                x: sourceNode.position.x + 250,
                y: sourceNode.position.y
            };
            const customNodeId = `custom-${Date.now()}`;
            const customNode = {
                id: customNodeId,
                type: 'custom',
                position: newPosition,
                data: {
                    blockType: 'custom',
                    label: `Custom ${sourceNode.data.label}`,
                    config: {
                        name: `custom_${sourceNode.data.blockType}`,
                        code: response.data.code,
                        output_shape: sourceNode.data.outputShape
                            ? JSON.stringify(sourceNode.data.outputShape.dims)
                            : undefined,
                        description: `Replicated from ${sourceNode.data.label}`,
                        // Preserve original config parameters
                        ...sourceNode.data.config
                    },
                    category: 'utility',
                    inputShape: sourceNode.data.inputShape,
                    outputShape: sourceNode.data.outputShape
                }
            };
            // 3. Add to canvas
            addNode(customNode);
            // 4. Select new node (triggers ConfigPanel to show CustomLayerModal)
            setSelectedNodeId(customNodeId);
            // 5. Trigger dimension inference
            setTimeout(() => {
                useModelBuilderStore.getState().inferDimensions();
            }, 0);
            toast.dismiss(toastId);
            toast.success('Block replicated as custom layer', {
                description: 'Click on the new block to edit its code'
            });
        }
        catch (error) {
            toast.dismiss(toastId);
            toast.error('Replication failed', {
                description: error instanceof Error ? error.message : 'Unknown error'
            });
        }
    }, [nodes, currentProject, addNode, setSelectedNodeId]);
    const handleReplicateNode = useCallback((nodeId) => {
        const node = nodes.find(n => n.id === nodeId);
        if (!node)
            return;
        const nodeDef = getNodeDefinition(node.data.blockType, BackendFramework.PyTorch);
        if (!nodeDef)
            return;
        const newNode = {
            id: `custom-${Date.now()}`,
            type: 'custom',
            position: {
                x: node.position.x + 50,
                y: node.position.y + 50
            },
            data: {
                blockType: 'custom',
                label: `Custom ${nodeDef.metadata.label}`,
                config: { ...node.data.config },
                category: node.data.category
            }
        };
        addNode(newNode);
        setTimeout(() => {
            useModelBuilderStore.getState().inferDimensions();
        }, 0);
        toast.success('Node replicated as custom', {
            description: 'Custom node created from original'
        });
    }, [nodes, addNode]);
    // Memoized nodes with action handlers attached
    const nodesWithHandlers = useMemo(() => {
        return nodes.map(node => ({
            ...node,
            data: {
                ...node.data,
                onViewCode: handleViewCode,
                onReplicate: handleReplicateAsCustom
            }
        }));
    }, [nodes, handleViewCode, handleReplicateAsCustom]);
    return (_jsxs("div", { className: "flex-1 h-full", onDrop: onDrop, onDragOver: onDragOver, children: [_jsx(HistoryToolbar, {}), _jsxs(ReactFlow, { nodes: nodesWithHandlers, edges: edges, onNodesChange: onNodesChange, onEdgesChange: onEdgesChange, onConnect: onConnect, onNodeClick: onNodeClick, onEdgeClick: onEdgeClick, onPaneClick: onPaneClick, onPaneContextMenu: onPaneContextMenu, onNodeContextMenu: onNodeContextMenu, onReconnect: onReconnect, edgesReconnectable: true, nodeTypes: nodeTypes, connectionLineComponent: CustomConnectionLine, fitView: true, minZoom: 0.5, maxZoom: 1.5, defaultEdgeOptions: {
                    animated: true,
                    style: { stroke: '#6366f1', strokeWidth: 2 }
                }, children: [_jsx(Background, { gap: 20, size: 1 }), _jsx(Controls, {}), _jsx(MiniMap, { nodeColor: (node) => {
                            const nodeDef = getNodeDefinition(node.data.blockType, BackendFramework.PyTorch);
                            return nodeDef?.metadata.color || '#3b82f6';
                        }, nodeStrokeColor: (node) => {
                            const nodeDef = getNodeDefinition(node.data.blockType, BackendFramework.PyTorch);
                            const baseColor = nodeDef?.metadata.color || '#3b82f6';
                            // Return a slightly darker version for the stroke
                            return baseColor;
                        }, nodeStrokeWidth: 2, nodeBorderRadius: 4, maskColor: "rgba(0, 0, 0, 0.05)", className: "bg-card border border-border rounded-lg shadow-lg", style: {
                            backgroundColor: 'var(--card)',
                            borderColor: 'var(--border)',
                        }, zoomable: true, pannable: true, position: "bottom-right" })] }), contextMenu && (_jsx(ContextMenu, { x: contextMenu.x, y: contextMenu.y, type: contextMenu.type, nodeId: contextMenu.nodeId, recentlyUsedNodes: recentlyUsedNodes, onClose: () => setContextMenu(null), onAddNode: handleAddNodeFromContextMenu, onDeleteNode: removeNode, onDuplicateNode: duplicateNode, onReplicateNode: handleReplicateNode })), _jsx(ViewCodeModal, { isOpen: isViewCodeModalOpen, onClose: () => setIsViewCodeModalOpen(false), code: viewCodeData.code, nodeType: viewCodeData.nodeType, framework: viewCodeData.framework, isLoading: isLoadingCode })] }));
}
export default function Canvas({ onDragStart, onRegisterAddNode }) {
    return (_jsx(ReactFlowProvider, { children: _jsx(FlowCanvas, { onRegisterAddNode: onRegisterAddNode }) }));
}
export const draggedBlockTypeGlobal = null;
