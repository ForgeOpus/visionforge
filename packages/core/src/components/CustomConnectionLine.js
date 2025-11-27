import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useModelBuilderStore } from '../lib/store';
import { getNodeDefinition, BackendFramework } from '../lib/nodes/registry';
export default function CustomConnectionLine({ fromX, fromY, toX, toY, fromNode, fromHandle }) {
    const { nodes } = useModelBuilderStore();
    // Get source node
    const sourceNode = fromNode ? nodes.find(n => n.id === fromNode.id) : null;
    // Calculate position for tooltip
    const midX = (fromX + toX) / 2;
    const midY = (fromY + toY) / 2;
    // Default to valid connection line
    let strokeColor = '#6366f1';
    let errorMessage = null;
    // Check if we're hovering over a target node
    const targetNode = nodes.find(n => {
        const nodeEl = document.querySelector(`[data-id="${n.id}"]`);
        if (!nodeEl)
            return false;
        const rect = nodeEl.getBoundingClientRect();
        return (toX >= rect.left &&
            toX <= rect.right &&
            toY >= rect.top &&
            toY <= rect.bottom);
    });
    // Validate connection if we have both nodes
    if (sourceNode && targetNode) {
        const targetNodeDef = getNodeDefinition(targetNode.data.blockType, BackendFramework.PyTorch);
        if (targetNodeDef) {
            const validationError = targetNodeDef.validateIncomingConnection(sourceNode.data.blockType, sourceNode.data.outputShape, targetNode.data.config);
            if (validationError) {
                strokeColor = 'var(--color-destructive)';
                errorMessage = validationError;
            }
        }
    }
    return (_jsxs("g", { children: [_jsx("path", { d: `M ${fromX} ${fromY} C ${fromX + 50} ${fromY}, ${toX - 50} ${toY}, ${toX} ${toY}`, fill: "none", stroke: strokeColor, strokeWidth: 3, strokeDasharray: "5,5", className: "animated-dash" }), _jsx("circle", { cx: toX, cy: toY, r: 5, fill: strokeColor, stroke: "white", strokeWidth: 2 }), errorMessage && (_jsxs("g", { transform: `translate(${midX}, ${midY - 30})`, children: [_jsx("rect", { x: -100, y: -20, width: 200, height: 40, rx: 4, fill: "var(--color-destructive)", opacity: 0.95 }), _jsxs("text", { x: 0, y: 0, textAnchor: "middle", dominantBaseline: "middle", fill: "white", fontSize: 11, fontWeight: "500", children: [_jsx("tspan", { x: 0, dy: -5, children: "\u274C Connection Invalid" }), _jsx("tspan", { x: 0, dy: 15, fontSize: 9, children: errorMessage.length > 40
                                    ? errorMessage.substring(0, 40) + '...'
                                    : errorMessage })] })] }))] }));
}
