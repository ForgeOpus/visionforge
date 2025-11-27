import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useMemo } from 'react';
import { ScrollArea } from './ui/scroll-area';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from './ui/accordion';
import { Card } from './ui/card';
import { Input } from './ui/input';
import { getAllNodeDefinitions, BackendFramework } from '../lib/nodes/registry';
import * as Icons from '@phosphor-icons/react';
import Fuse from 'fuse.js';
export default function BlockPalette({ onDragStart, onBlockClick, isCollapsed }) {
    const [searchQuery, setSearchQuery] = useState('');
    const categories = [
        { key: 'input', label: 'Input & Data', icon: Icons.DownloadSimple },
        { key: 'basic', label: 'Base Layers', icon: Icons.SquaresFour },
        { key: 'activation', label: 'Activation Functions', icon: Icons.Lightning },
        { key: 'advanced', label: 'Advanced Layers', icon: Icons.CubeFocus },
        { key: 'merge', label: 'Operations', icon: Icons.Unite },
        { key: 'output', label: 'Output & Loss', icon: Icons.UploadSimple },
        { key: 'utility', label: 'Utility', icon: Icons.Wrench }
    ];
    // Prepare all blocks for fuzzy search - maintain category order
    const allBlocks = useMemo(() => {
        const categoryOrder = ['input', 'basic', 'activation', 'advanced', 'merge', 'output', 'utility'];
        const nodes = getAllNodeDefinitions(BackendFramework.PyTorch);
        // Group by category
        const nodesByCategory = new Map();
        nodes.forEach(node => {
            const cat = node.metadata.category;
            if (!nodesByCategory.has(cat)) {
                nodesByCategory.set(cat, []);
            }
            nodesByCategory.get(cat).push(node);
        });
        // Build ordered list
        const orderedNodes = [];
        categoryOrder.forEach(category => {
            const categoryNodes = nodesByCategory.get(category) || [];
            orderedNodes.push(...categoryNodes);
        });
        // Add any remaining categories not in the order list
        nodesByCategory.forEach((nodes, category) => {
            if (!categoryOrder.includes(category)) {
                orderedNodes.push(...nodes);
            }
        });
        const blocks = orderedNodes.map(node => ({
            type: node.metadata.type,
            label: node.metadata.label,
            category: node.metadata.category,
            color: node.metadata.color,
            icon: node.metadata.icon,
            description: node.metadata.description
        }));
        // Debug: log all icons
        console.log('Block icons loaded:', blocks.map(b => `${b.label}: ${b.icon}`));
        return blocks;
    }, []);
    // Setup fuzzy search
    const fuse = useMemo(() => {
        return new Fuse(allBlocks, {
            keys: ['label', 'description', 'type'],
            threshold: 0.3,
            includeScore: true
        });
    }, [allBlocks]);
    // Filter blocks based on search
    const filteredBlocks = useMemo(() => {
        if (!searchQuery.trim()) {
            return null; // Return null to show categorized view
        }
        const results = fuse.search(searchQuery);
        return results.map(result => result.item);
    }, [searchQuery, fuse]);
    const handleDragStart = (type) => {
        window.draggedBlockTypeGlobal = type;
        onDragStart(type);
    };
    const renderBlockCard = (block) => {
        const IconComponent = Icons[block.icon];
        // Debug: log if icon is missing
        if (!IconComponent && block.icon) {
            console.warn(`Icon "${block.icon}" not found for block "${block.label}" (${block.type})`);
        }
        const FinalIcon = IconComponent || Icons.Cube;
        return (_jsx(Card, { className: "p-2 cursor-pointer hover:shadow-md hover:scale-[1.02] transition-all overflow-hidden", draggable: true, onDragStart: (e) => {
                e.dataTransfer.effectAllowed = 'move';
                handleDragStart(block.type);
            }, onClick: () => onBlockClick(block.type), children: _jsxs("div", { className: "flex items-center gap-2 min-w-0", children: [_jsx("div", { className: "p-1.5 rounded shrink-0", style: {
                            backgroundColor: block.color,
                            color: 'white'
                        }, children: _jsx(FinalIcon, { size: 14, weight: "bold" }) }), _jsxs("div", { className: "flex-1 min-w-0 overflow-hidden", children: [_jsx("div", { className: "text-sm font-medium truncate", children: block.label }), _jsx("div", { className: "text-[10px] text-muted-foreground truncate", children: block.description })] })] }) }, block.type));
    };
    if (isCollapsed) {
        return (_jsx("div", { className: "w-full bg-card h-full flex flex-col items-center relative overflow-hidden", children: _jsx(ScrollArea, { className: "flex-1 w-full min-h-0", children: _jsx("div", { className: "py-2 space-y-1 flex flex-col items-center px-2", children: allBlocks.map((block) => {
                        const IconComponent = Icons[block.icon];
                        // Debug: log if icon is missing
                        if (!IconComponent && block.icon) {
                            console.warn(`Icon "${block.icon}" not found for block "${block.label}" (${block.type})`);
                        }
                        const FinalIcon = IconComponent || Icons.Cube;
                        return (_jsxs("button", { className: "w-12 h-12 rounded flex items-center justify-center hover:bg-accent transition-colors cursor-pointer group relative flex-shrink-0", draggable: true, onDragStart: (e) => {
                                e.dataTransfer.effectAllowed = 'move';
                                handleDragStart(block.type);
                            }, onClick: () => onBlockClick(block.type), title: block.label, style: {
                                backgroundColor: 'transparent'
                            }, children: [_jsx("div", { className: "w-8 h-8 rounded flex items-center justify-center", style: {
                                        backgroundColor: block.color,
                                        color: 'white'
                                    }, children: _jsx(FinalIcon, { size: 16, weight: "bold" }) }), _jsx("div", { className: "absolute left-full ml-2 px-2 py-1 bg-popover text-popover-foreground text-xs rounded shadow-md border border-border whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50", children: block.label })] }, block.type));
                    }) }) }) }));
    }
    return (_jsxs("div", { className: "w-full bg-card h-full flex flex-col relative", children: [_jsx("div", { className: "p-3 border-b border-border sticky top-0 bg-card z-10", children: _jsxs("div", { className: "relative", children: [_jsx(Icons.MagnifyingGlass, { size: 16, className: "absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" }), _jsx(Input, { type: "text", placeholder: "Search blocks...", value: searchQuery, onChange: (e) => setSearchQuery(e.target.value), className: "pl-9 h-9" }), searchQuery && (_jsx("button", { onClick: () => setSearchQuery(''), className: "absolute right-2 top-1/2 -translate-y-1/2 p-1 hover:bg-accent rounded transition-colors", children: _jsx(Icons.X, { size: 14 }) }))] }) }), _jsx(ScrollArea, { className: "flex-1 overflow-y-auto", children: _jsx("div", { className: "h-full", children: filteredBlocks !== null ? (
                    // Search results view
                    _jsx("div", { className: "p-2 space-y-2", children: filteredBlocks.length > 0 ? (filteredBlocks.map((block) => renderBlockCard(block))) : (_jsxs("div", { className: "text-center text-muted-foreground p-6", children: [_jsx(Icons.MagnifyingGlass, { size: 32, className: "mx-auto mb-2 opacity-50" }), _jsx("p", { className: "text-sm", children: "No blocks found" }), _jsx("p", { className: "text-xs mt-1", children: "Try a different search term" })] })) })) : (
                    // Categorized view
                    _jsx(Accordion, { type: "multiple", defaultValue: ['input', 'basic', 'activation'], className: "px-2 py-2", children: categories.map((category) => {
                            const blocks = allBlocks.filter(b => b.category === category.key);
                            const CategoryIcon = category.icon;
                            return (_jsxs(AccordionItem, { value: category.key, children: [_jsx(AccordionTrigger, { className: "text-sm font-medium", children: _jsxs("div", { className: "flex items-center gap-2", children: [_jsx(CategoryIcon, { size: 16 }), category.label] }) }), _jsx(AccordionContent, { children: _jsx("div", { className: "space-y-2 pt-2", children: blocks.map((block) => renderBlockCard(block)) }) })] }, category.key));
                        }) })) }) })] }));
}
