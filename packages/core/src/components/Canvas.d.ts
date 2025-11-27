import '@xyflow/react/dist/style.css';
interface CanvasProps {
    onDragStart: (type: string) => void;
    onRegisterAddNode: (handler: (blockType: string) => void) => void;
}
export default function Canvas({ onDragStart, onRegisterAddNode }: CanvasProps): any;
export declare const draggedBlockTypeGlobal: null;
export {};
//# sourceMappingURL=Canvas.d.ts.map