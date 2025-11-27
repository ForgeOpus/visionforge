import { ComponentProps } from "react";
import * as ResizablePrimitive from "react-resizable-panels";
declare function ResizablePanelGroup({ className, ...props }: ComponentProps<typeof ResizablePrimitive.PanelGroup>): any;
declare function ResizablePanel({ ...props }: ComponentProps<typeof ResizablePrimitive.Panel>): any;
declare function ResizableHandle({ withHandle, className, ...props }: ComponentProps<typeof ResizablePrimitive.PanelResizeHandle> & {
    withHandle?: boolean;
}): any;
export { ResizablePanelGroup, ResizablePanel, ResizableHandle };
//# sourceMappingURL=resizable.d.ts.map