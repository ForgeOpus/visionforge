import { ComponentProps } from "react";
import * as ResizablePrimitive from "react-resizable-panels";
declare function ResizablePanelGroup({ className, ...props }: ComponentProps<typeof ResizablePrimitive.PanelGroup>): import("react/jsx-runtime").JSX.Element;
declare function ResizablePanel({ ...props }: ComponentProps<typeof ResizablePrimitive.Panel>): import("react/jsx-runtime").JSX.Element;
declare function ResizableHandle({ withHandle, className, ...props }: ComponentProps<typeof ResizablePrimitive.PanelResizeHandle> & {
    withHandle?: boolean;
}): import("react/jsx-runtime").JSX.Element;
export { ResizablePanelGroup, ResizablePanel, ResizableHandle };
//# sourceMappingURL=resizable.d.ts.map