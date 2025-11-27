import { ComponentProps, ComponentType, ReactNode } from "react";
import * as RechartsPrimitive from "recharts";
declare const THEMES: {
    readonly light: "";
    readonly dark: ".dark";
};
export type ChartConfig = {
    [k in string]: {
        label?: ReactNode;
        icon?: ComponentType;
    } & ({
        color?: string;
        theme?: never;
    } | {
        color?: never;
        theme: Record<keyof typeof THEMES, string>;
    });
};
declare function ChartContainer({ id, className, children, config, ...props }: ComponentProps<"div"> & {
    config: ChartConfig;
    children: ComponentProps<typeof RechartsPrimitive.ResponsiveContainer>["children"];
}): any;
declare const ChartStyle: ({ id, config }: {
    id: string;
    config: ChartConfig;
}) => any;
declare const ChartTooltip: any;
declare function ChartTooltipContent({ active, payload, className, indicator, hideLabel, hideIndicator, label, labelFormatter, labelClassName, formatter, color, nameKey, labelKey, }: ComponentProps<typeof RechartsPrimitive.Tooltip> & ComponentProps<"div"> & {
    hideLabel?: boolean;
    hideIndicator?: boolean;
    indicator?: "line" | "dot" | "dashed";
    nameKey?: string;
    labelKey?: string;
}): any;
declare const ChartLegend: any;
declare function ChartLegendContent({ className, hideIcon, payload, verticalAlign, nameKey, }: ComponentProps<"div"> & Pick<RechartsPrimitive.LegendProps, "payload" | "verticalAlign"> & {
    hideIcon?: boolean;
    nameKey?: string;
}): any;
export { ChartContainer, ChartTooltip, ChartTooltipContent, ChartLegend, ChartLegendContent, ChartStyle, };
//# sourceMappingURL=chart.d.ts.map