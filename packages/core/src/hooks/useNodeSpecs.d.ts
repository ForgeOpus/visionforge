/**
 * React hook for fetching and managing node specifications from backend
 */
import type { NodeSpec, Framework } from './nodeSpec.types';
interface UseNodeSpecsOptions {
    framework?: Framework;
    autoFetch?: boolean;
}
interface UseNodeSpecsReturn {
    specs: NodeSpec[];
    loading: boolean;
    error: string | null;
    refetch: () => Promise<void>;
    getSpec: (nodeType: string) => NodeSpec | undefined;
    renderCode: (nodeType: string, config: Record<string, any>) => Promise<string | null>;
}
/**
 * Hook to fetch and manage all node specifications for a framework
 */
export declare function useNodeSpecs(options?: UseNodeSpecsOptions): UseNodeSpecsReturn;
/**
 * Hook to fetch a single node specification
 */
export declare function useNodeSpec(nodeType: string, framework?: Framework): {
    spec: any;
    loading: boolean;
    error: string | null;
};
export {};
//# sourceMappingURL=useNodeSpecs.d.ts.map