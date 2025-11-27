import { ComponentProps } from "react";
import * as LabelPrimitive from "@radix-ui/react-label";
import { Slot } from "@radix-ui/react-slot";
import { type ControllerProps, type FieldPath, type FieldValues } from "react-hook-form";
declare const Form: <TFieldValues extends FieldValues, TContext = any, TTransformedValues = TFieldValues>(props: import("react-hook-form").FormProviderProps<TFieldValues, TContext, TTransformedValues>) => React.JSX.Element;
declare const FormField: <TFieldValues extends FieldValues = FieldValues, TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>>({ ...props }: ControllerProps<TFieldValues, TName>) => import("react/jsx-runtime").JSX.Element;
declare const useFormField: () => {
    invalid: boolean;
    isDirty: boolean;
    isTouched: boolean;
    isValidating: boolean;
    error?: import("react-hook-form").FieldError;
    id: string;
    name: string;
    formItemId: string;
    formDescriptionId: string;
    formMessageId: string;
};
declare function FormItem({ className, ...props }: ComponentProps<"div">): import("react/jsx-runtime").JSX.Element;
declare function FormLabel({ className, ...props }: ComponentProps<typeof LabelPrimitive.Root>): import("react/jsx-runtime").JSX.Element;
declare function FormControl({ ...props }: ComponentProps<typeof Slot>): import("react/jsx-runtime").JSX.Element;
declare function FormDescription({ className, ...props }: ComponentProps<"p">): import("react/jsx-runtime").JSX.Element;
declare function FormMessage({ className, ...props }: ComponentProps<"p">): import("react/jsx-runtime").JSX.Element | null;
export { useFormField, Form, FormItem, FormLabel, FormControl, FormDescription, FormMessage, FormField, };
//# sourceMappingURL=form.d.ts.map