import { ComponentProps } from "react";
import * as LabelPrimitive from "@radix-ui/react-label";
import { Slot } from "@radix-ui/react-slot";
import { type ControllerProps, type FieldPath, type FieldValues } from "react-hook-form";
declare const Form: any;
declare const FormField: <TFieldValues extends FieldValues = FieldValues, TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>>({ ...props }: ControllerProps<TFieldValues, TName>) => any;
declare const useFormField: () => any;
declare function FormItem({ className, ...props }: ComponentProps<"div">): any;
declare function FormLabel({ className, ...props }: ComponentProps<typeof LabelPrimitive.Root>): any;
declare function FormControl({ ...props }: ComponentProps<typeof Slot>): any;
declare function FormDescription({ className, ...props }: ComponentProps<"p">): any;
declare function FormMessage({ className, ...props }: ComponentProps<"p">): any;
export { useFormField, Form, FormItem, FormLabel, FormControl, FormDescription, FormMessage, FormField, };
//# sourceMappingURL=form.d.ts.map