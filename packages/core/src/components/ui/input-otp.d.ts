import { ComponentProps } from "react";
import { OTPInput } from "input-otp";
declare function InputOTP({ className, containerClassName, ...props }: ComponentProps<typeof OTPInput> & {
    containerClassName?: string;
}): any;
declare function InputOTPGroup({ className, ...props }: ComponentProps<"div">): any;
declare function InputOTPSlot({ index, className, ...props }: ComponentProps<"div"> & {
    index: number;
}): any;
declare function InputOTPSeparator({ ...props }: ComponentProps<"div">): any;
export { InputOTP, InputOTPGroup, InputOTPSlot, InputOTPSeparator };
//# sourceMappingURL=input-otp.d.ts.map