import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium font-body transition-all duration-200 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default:
          "bg-signal text-background hover:bg-signal-soft shadow-[0_0_0_1px_rgba(76,224,210,0.3)] hover:shadow-[0_0_24px_rgba(76,224,210,0.35)]",
        secondary:
          "glass-panel text-ink hover:bg-glass-hover",
        outline:
          "border border-glass-border text-ink hover:bg-glass-hover",
        ghost: "text-ink-muted hover:text-ink hover:bg-glass-hover",
        amber:
          "bg-amber text-background hover:bg-amber-soft shadow-[0_0_0_1px_rgba(245,166,35,0.3)] hover:shadow-[0_0_24px_rgba(245,166,35,0.35)]",
        link: "text-signal underline-offset-4 hover:underline",
      },
      size: {
        default: "h-11 px-6 py-2",
        sm: "h-9 rounded-md px-4 text-xs",
        lg: "h-14 rounded-xl px-8 text-base",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
