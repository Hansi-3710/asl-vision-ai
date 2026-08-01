import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium font-mono transition-colors",
  {
    variants: {
      variant: {
        default: "border-signal/30 bg-signal/10 text-signal",
        amber: "border-amber/30 bg-amber/10 text-amber",
        neutral: "border-glass-border bg-glass text-ink-muted",
        danger: "border-danger/30 bg-danger/10 text-danger",
      },
    },
    defaultVariants: { variant: "default" },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
