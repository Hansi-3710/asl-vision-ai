"use client";

import * as React from "react";
import { Toaster as Sonner } from "sonner";

type ToasterProps = React.ComponentProps<typeof Sonner>;

export function Toaster({ ...props }: ToasterProps) {
  return (
    <Sonner
      theme="dark"
      className="toaster group"
      toastOptions={{
        classNames: {
          toast:
            "group toast glass-panel !rounded-xl !shadow-2xl !shadow-black/30 group-[.toaster]:text-ink",
          description: "group-[.toast]:text-ink-muted",
          actionButton: "group-[.toast]:bg-signal group-[.toast]:text-background",
          cancelButton: "group-[.toast]:bg-glass group-[.toast]:text-ink-muted",
          success: "group-[.toast]:!border-signal/30",
          error: "group-[.toast]:!border-danger/30",
        },
      }}
      {...props}
    />
  );
}
