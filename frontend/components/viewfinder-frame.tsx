import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

interface ViewfinderFrameProps {
  children: ReactNode;
  className?: string;
  bracketClassName?: string;
  /** Size of each corner bracket in pixels. */
  size?: number;
  /** Show brackets only on hover/focus rather than always. */
  revealOnHover?: boolean;
}

/**
 * The site's signature element: L-shaped corner brackets styled after a
 * camera/computer-vision viewfinder locking onto a detected object --
 * directly echoing the live bounding box drawn around a signer's hand in
 * the camera feature. Wrap any card, image, or section with this to tie it
 * back to the product's actual visual language instead of a generic border
 * or shadow.
 */
export function ViewfinderFrame({
  children,
  className,
  bracketClassName,
  size = 20,
  revealOnHover = false,
}: ViewfinderFrameProps) {
  const bracketBase = cn(
    "pointer-events-none absolute border-signal transition-opacity duration-300",
    revealOnHover && "opacity-0 group-hover:opacity-100",
    bracketClassName
  );

  return (
    <div className={cn("group relative", className)}>
      <span
        className={cn(bracketBase, "left-0 top-0 border-l-2 border-t-2 rounded-tl-md")}
        style={{ width: size, height: size }}
        aria-hidden="true"
      />
      <span
        className={cn(bracketBase, "right-0 top-0 border-r-2 border-t-2 rounded-tr-md")}
        style={{ width: size, height: size }}
        aria-hidden="true"
      />
      <span
        className={cn(bracketBase, "left-0 bottom-0 border-l-2 border-b-2 rounded-bl-md")}
        style={{ width: size, height: size }}
        aria-hidden="true"
      />
      <span
        className={cn(bracketBase, "right-0 bottom-0 border-r-2 border-b-2 rounded-br-md")}
        style={{ width: size, height: size }}
        aria-hidden="true"
      />
      {children}
    </div>
  );
}
