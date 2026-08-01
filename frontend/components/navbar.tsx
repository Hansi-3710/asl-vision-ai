"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Camera, LayoutDashboard, ScanLine, Upload } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

const NAV_LINKS = [
  { href: "/camera", label: "Live Camera", icon: Camera },
  { href: "/upload", label: "Upload", icon: Upload },
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 w-full">
      <div className="mx-auto max-w-6xl px-6 pt-4">
        <nav className="glass-panel flex items-center justify-between rounded-2xl px-4 py-3">
          <Link href="/" className="flex items-center gap-2 group">
            <span className="relative flex h-8 w-8 items-center justify-center rounded-lg bg-signal/10 border border-signal/30">
              <ScanLine className="h-4 w-4 text-signal" strokeWidth={2.5} />
            </span>
            <span className="font-display text-lg font-semibold tracking-tight text-ink">
              ASL Vision <span className="text-signal">AI</span>
            </span>
          </Link>

          <div className="hidden items-center gap-1 md:flex">
            {NAV_LINKS.map(({ href, label, icon: Icon }) => {
              const active = pathname === href;
              return (
                <Link
                  key={href}
                  href={href}
                  className={cn(
                    "flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors",
                    active ? "bg-signal/10 text-signal" : "text-ink-muted hover:text-ink hover:bg-glass-hover"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {label}
                </Link>
              );
            })}
          </div>

          <Button asChild size="sm">
            <Link href="/camera">Start Camera</Link>
          </Button>
        </nav>
      </div>
    </header>
  );
}
