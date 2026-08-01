import Link from "next/link";
import { ScanLine, Github } from "lucide-react";

export function Footer() {
  return (
    <footer className="mt-32 border-t border-glass-border">
      <div className="mx-auto max-w-6xl px-6 py-12">
        <div className="flex flex-col items-start justify-between gap-8 md:flex-row md:items-center">
          <div className="flex items-center gap-2">
            <ScanLine className="h-4 w-4 text-signal" />
            <span className="font-display text-sm font-semibold text-ink">ASL Vision AI</span>
          </div>

          <p className="max-w-md text-sm text-ink-muted">
            A research and engineering demo -- recognizes static ASL alphabet letters, not full
            ASL (which includes grammar, motion, and non-manual markers). Not a substitute for an
            interpreter.
          </p>

          <Link
            href="https://github.com"
            className="flex items-center gap-2 text-sm text-ink-muted transition-colors hover:text-ink"
          >
            <Github className="h-4 w-4" />
            Source
          </Link>
        </div>

        <div className="mt-8 border-t border-glass-border pt-6 text-xs text-ink-faint">
          © {new Date().getFullYear()} ASL Vision AI. Built with Next.js, FastAPI, and PyTorch.
        </div>
      </div>
    </footer>
  );
}
