"use client";

import { useCallback, useRef, useState } from "react";
import { motion } from "framer-motion";
import { ImagePlus, UploadCloud } from "lucide-react";
import { cn } from "@/lib/utils";

interface DropzoneProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];

export function Dropzone({ onFileSelected, disabled }: DropzoneProps) {
  const [isDragActive, setIsDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const file = files[0];
      if (!ACCEPTED_TYPES.includes(file.type)) return;
      onFileSelected(file);
    },
    [onFileSelected]
  );

  return (
    <motion.div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setIsDragActive(true);
      }}
      onDragLeave={() => setIsDragActive(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragActive(false);
        if (!disabled) handleFiles(e.dataTransfer.files);
      }}
      onClick={() => !disabled && inputRef.current?.click()}
      whileHover={disabled ? undefined : { scale: 1.01 }}
      className={cn(
        "glass-panel flex aspect-[4/3] w-full cursor-pointer flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed transition-colors",
        isDragActive ? "border-signal bg-signal/5" : "border-glass-border",
        disabled && "cursor-not-allowed opacity-60"
      )}
      role="button"
      tabIndex={0}
      aria-label="Upload an image"
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(",")}
        className="hidden"
        disabled={disabled}
        onChange={(e) => handleFiles(e.target.files)}
      />

      <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-signal/10 border border-signal/20">
        {isDragActive ? (
          <ImagePlus className="h-7 w-7 text-signal" />
        ) : (
          <UploadCloud className="h-7 w-7 text-signal" />
        )}
      </div>

      <div className="text-center">
        <p className="font-medium text-ink">
          {isDragActive ? "Drop your image here" : "Drag & drop an image, or click to browse"}
        </p>
        <p className="mt-1 text-sm text-ink-muted">JPEG, PNG, or WebP</p>
      </div>
    </motion.div>
  );
}
