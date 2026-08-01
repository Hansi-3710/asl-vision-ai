"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Camera, Upload, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ViewfinderFrame } from "@/components/viewfinder-frame";

const LETTERS = ["A", "S", "L"];

export function Hero() {
  return (
    <section className="relative overflow-hidden px-6 pb-24 pt-20 md:pt-28">
      <div className="mx-auto grid max-w-6xl items-center gap-16 lg:grid-cols-[1.1fr_0.9fr]">
        {/* Left: thesis statement */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
        >
          <Badge className="mb-6">
            <Zap className="h-3 w-3" />
            Real-time inference, under 50ms
          </Badge>

          <h1 className="font-display text-5xl font-semibold leading-[1.05] tracking-tight text-ink md:text-6xl">
            Sign a letter.
            <br />
            Watch it get <span className="text-gradient-signal">read</span>.
          </h1>

          <p className="mt-6 max-w-lg text-lg leading-relaxed text-ink-muted">
            ASL Vision AI recognizes American Sign Language alphabet letters the instant you form
            them -- live through your webcam, or from a single uploaded photo. No setup, no
            waiting.
          </p>

          <div className="mt-10 flex flex-col gap-4 sm:flex-row">
            <Button asChild size="lg">
              <Link href="/camera">
                <Camera className="h-5 w-5" />
                Start Camera
              </Link>
            </Button>
            <Button asChild size="lg" variant="secondary">
              <Link href="/upload">
                <Upload className="h-5 w-5" />
                Upload Image
              </Link>
            </Button>
          </div>

          <div className="mt-12 flex items-center gap-6 text-sm text-ink-faint">
            <span className="font-mono">29 classes</span>
            <span className="h-1 w-1 rounded-full bg-ink-faint" />
            <span className="font-mono">EfficientNetV2</span>
            <span className="h-1 w-1 rounded-full bg-ink-faint" />
            <span className="font-mono">On-device webcam, nothing uploaded unless you choose to</span>
          </div>
        </motion.div>

        {/* Right: animated viewfinder demo */}
        <motion.div
          initial={{ opacity: 0, scale: 0.94 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
          className="relative mx-auto w-full max-w-sm"
        >
          <ViewfinderFrame size={28} className="rounded-2xl">
            <div className="glass-panel relative aspect-[3/4] overflow-hidden rounded-2xl">
              {/* Ambient scanning line */}
              <div className="pointer-events-none absolute inset-x-6 top-0 h-32 overflow-hidden opacity-60">
                <div className="h-px w-full animate-scan-line bg-gradient-to-r from-transparent via-signal to-transparent" />
              </div>

              <div className="flex h-full flex-col items-center justify-center gap-6 p-8">
                <div className="flex gap-4">
                  {LETTERS.map((letter, i) => (
                    <motion.div
                      key={letter}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.6 + i * 0.15, duration: 0.5 }}
                      className="flex h-16 w-16 items-center justify-center rounded-xl border border-signal/30 bg-signal/5 font-display text-2xl font-semibold text-signal"
                    >
                      {letter}
                    </motion.div>
                  ))}
                </div>

                <div className="w-full space-y-2">
                  <div className="flex items-center justify-between text-xs font-mono text-ink-muted">
                    <span>CONFIDENCE</span>
                    <span className="text-signal">98.4%</span>
                  </div>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-glass">
                    <motion.div
                      initial={{ width: "0%" }}
                      animate={{ width: "98.4%" }}
                      transition={{ delay: 1.1, duration: 0.8, ease: "easeOut" }}
                      className="h-full rounded-full bg-signal"
                    />
                  </div>
                </div>

                <p className="text-center text-xs text-ink-faint">
                  Live prediction preview -- try it with your own hand
                </p>
              </div>
            </div>
          </ViewfinderFrame>

          {/* Ambient glow behind the card */}
          <div
            className="absolute inset-0 -z-10 rounded-2xl bg-signal/20 blur-3xl"
            aria-hidden="true"
          />
        </motion.div>
      </div>
    </section>
  );
}
