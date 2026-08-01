"use client";

import { motion } from "framer-motion";
import { Clock, Sparkles } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ViewfinderFrame } from "@/components/viewfinder-frame";
import { formatConfidence, formatLatency } from "@/lib/utils";
import type { Prediction } from "@/types/prediction";

interface UploadResultProps {
  imagePreviewUrl: string;
  prediction: Prediction;
}

export function UploadResult({ imagePreviewUrl, prediction }: UploadResultProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="grid gap-6 md:grid-cols-2"
    >
      <ViewfinderFrame size={24} className="rounded-2xl">
        {/* eslint-disable-next-line @next/next/no-img-element -- data/blob URL preview, not an optimizable remote asset */}
        <img
          src={imagePreviewUrl}
          alt="Uploaded hand sign"
          className="aspect-square w-full rounded-2xl object-cover"
        />
      </ViewfinderFrame>

      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-signal" />
              Prediction
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-baseline gap-3">
              <span className="font-display text-5xl font-bold text-gradient-signal">
                {prediction.predicted_class.toUpperCase()}
              </span>
              <span className="font-mono text-lg text-ink-muted">
                {formatConfidence(prediction.confidence)}
              </span>
            </div>
            {prediction.hand_detected === false && (
              <p className="mt-3 text-sm text-amber">
                No hand was confidently detected -- this is a best guess on the full image.
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Top 5 Predictions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {prediction.top_k.map((entry, i) => (
              <div key={entry.class} className="space-y-1.5">
                <div className="flex items-center justify-between text-sm">
                  <span className={`font-mono font-medium ${i === 0 ? "text-signal" : "text-ink-muted"}`}>
                    {entry.class.toUpperCase()}
                  </span>
                  <span className="font-mono text-xs text-ink-muted">{formatConfidence(entry.confidence)}</span>
                </div>
                <Progress value={entry.confidence * 100} indicatorClassName={i === 0 ? "bg-signal" : "bg-ink-faint"} />
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex items-center gap-3 pt-6">
            <Clock className="h-4 w-4 text-ink-muted" />
            <span className="text-sm text-ink-muted">Inference time</span>
            <span className="ml-auto font-mono text-sm text-ink">{formatLatency(prediction.latency_ms)}</span>
          </CardContent>
        </Card>
      </div>
    </motion.div>
  );
}
