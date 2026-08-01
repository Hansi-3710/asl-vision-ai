"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { formatConfidence } from "@/lib/utils";
import type { Prediction } from "@/types/prediction";

interface TopPredictionsPanelProps {
  prediction: Prediction | null;
}

export function TopPredictionsPanel({ prediction }: TopPredictionsPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Top Predictions</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {!prediction && (
          <p className="text-sm text-ink-muted">Show a hand sign to the camera to see predictions.</p>
        )}
        {prediction?.top_k.map((entry, i) => (
          <motion.div
            key={entry.class}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.05 }}
            className="space-y-1.5"
          >
            <div className="flex items-center justify-between text-sm">
              <span className={`font-mono font-medium ${i === 0 ? "text-signal" : "text-ink-muted"}`}>
                {entry.class.toUpperCase()}
              </span>
              <span className="font-mono text-xs text-ink-muted">{formatConfidence(entry.confidence)}</span>
            </div>
            <Progress
              value={entry.confidence * 100}
              indicatorClassName={i === 0 ? "bg-signal" : "bg-ink-faint"}
            />
          </motion.div>
        ))}
      </CardContent>
    </Card>
  );
}
