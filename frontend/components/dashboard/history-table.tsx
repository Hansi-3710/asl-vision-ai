"use client";

import { Camera, Upload } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { formatConfidence, formatLatency } from "@/lib/utils";
import type { Prediction } from "@/types/prediction";

interface HistoryTableProps {
  predictions: Prediction[];
}

function formatTimestamp(iso: string | null): string {
  if (!iso) return "--";
  const date = new Date(iso);
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function HistoryTable({ predictions }: HistoryTableProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Recent Predictions</CardTitle>
      </CardHeader>
      <CardContent>
        {predictions.length === 0 ? (
          <p className="py-12 text-center text-sm text-ink-muted">No predictions yet -- try the camera or upload a photo.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-glass-border text-left text-xs uppercase tracking-wide text-ink-faint">
                  <th className="pb-3 pr-4 font-medium">Letter</th>
                  <th className="pb-3 pr-4 font-medium">Confidence</th>
                  <th className="pb-3 pr-4 font-medium">Source</th>
                  <th className="pb-3 pr-4 font-medium">Latency</th>
                  <th className="pb-3 font-medium">Time</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((p) => (
                  <tr key={p.id} className="border-b border-glass-border/60 last:border-0">
                    <td className="py-3 pr-4">
                      <span className="font-mono font-semibold text-signal">{p.predicted_class.toUpperCase()}</span>
                    </td>
                    <td className="py-3 pr-4 font-mono text-ink-muted">{formatConfidence(p.confidence)}</td>
                    <td className="py-3 pr-4">
                      <Badge variant="neutral" className="normal-case">
                        {p.source === "webcam" ? <Camera className="h-3 w-3" /> : <Upload className="h-3 w-3" />}
                        {p.source}
                      </Badge>
                    </td>
                    <td className="py-3 pr-4 font-mono text-ink-muted">{formatLatency(p.latency_ms)}</td>
                    <td className="py-3 text-ink-muted">{formatTimestamp(p.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
