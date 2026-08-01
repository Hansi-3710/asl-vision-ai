"use client";

import { motion } from "framer-motion";
import { Gauge, Hash, Timer } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { formatLatency } from "@/lib/utils";
import type { Metrics } from "@/types/prediction";

interface StatsCardsProps {
  metrics: Metrics;
}

export function StatsCards({ metrics }: StatsCardsProps) {
  const stats = [
    {
      icon: Hash,
      label: "Total Predictions",
      value: metrics.total_predictions.toLocaleString(),
    },
    {
      icon: Gauge,
      label: "Average Confidence",
      value: `${(metrics.average_confidence * 100).toFixed(1)}%`,
    },
    {
      icon: Timer,
      label: "Average Latency",
      value: formatLatency(metrics.average_latency_ms),
    },
  ];

  return (
    <div className="grid gap-6 sm:grid-cols-3">
      {stats.map((stat, i) => (
        <motion.div
          key={stat.label}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.08, duration: 0.4 }}
        >
          <Card>
            <CardContent className="flex items-center gap-4 pt-6">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-signal/10 border border-signal/20">
                <stat.icon className="h-5 w-5 text-signal" />
              </div>
              <div>
                <p className="font-mono text-2xl font-semibold text-ink">{stat.value}</p>
                <p className="text-sm text-ink-muted">{stat.label}</p>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      ))}
    </div>
  );
}
