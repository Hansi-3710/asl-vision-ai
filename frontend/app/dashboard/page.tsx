"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { RefreshCw } from "lucide-react";
import { Navbar } from "@/components/navbar";
import { Footer } from "@/components/footer";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { StatsCards } from "@/components/dashboard/stats-cards";
import { LetterChart } from "@/components/dashboard/letter-chart";
import { HistoryTable } from "@/components/dashboard/history-table";
import { getHistory, getMetrics, describeApiError } from "@/lib/api";
import type { Metrics, Prediction } from "@/types/prediction";

export default function DashboardPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [history, setHistory] = useState<Prediction[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [metricsData, historyData] = await Promise.all([getMetrics(), getHistory({ limit: 20 })]);
      setMetrics(metricsData);
      setHistory(historyData);
    } catch (err) {
      setError(describeApiError(err));
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <main className="min-h-screen">
      <Navbar />

      <section className="mx-auto max-w-6xl px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-8 flex items-center justify-between"
        >
          <div>
            <h1 className="font-display text-3xl font-semibold tracking-tight text-ink">Dashboard</h1>
            <p className="mt-2 text-ink-muted">A live look at every prediction the model has served.</p>
          </div>
          <Button variant="secondary" size="sm" onClick={load} disabled={isLoading}>
            <RefreshCw className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </motion.div>

        {error && (
          <div className="mb-6 rounded-xl border border-danger/30 bg-danger/5 p-4 text-sm text-danger">{error}</div>
        )}

        {isLoading && !metrics ? (
          <div className="space-y-6">
            <div className="grid gap-6 sm:grid-cols-3">
              <Skeleton className="h-24 rounded-2xl" />
              <Skeleton className="h-24 rounded-2xl" />
              <Skeleton className="h-24 rounded-2xl" />
            </div>
            <Skeleton className="h-80 rounded-2xl" />
            <Skeleton className="h-96 rounded-2xl" />
          </div>
        ) : (
          metrics && (
            <div className="space-y-6">
              <StatsCards metrics={metrics} />
              <LetterChart data={metrics.most_predicted_letters} />
              <HistoryTable predictions={history} />
            </div>
          )
        )}
      </section>

      <Footer />
    </main>
  );
}
