"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { LetterCount } from "@/types/prediction";

interface LetterChartProps {
  data: LetterCount[];
}

export function LetterChart({ data }: LetterChartProps) {
  const chartData = data.map((d) => ({ letter: d.letter.toUpperCase(), count: d.count }));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Most Predicted Letters</CardTitle>
      </CardHeader>
      <CardContent>
        {chartData.length === 0 ? (
          <p className="py-12 text-center text-sm text-ink-muted">No predictions yet.</p>
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={chartData} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
              <XAxis
                dataKey="letter"
                tick={{ fill: "#8B93A8", fontSize: 12, fontFamily: "var(--font-jetbrains-mono)" }}
                axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: "#8B93A8", fontSize: 12 }}
                axisLine={false}
                tickLine={false}
                allowDecimals={false}
              />
              <Tooltip
                cursor={{ fill: "rgba(76,224,210,0.06)" }}
                contentStyle={{
                  background: "#0F1420",
                  border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: 12,
                  color: "#F3F5F8",
                  fontSize: 13,
                }}
              />
              <Bar dataKey="count" fill="#4CE0D2" radius={[6, 6, 0, 0]} maxBarSize={36} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
