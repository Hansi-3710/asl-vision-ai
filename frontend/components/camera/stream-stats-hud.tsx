"use client";

import { Activity, AlertTriangle, FlaskConical, Hand, Wifi, WifiOff } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { ConnectionStatus } from "@/hooks/use-continuous-recognition";
import { formatLatency } from "@/lib/utils";

interface StreamStatsHudProps {
  status: ConnectionStatus;
  fps: number;
  latencyMs: number;
  handsDetected: boolean;
  isSyntheticPlaceholder: boolean;
}

export function StreamStatsHud({ status, fps, latencyMs, handsDetected, isSyntheticPlaceholder }: StreamStatsHudProps) {
  const isLive = status === "open";

  return (
    <div className="absolute right-4 top-4 flex flex-col items-end gap-2">
      <Badge variant={isLive ? "default" : "danger"}>
        {isLive ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
        {isLive ? "Live" : status === "connecting" ? "Connecting..." : "Offline"}
      </Badge>

      {isLive && (
        <>
          <Badge variant="neutral">
            <Activity className="h-3 w-3" />
            {fps.toFixed(1)} FPS · {formatLatency(latencyMs)}
          </Badge>
          <Badge variant={handsDetected ? "default" : "neutral"}>
            <Hand className="h-3 w-3" />
            {handsDetected ? "Hands detected" : "No hands visible"}
          </Badge>
        </>
      )}

      {isSyntheticPlaceholder && (
        <Badge variant="amber" className="max-w-[240px] text-right">
          <FlaskConical className="h-3 w-3 shrink-0" />
          Demo model -- not trained on real ASL yet
        </Badge>
      )}
    </div>
  );
}
