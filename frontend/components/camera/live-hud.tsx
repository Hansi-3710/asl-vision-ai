"use client";

import { Activity, AlertTriangle, Wifi } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface LiveHudProps {
  fps: number;
  isLoading: boolean;
  error: string | null;
  isModelUnavailable: boolean;
}

export function LiveHud({ fps, isLoading, error, isModelUnavailable }: LiveHudProps) {
  return (
    <div className="absolute right-4 top-4 flex flex-col items-end gap-2">
      <Badge variant={error ? "danger" : "default"}>
        {error ? <AlertTriangle className="h-3 w-3" /> : <Wifi className="h-3 w-3" />}
        {error ? "Offline" : "Live"}
      </Badge>
      <Badge variant="neutral">
        <Activity className={`h-3 w-3 ${isLoading ? "animate-pulse" : ""}`} />
        {fps.toFixed(1)} FPS
      </Badge>
      {isModelUnavailable && (
        <Badge variant="amber" className="max-w-[220px] text-right">
          Model not trained yet
        </Badge>
      )}
    </div>
  );
}
