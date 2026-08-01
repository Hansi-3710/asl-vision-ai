"use client";

import { useState } from "react";
import Webcam from "react-webcam";
import { motion } from "framer-motion";
import { CameraOff, RotateCcw } from "lucide-react";
import { Navbar } from "@/components/navbar";
import { Footer } from "@/components/footer";
import { ViewfinderFrame } from "@/components/viewfinder-frame";
import { BoundingBoxOverlay } from "@/components/camera/bounding-box-overlay";
import { LiveHud } from "@/components/camera/live-hud";
import { TopPredictionsPanel } from "@/components/camera/top-predictions-panel";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useWebcamPrediction } from "@/hooks/use-webcam-prediction";
import { formatConfidence, formatLatency } from "@/lib/utils";

const VIDEO_CONSTRAINTS = { width: 640, height: 480, facingMode: "user" };

export default function CameraPage() {
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const { webcamRef, prediction, isLoading, error, fps, isModelUnavailable } = useWebcamPrediction({
    intervalMs: 900,
    enabled: isCameraReady && !cameraError,
  });

  return (
    <main className="min-h-screen">
      <Navbar />

      <section className="mx-auto max-w-6xl px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-8"
        >
          <h1 className="font-display text-3xl font-semibold tracking-tight text-ink">Live Camera</h1>
          <p className="mt-2 text-ink-muted">
            Hold a hand sign steady in view of your camera. Predictions update automatically.
          </p>
        </motion.div>

        <div className="grid gap-6 lg:grid-cols-[1.3fr_0.7fr]">
          {/* Video feed */}
          <ViewfinderFrame size={28} className="rounded-2xl">
            <div className="glass-panel relative aspect-[4/3] w-full overflow-hidden rounded-2xl bg-background-elevated">
              {cameraError ? (
                <div className="flex h-full flex-col items-center justify-center gap-4 p-8 text-center">
                  <CameraOff className="h-10 w-10 text-ink-faint" />
                  <div>
                    <p className="font-medium text-ink">Camera access needed</p>
                    <p className="mt-1 max-w-xs text-sm text-ink-muted">{cameraError}</p>
                  </div>
                  <Button variant="secondary" size="sm" onClick={() => setCameraError(null)}>
                    <RotateCcw className="h-4 w-4" />
                    Try again
                  </Button>
                </div>
              ) : (
                <>
                  <Webcam
                    ref={webcamRef}
                    audio={false}
                    mirrored
                    screenshotFormat="image/jpeg"
                    videoConstraints={VIDEO_CONSTRAINTS}
                    onUserMedia={() => setIsCameraReady(true)}
                    onUserMediaError={() =>
                      setCameraError(
                        "We couldn't access your camera. Check your browser's permissions and make sure no other app is using it."
                      )
                    }
                    className="h-full w-full object-cover"
                  />
                  <BoundingBoxOverlay prediction={prediction} />
                  <LiveHud fps={fps} isLoading={isLoading} error={error} isModelUnavailable={isModelUnavailable} />
                </>
              )}
            </div>
          </ViewfinderFrame>

          {/* Side panel: top predictions + stats */}
          <div className="space-y-6">
            <TopPredictionsPanel prediction={prediction} />

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Latency</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="font-mono text-2xl font-semibold text-ink">
                  {prediction ? formatLatency(prediction.latency_ms) : "--"}
                </p>
                <p className="mt-1 text-xs text-ink-muted">Time from frame capture to prediction</p>
              </CardContent>
            </Card>

            {error && !isModelUnavailable && (
              <Card className="border-danger/30">
                <CardContent className="pt-6 text-sm text-danger">{error}</CardContent>
              </Card>
            )}

            {isModelUnavailable && (
              <Card className="border-amber/30">
                <CardContent className="pt-6 text-sm text-ink-muted">
                  The model hasn&apos;t been trained/deployed yet, so predictions aren&apos;t available.
                  See the project README&apos;s Training section to train one.
                </CardContent>
              </Card>
            )}
          </div>
        </div>

        {prediction && (
          <p className="mt-6 text-center text-xs text-ink-faint">
            Confidence: {formatConfidence(prediction.confidence)} · Source: {prediction.source}
          </p>
        )}
      </section>

      <Footer />
    </main>
  );
}
