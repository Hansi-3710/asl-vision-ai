"use client";

import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import { motion } from "framer-motion";
import { CameraOff, RotateCcw } from "lucide-react";
import { Navbar } from "@/components/navbar";
import { Footer } from "@/components/footer";
import { ViewfinderFrame } from "@/components/viewfinder-frame";
import { SkeletonOverlay } from "@/components/camera/skeleton-overlay";
import { StreamStatsHud } from "@/components/camera/stream-stats-hud";
import { TranslationPanel } from "@/components/camera/translation-panel";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { useHolisticLandmarks, type HolisticFrame } from "@/hooks/use-holistic-landmarks";
import { useContinuousRecognition } from "@/hooks/use-continuous-recognition";

const VIDEO_CONSTRAINTS = { width: 640, height: 480, facingMode: "user" };

export default function ContinuousCameraPage() {
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [latestFrame, setLatestFrame] = useState<HolisticFrame | null>(null);

  const webcamRef = useRef<Webcam>(null);
  const enabled = isCameraReady && !cameraError;

  const { sendFrame, ...recognition } = useContinuousRecognition({ enabled });

  const handleFrame = (frame: HolisticFrame) => {
    setLatestFrame(frame);
    sendFrame(frame.features);
  };

  const { attachVideo, isModelLoading, modelError } = useHolisticLandmarks({ enabled, onFrame: handleFrame });

  // react-webcam only exposes its underlying <video> element once the
  // stream is attached -- re-attach whenever camera readiness flips so
  // the holistic hook always has a live element to read frames from.
  useEffect(() => {
    if (isCameraReady) {
      attachVideo(webcamRef.current?.video ?? null);
    }
    return () => attachVideo(null);
  }, [isCameraReady, attachVideo]);

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
          <h1 className="font-display text-3xl font-semibold tracking-tight text-ink">Live Translate</h1>
          <p className="mt-2 text-ink-muted">
            Sign naturally in view of your camera -- full sentences are recognized continuously, no capture button
            needed.
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
                    videoConstraints={VIDEO_CONSTRAINTS}
                    onUserMedia={() => setIsCameraReady(true)}
                    onUserMediaError={() =>
                      setCameraError(
                        "We couldn't access your camera. Check your browser's permissions and make sure no other app is using it."
                      )
                    }
                    className="h-full w-full object-cover"
                  />
                  <SkeletonOverlay
                    poseLandmarks={latestFrame?.poseLandmarks ?? null}
                    leftHandLandmarks={latestFrame?.leftHandLandmarks ?? null}
                    rightHandLandmarks={latestFrame?.rightHandLandmarks ?? null}
                  />
                  <StreamStatsHud
                    status={recognition.status}
                    fps={recognition.fps}
                    latencyMs={recognition.latencyMs}
                    handsDetected={latestFrame?.handsDetected ?? false}
                    isSyntheticPlaceholder={recognition.readyInfo?.is_synthetic_placeholder ?? false}
                  />

                  {isModelLoading && (
                    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-full bg-glass px-4 py-1.5 text-xs text-ink-muted backdrop-blur-md">
                      Loading hand/pose tracking model...
                    </div>
                  )}
                </>
              )}
            </div>
          </ViewfinderFrame>

          {/* Side panel: live translation + stats */}
          <div className="space-y-6">
            <TranslationPanel
              transcript={recognition.transcript}
              words={recognition.words}
              history={recognition.history}
              onStartNewConversation={recognition.startNewConversation}
            />

            {(modelError || recognition.lastError) && (
              <Card className="border-danger/30">
                <CardContent className="pt-6 text-sm text-danger">{modelError || recognition.lastError}</CardContent>
              </Card>
            )}

            {recognition.readyInfo?.is_synthetic_placeholder && (
              <Card className="border-amber/30">
                <CardContent className="pt-6 text-sm text-ink-muted">
                  This model was trained on synthetic placeholder data to prove the pipeline works end to end -- it
                  isn&apos;t trained on real ASL yet. See the README&apos;s Continuous Recognition section for how to
                  train it on WLASL/How2Sign.
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}
