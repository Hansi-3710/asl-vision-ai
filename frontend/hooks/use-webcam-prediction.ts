"use client";

import * as React from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import type Webcam from "react-webcam";
import { predictWebcamFrame, describeApiError } from "@/lib/api";
import type { Prediction } from "@/types/prediction";

interface UseWebcamPredictionOptions {
  /** Milliseconds between prediction requests. Throttled deliberately --
   * polling on every animation frame would flood the backend and the
   * database with near-duplicate predictions for no benefit. */
  intervalMs?: number;
  enabled?: boolean;
}

interface UseWebcamPredictionResult {
  webcamRef: React.RefObject<Webcam>;
  prediction: Prediction | null;
  isLoading: boolean;
  error: string | null;
  fps: number;
  isModelUnavailable: boolean;
}

/**
 * Captures a frame from the given webcam ref on a fixed interval, sends it
 * to /predict-webcam, and exposes the latest result. Deliberately does its
 * OWN capture (via Webcam.getScreenshot()) rather than requiring the
 * caller to pass frames in, so any component using this hook just needs
 * to render <Webcam ref={webcamRef} />.
 */
export function useWebcamPrediction({
  intervalMs = 900,
  enabled = true,
}: UseWebcamPredictionOptions = {}): UseWebcamPredictionResult {
  const webcamRef = useRef<Webcam>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isModelUnavailable, setIsModelUnavailable] = useState(false);
  const [fps, setFps] = useState(0);

  const inFlightRef = useRef(false);
  const lastTickRef = useRef<number>(performance.now());

  const captureAndPredict = useCallback(async () => {
    if (inFlightRef.current || !enabled) return;
    const screenshot = webcamRef.current?.getScreenshot();
    if (!screenshot) return;

    inFlightRef.current = true;
    setIsLoading(true);

    const now = performance.now();
    const elapsedSec = (now - lastTickRef.current) / 1000;
    lastTickRef.current = now;
    if (elapsedSec > 0) setFps(1 / elapsedSec);

    try {
      const result = await predictWebcamFrame(screenshot);
      setPrediction(result);
      setError(null);
      setIsModelUnavailable(false);
    } catch (err) {
      const message = describeApiError(err);
      setError(message);
      // Distinguish "model not trained yet" so the UI can show setup
      // guidance instead of a generic error banner every second.
      setIsModelUnavailable(message.toLowerCase().includes("model isn't loaded"));
    } finally {
      inFlightRef.current = false;
      setIsLoading(false);
    }
  }, [enabled]);

  useEffect(() => {
    if (!enabled) return;
    const id = setInterval(captureAndPredict, intervalMs);
    return () => clearInterval(id);
  }, [captureAndPredict, enabled, intervalMs]);

  return { webcamRef, prediction, isLoading, error, fps, isModelUnavailable };
}
