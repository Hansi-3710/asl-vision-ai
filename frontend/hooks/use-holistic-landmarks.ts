"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { flattenHolisticFrame, hasAnyHandDetection, type RawLandmark } from "@/lib/holistic-features";

// Per the source design doc's "Render-Friendly Backend" section: run
// MediaPipe entirely in the browser (WASM) and send only landmark
// packets to the backend, never raw video frames. The .task model asset
// and WASM binaries load from jsDelivr's CDN mirror of the npm package --
// the standard way Google's own MediaPipe Tasks examples serve these
// (see the module docstring in continuous_pipeline/landmark_extraction.py
// for the equivalent server-side/Python setup, which needs a locally
// downloaded copy of the same model instead).
const WASM_BASE_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const HOLISTIC_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task";

/** The result shape actually used from HolisticLandmarkerResult, kept
 * loose/defensive (`unknown` narrowed at the call site) rather than
 * importing the SDK's own result type -- MediaPipe's JS Tasks results
 * have historically varied between a flat landmark array and a
 * one-detection-per-person nested array across task types; normalizing
 * defensively here means a version bump can't silently break frame
 * parsing without at least still returning *something* sane. */
function normalizeLandmarkList(raw: unknown): RawLandmark[] | null {
  if (!raw) return null;
  const arr = raw as unknown[];
  if (arr.length === 0) return null;
  // Nested (one array of landmarks per detected instance) -> take the first.
  if (Array.isArray(arr[0])) return (arr[0] as RawLandmark[]) ?? null;
  return arr as RawLandmark[];
}

export interface HolisticFrame {
  features: Float32Array;
  poseLandmarks: RawLandmark[] | null;
  leftHandLandmarks: RawLandmark[] | null;
  rightHandLandmarks: RawLandmark[] | null;
  handsDetected: boolean;
  timestampMs: number;
}

interface UseHolisticLandmarksOptions {
  enabled: boolean;
  onFrame: (frame: HolisticFrame) => void;
}

interface UseHolisticLandmarksResult {
  attachVideo: (video: HTMLVideoElement | null) => void;
  isModelLoading: boolean;
  modelError: string | null;
}

/**
 * Loads HolisticLandmarker once, then runs a requestAnimationFrame loop
 * calling detectForVideo() against whatever <video> element was attached
 * via `attachVideo` (react-webcam exposes its underlying <video> as
 * `webcamRef.current.video`). Calls `onFrame` with the flattened feature
 * vector plus raw landmarks (for skeleton drawing) on every successful
 * detection -- deliberately does NOT throttle itself to the backend's
 * target send rate; that's use-continuous-recognition's job, keeping
 * this hook's only responsibility "run MediaPipe, report what it sees".
 */
export function useHolisticLandmarks({
  enabled,
  onFrame,
}: UseHolisticLandmarksOptions): UseHolisticLandmarksResult {
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [modelError, setModelError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const landmarkerRef = useRef<any>(null);
  const rafRef = useRef<number | null>(null);
  const onFrameRef = useRef(onFrame);
  onFrameRef.current = onFrame;

  const attachVideo = useCallback((video: HTMLVideoElement | null) => {
    videoRef.current = video;
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        setIsModelLoading(true);
        setModelError(null);

        // Dynamic import: @mediapipe/tasks-vision touches `document`/WASM
        // at module init in some builds, which breaks Next.js's server-side
        // render pass if imported statically at the top of a "use client"
        // file that still gets evaluated during SSR/build.
        const { FilesetResolver, HolisticLandmarker } = await import("@mediapipe/tasks-vision");
        const vision = await FilesetResolver.forVisionTasks(WASM_BASE_URL);
        if (cancelled) return;

        const landmarker = await HolisticLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: HOLISTIC_MODEL_URL, delegate: "GPU" },
          runningMode: "VIDEO",
        });
        if (cancelled) {
          landmarker.close();
          return;
        }

        landmarkerRef.current = landmarker;
        setIsModelLoading(false);
      } catch (err) {
        if (cancelled) return;
        console.error("Failed to load MediaPipe HolisticLandmarker:", err);
        setModelError(
          "Couldn't load the hand/pose tracking model. Check your internet connection (it loads from a CDN) and try refreshing."
        );
        setIsModelLoading(false);
      }
    }

    init();
    return () => {
      cancelled = true;
      if (landmarkerRef.current) {
        landmarkerRef.current.close();
        landmarkerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!enabled) return;

    function tick() {
      const landmarker = landmarkerRef.current;
      const video = videoRef.current;

      if (landmarker && video && video.readyState >= 2 && video.videoWidth > 0) {
        const timestampMs = performance.now();
        try {
          const result = landmarker.detectForVideo(video, timestampMs);
          const poseLandmarks = normalizeLandmarkList(result?.poseLandmarks);
          const leftHandLandmarks = normalizeLandmarkList(result?.leftHandLandmarks);
          const rightHandLandmarks = normalizeLandmarkList(result?.rightHandLandmarks);

          const features = flattenHolisticFrame(poseLandmarks, leftHandLandmarks, rightHandLandmarks);
          onFrameRef.current({
            features,
            poseLandmarks,
            leftHandLandmarks,
            rightHandLandmarks,
            handsDetected: hasAnyHandDetection(leftHandLandmarks, rightHandLandmarks),
            timestampMs,
          });
        } catch (err) {
          // A single bad frame (e.g. mid-resize) shouldn't kill the loop --
          // log and keep going, matching the backend's per-frame error
          // tolerance in api/stream.py.
          console.warn("HolisticLandmarker.detectForVideo failed for one frame:", err);
        }
      }

      rafRef.current = requestAnimationFrame(tick);
    }

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [enabled]);

  return { attachVideo, isModelLoading, modelError };
}
