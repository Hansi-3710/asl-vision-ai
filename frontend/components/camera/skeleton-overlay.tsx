"use client";

import type { RawLandmark } from "@/lib/holistic-features";

interface SkeletonOverlayProps {
  poseLandmarks: RawLandmark[] | null;
  leftHandLandmarks: RawLandmark[] | null;
  rightHandLandmarks: RawLandmark[] | null;
}

// A relevant-to-signing subset of MediaPipe Pose's 33 landmarks -- upper
// body only (shoulders/elbows/wrists/hips/nose), since legs are never
// meaningful for ASL and drawing them would just add visual noise.
// Indices match MediaPipe Pose's standard landmark ordering.
const POSE_CONNECTIONS: [number, number][] = [
  [11, 12], // shoulder to shoulder
  [11, 13], [13, 15], // left arm
  [12, 14], [14, 16], // right arm
  [11, 23], [12, 24], // torso sides
  [23, 24], // hips
  [0, 11], [0, 12], // nose to shoulders (rough neck reference)
];
const POSE_POINT_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24];

// Standard 21-point MediaPipe Hand connection graph.
const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], // thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // index
  [5, 9], [9, 10], [10, 11], [11, 12], // middle
  [9, 13], [13, 14], [14, 15], [15, 16], // ring
  [13, 17], [17, 18], [18, 19], [19, 20], // pinky
  [0, 17], // wrist to pinky base
];

function Bones({
  landmarks,
  connections,
  color,
  pointIndices,
}: {
  landmarks: RawLandmark[] | null;
  connections: [number, number][];
  color: string;
  pointIndices?: number[];
}) {
  if (!landmarks || landmarks.length === 0) return null;
  const indices = pointIndices ?? landmarks.map((_, i) => i);

  return (
    <g>
      {connections.map(([a, b], i) => {
        const la = landmarks[a];
        const lb = landmarks[b];
        if (!la || !lb) return null;
        return (
          <line
            key={i}
            x1={la.x} y1={la.y}
            x2={lb.x} y2={lb.y}
            stroke={color}
            strokeWidth={0.006}
            strokeLinecap="round"
            opacity={0.85}
          />
        );
      })}
      {indices.map((i) => {
        const lm = landmarks[i];
        if (!lm) return null;
        return <circle key={i} cx={lm.x} cy={lm.y} r={0.008} fill={color} opacity={0.9} />;
      })}
    </g>
  );
}

/**
 * Renders the live pose + hand skeleton as an SVG overlay using
 * MediaPipe's normalized (0-1) landmark coordinates directly as the SVG
 * viewBox coordinates -- no pixel math needed, it scales with the video
 * element automatically. Mirrored via CSS transform (scaleX(-1)) to match
 * the underlying <Webcam mirrored /> element: MediaPipe always detects
 * against the RAW (unmirrored) video frame, so without this the skeleton
 * would appear on the wrong side of the mirrored video.
 */
export function SkeletonOverlay({ poseLandmarks, leftHandLandmarks, rightHandLandmarks }: SkeletonOverlayProps) {
  return (
    <svg
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
      className="pointer-events-none absolute inset-0 h-full w-full"
      style={{ transform: "scaleX(-1)" }}
    >
      <Bones landmarks={poseLandmarks} connections={POSE_CONNECTIONS} color="#4ce0d2" pointIndices={POSE_POINT_INDICES} />
      <Bones landmarks={leftHandLandmarks} connections={HAND_CONNECTIONS} color="#f5a623" />
      <Bones landmarks={rightHandLandmarks} connections={HAND_CONNECTIONS} color="#f5a623" />
    </svg>
  );
}
