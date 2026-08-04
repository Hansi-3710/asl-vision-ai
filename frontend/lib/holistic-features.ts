/**
 * lib/holistic-features.ts
 * =========================
 * The browser-side twin of backend/continuous_pipeline/landmark_spec.py.
 * Flattens a MediaPipe HolisticLandmarker result into the EXACT SAME
 * 258-value feature vector layout the backend model was trained on and
 * expects over the WebSocket. THE TWO FILES MUST BE KEPT IN SYNC BY HAND
 * -- there is no shared source of truth across the Python/TypeScript
 * boundary, which is exactly why app/ml/landmarks.py validates every
 * incoming frame's length server-side and errors loudly (rather than
 * silently misbehaving) if it doesn't match FEATURE_DIM.
 *
 * Layout (258-dim float32 vector per frame):
 *
 *     [0    : 132]  pose:       33 landmarks x (x, y, z, visibility)
 *     [132  : 195]  left_hand:  21 landmarks x (x, y, z)
 *     [195  : 258]  right_hand: 21 landmarks x (x, y, z)
 *
 * Face landmarks are deliberately excluded (see landmark_spec.py's
 * docstring for why). Missing landmarks (a hand out of frame) are
 * encoded as zeros for that block, never omitted -- frame alignment
 * depends on every frame producing exactly FEATURE_DIM values.
 */

export const NUM_POSE_LANDMARKS = 33;
export const NUM_HAND_LANDMARKS = 21;

const POSE_VALUES_PER_LANDMARK = 4; // x, y, z, visibility
const HAND_VALUES_PER_LANDMARK = 3; // x, y, z

export const POSE_DIM = NUM_POSE_LANDMARKS * POSE_VALUES_PER_LANDMARK; // 132
export const HAND_DIM = NUM_HAND_LANDMARKS * HAND_VALUES_PER_LANDMARK; // 63
export const FEATURE_DIM = POSE_DIM + 2 * HAND_DIM; // 258

/** Minimal shape MediaPipe Tasks Vision landmark objects satisfy --
 * declared locally (rather than importing @mediapipe/tasks-vision's own
 * type here) so this module's core flatten logic is testable with plain
 * mock objects, no MediaPipe runtime needed, mirroring how
 * continuous_pipeline/landmark_extraction.py's `_flatten_holistic_result`
 * is tested against mock landmarks on the Python side. */
export interface RawLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

function flattenPart(
  landmarks: RawLandmark[] | null | undefined,
  valuesPerLandmark: number,
  expectedCount: number
): Float32Array {
  const flat = new Float32Array(expectedCount * valuesPerLandmark);
  if (!landmarks || landmarks.length === 0) return flat;

  const count = Math.min(landmarks.length, expectedCount);
  for (let i = 0; i < count; i++) {
    const lm = landmarks[i];
    const base = i * valuesPerLandmark;
    flat[base] = lm.x;
    flat[base + 1] = lm.y;
    flat[base + 2] = lm.z;
    if (valuesPerLandmark === 4) {
      flat[base + 3] = lm.visibility ?? 0;
    }
  }
  return flat;
}

/** Pure function: three landmark arrays (any may be missing/empty,
 * meaning that body part wasn't detected this frame) -> the
 * FEATURE_DIM-length feature vector. Mirrors
 * continuous_pipeline/landmark_extraction.py's `_flatten_holistic_result`
 * exactly -- same slice order, same zero-fill-on-missing behavior. */
export function flattenHolisticFrame(
  poseLandmarks: RawLandmark[] | null | undefined,
  leftHandLandmarks: RawLandmark[] | null | undefined,
  rightHandLandmarks: RawLandmark[] | null | undefined
): Float32Array {
  const pose = flattenPart(poseLandmarks, POSE_VALUES_PER_LANDMARK, NUM_POSE_LANDMARKS);
  const left = flattenPart(leftHandLandmarks, HAND_VALUES_PER_LANDMARK, NUM_HAND_LANDMARKS);
  const right = flattenPart(rightHandLandmarks, HAND_VALUES_PER_LANDMARK, NUM_HAND_LANDMARKS);

  const out = new Float32Array(FEATURE_DIM);
  out.set(pose, 0);
  out.set(left, POSE_DIM);
  out.set(right, POSE_DIM + HAND_DIM);
  return out;
}

/** Whether at least one hand was detected this frame -- used by the UI
 * to show a "no hands visible" hint, and to skip sending frames where
 * absolutely nothing was detected (pure padding, not worth the
 * bandwidth or the model's inference cycles). */
export function hasAnyHandDetection(
  leftHandLandmarks: RawLandmark[] | null | undefined,
  rightHandLandmarks: RawLandmark[] | null | undefined
): boolean {
  return Boolean(leftHandLandmarks?.length) || Boolean(rightHandLandmarks?.length);
}
