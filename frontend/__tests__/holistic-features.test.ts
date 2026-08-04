import {
  FEATURE_DIM,
  POSE_DIM,
  HAND_DIM,
  NUM_POSE_LANDMARKS,
  NUM_HAND_LANDMARKS,
  flattenHolisticFrame,
  hasAnyHandDetection,
  type RawLandmark,
} from "@/lib/holistic-features";
import { getStreamWebSocketUrl } from "@/lib/api";

function makeLandmarks(count: number, value: number, withVisibility = false): RawLandmark[] {
  return Array.from({ length: count }, () => ({
    x: value,
    y: value,
    z: value,
    ...(withVisibility ? { visibility: 1 } : {}),
  }));
}

describe("FEATURE_DIM layout constants", () => {
  it("matches backend/continuous_pipeline/landmark_spec.py's FEATURE_DIM (258)", () => {
    expect(FEATURE_DIM).toBe(258);
    expect(POSE_DIM).toBe(132);
    expect(HAND_DIM).toBe(63);
    expect(POSE_DIM + 2 * HAND_DIM).toBe(FEATURE_DIM);
  });
});

describe("flattenHolisticFrame", () => {
  it("produces a FEATURE_DIM-length Float32Array when everything is detected", () => {
    const pose = makeLandmarks(NUM_POSE_LANDMARKS, 0.5, true);
    const leftHand = makeLandmarks(NUM_HAND_LANDMARKS, 0.3);
    const rightHand = makeLandmarks(NUM_HAND_LANDMARKS, 0.7);

    const result = flattenHolisticFrame(pose, leftHand, rightHand);
    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(FEATURE_DIM);
  });

  it("zero-fills missing hands rather than shrinking the vector", () => {
    const pose = makeLandmarks(NUM_POSE_LANDMARKS, 0.5, true);
    const result = flattenHolisticFrame(pose, null, undefined);

    expect(result.length).toBe(FEATURE_DIM);
    const leftHandPart = result.slice(POSE_DIM, POSE_DIM + HAND_DIM);
    const rightHandPart = result.slice(POSE_DIM + HAND_DIM, POSE_DIM + 2 * HAND_DIM);
    expect(Array.from(leftHandPart).every((v) => v === 0)).toBe(true);
    expect(Array.from(rightHandPart).every((v) => v === 0)).toBe(true);
  });

  it("returns an all-zero vector when nothing was detected at all", () => {
    const result = flattenHolisticFrame(null, null, null);
    expect(result.length).toBe(FEATURE_DIM);
    expect(Array.from(result).every((v) => v === 0)).toBe(true);
  });

  it("packs pose as (x,y,z,visibility) and hands as (x,y,z) -- matches landmark_spec.py exactly", () => {
    const pose = [{ x: 0.1, y: 0.2, z: 0.3, visibility: 0.9 }, ...makeLandmarks(NUM_POSE_LANDMARKS - 1, 0)];
    const leftHand = [{ x: 0.4, y: 0.5, z: 0.6 }, ...makeLandmarks(NUM_HAND_LANDMARKS - 1, 0)];

    const result = flattenHolisticFrame(pose, leftHand, null);

    expect(Array.from(result.slice(0, 4))).toEqual([0.1, 0.2, 0.3, 0.9].map((v) => Math.fround(v)));
    expect(Array.from(result.slice(POSE_DIM, POSE_DIM + 3))).toEqual(
      [0.4, 0.5, 0.6].map((v) => Math.fround(v))
    );
  });

  it("truncates defensively if more landmarks than expected are somehow passed in", () => {
    const tooManyPose = makeLandmarks(NUM_POSE_LANDMARKS + 10, 0.5, true);
    const result = flattenHolisticFrame(tooManyPose, null, null);
    expect(result.length).toBe(FEATURE_DIM);
  });
});

describe("hasAnyHandDetection", () => {
  it("is false when neither hand is present", () => {
    expect(hasAnyHandDetection(null, undefined)).toBe(false);
    expect(hasAnyHandDetection([], [])).toBe(false);
  });

  it("is true when at least one hand is present", () => {
    expect(hasAnyHandDetection(makeLandmarks(NUM_HAND_LANDMARKS, 0.5), null)).toBe(true);
    expect(hasAnyHandDetection(null, makeLandmarks(NUM_HAND_LANDMARKS, 0.5))).toBe(true);
  });
});

describe("getStreamWebSocketUrl", () => {
  it("derives a ws:// URL from the default http API base URL, stripping /api and appending /ws/stream", () => {
    // No NEXT_PUBLIC_API_URL set in the test env -> lib/api.ts's default
    // "http://localhost:8000/api" applies.
    expect(getStreamWebSocketUrl()).toBe("ws://localhost:8000/ws/stream");
  });
});
