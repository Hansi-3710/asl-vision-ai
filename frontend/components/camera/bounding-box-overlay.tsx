"use client";

import { motion, AnimatePresence } from "framer-motion";
import type { Prediction } from "@/types/prediction";
import { formatConfidence } from "@/lib/utils";

interface BoundingBoxOverlayProps {
  prediction: Prediction | null;
}

/**
 * Draws the detected-hand bounding box (normalized 0-1 coordinates from
 * the backend's MediaPipe hand detector) as an absolutely-positioned
 * overlay using percentage-based CSS, so it stays correctly aligned
 * regardless of the video element's rendered size.
 */
export function BoundingBoxOverlay({ prediction }: BoundingBoxOverlayProps) {
  const box = prediction?.bounding_box;

  return (
    <div className="pointer-events-none absolute inset-0">
      <AnimatePresence>
        {box && (
          <motion.div
            key="bbox"
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ duration: 0.15 }}
            className="absolute border-2 border-signal rounded-lg"
            style={{
              left: `${box.x_min * 100}%`,
              top: `${box.y_min * 100}%`,
              width: `${(box.x_max - box.x_min) * 100}%`,
              height: `${(box.y_max - box.y_min) * 100}%`,
              boxShadow: "0 0 24px rgba(76,224,210,0.4)",
            }}
          >
            {/* Corner brackets, echoing the site's signature viewfinder motif */}
            {(["-top-1 -left-1", "-top-1 -right-1", "-bottom-1 -left-1", "-bottom-1 -right-1"] as const).map(
              (position, i) => (
                <span
                  key={position}
                  className={`absolute h-3 w-3 border-signal ${position} ${
                    i === 0
                      ? "border-l-2 border-t-2 rounded-tl-sm"
                      : i === 1
                      ? "border-r-2 border-t-2 rounded-tr-sm"
                      : i === 2
                      ? "border-l-2 border-b-2 rounded-bl-sm"
                      : "border-r-2 border-b-2 rounded-br-sm"
                  }`}
                />
              )
            )}

            {prediction && prediction.predicted_class && (
              <div className="absolute -top-9 left-0 flex items-center gap-2 rounded-md bg-signal px-2.5 py-1 font-mono text-xs font-semibold text-background">
                <span>{prediction.predicted_class.toUpperCase()}</span>
                <span className="opacity-70">{formatConfidence(prediction.confidence)}</span>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {prediction && prediction.hand_detected === false && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-full bg-glass px-4 py-1.5 text-xs text-ink-muted backdrop-blur-md">
          No hand detected -- show your hand to the camera
        </div>
      )}
    </div>
  );
}
