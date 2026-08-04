/** Mirrors backend/app/schemas/prediction.py's TopKEntry. */
export interface TopKEntry {
  class: string;
  confidence: number;
}

/** Mirrors backend/app/schemas/prediction.py's BoundingBoxSchema. */
export interface BoundingBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  confidence: number;
}

/** Mirrors backend/app/schemas/prediction.py's PredictionResponse. */
export interface Prediction {
  id: string;
  predicted_class: string;
  confidence: number;
  top_k: TopKEntry[];
  source: "upload" | "webcam";
  latency_ms: number;
  image_path: string | null;
  bounding_box: BoundingBox | null;
  /** true = hand found, false = detector ran but found nothing (full-frame
   * fallback used), null = hand detection wasn't available at all. */
  hand_detected: boolean | null;
  created_at: string | null;
}

/** Mirrors backend/app/schemas/metrics.py's HealthResponse. */
export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  model_architecture: string | null;
  device: string | null;
}

/** Mirrors backend/app/schemas/metrics.py's LetterCount. */
export interface LetterCount {
  letter: string;
  count: number;
}

/** Mirrors backend/app/schemas/metrics.py's MetricsResponse. */
export interface Metrics {
  total_predictions: number;
  average_confidence: number;
  average_latency_ms: number;
  most_predicted_letters: LetterCount[];
  predictions_by_source: Record<string, number>;
}

export interface HistoryFilters {
  limit?: number;
  offset?: number;
  source?: "upload" | "webcam";
  predicted_class?: string;
}
