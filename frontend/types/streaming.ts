/** Mirrors backend/app/schemas/streaming.py's WordPrediction. */
export interface WordPrediction {
  word: string;
  confidence: number;
}

/** Sent once by the server right after the WebSocket connects.
 * Mirrors backend/app/schemas/streaming.py's StreamReadyMessage. */
export interface StreamReadyMessage {
  type: "ready";
  model_ready: boolean;
  is_synthetic_placeholder: boolean;
  feature_dim: number;
  vocab_size: number;
  window_frames: number;
}

/** Sent after each inference tick. Mirrors StreamUpdateMessage. */
export interface StreamUpdateMessage {
  type: "update";
  transcript: string;
  words: WordPrediction[];
  buffer_frames: number;
  latency_ms: number;
  fps: number;
}

/** Mirrors StreamErrorMessage -- sent for a malformed individual frame;
 * does NOT close the connection. */
export interface StreamErrorMessage {
  type: "error";
  message: string;
}

export interface StreamPongMessage {
  type: "pong";
}

export type StreamServerMessage =
  | StreamReadyMessage
  | StreamUpdateMessage
  | StreamErrorMessage
  | StreamPongMessage;

/** Client -> server messages. Mirrors LandmarkFrameMessage / ResetMessage. */
export interface StreamFrameMessage {
  type: "frame";
  features: number[];
  timestamp_ms?: number;
}

export interface StreamResetMessage {
  type: "reset";
}

export interface StreamPingMessage {
  type: "ping";
}

export type StreamClientMessage = StreamFrameMessage | StreamResetMessage | StreamPingMessage;

/** One completed utterance archived into the on-screen conversation
 * history when the user resets/starts a new one -- a purely
 * frontend-side concept, no backend equivalent (the backend only ever
 * knows about "the current buffer's transcript"). */
export interface ConversationEntry {
  id: string;
  transcript: string;
  words: WordPrediction[];
  endedAt: number; // Date.now() timestamp
}
