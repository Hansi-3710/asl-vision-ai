"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getStreamWebSocketUrl } from "@/lib/api";
import type {
  ConversationEntry,
  StreamClientMessage,
  StreamReadyMessage,
  StreamServerMessage,
  WordPrediction,
} from "@/types/streaming";

// Forwarding every MediaPipe detection (often 30-60fps) to the backend
// would flood the WebSocket and the transformer's inference cadence for
// no benefit -- the backend only runs an inference tick every
// SEQUENCE_INFERENCE_STRIDE_FRAMES (8) frames anyway. ~15fps client-side
// sampling is a reasonable middle ground: smooth enough motion capture,
// a small fraction of the bandwidth of sending images.
const SEND_INTERVAL_MS = 1000 / 15;

// If the connection drops (backend restart, network blip), retry rather
// than leaving the user stuck on "Offline" until they reload the page.
const RECONNECT_DELAY_MS = 2000;

export type ConnectionStatus = "connecting" | "open" | "closed" | "error";

interface UseContinuousRecognitionOptions {
  enabled: boolean;
}

interface UseContinuousRecognitionResult {
  status: ConnectionStatus;
  readyInfo: StreamReadyMessage | null;
  transcript: string;
  words: WordPrediction[];
  bufferFrames: number;
  latencyMs: number;
  fps: number;
  history: ConversationEntry[];
  lastError: string | null;
  sendFrame: (features: Float32Array) => void;
  startNewConversation: () => void;
}

export function useContinuousRecognition({
  enabled,
}: UseContinuousRecognitionOptions): UseContinuousRecognitionResult {
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const [readyInfo, setReadyInfo] = useState<StreamReadyMessage | null>(null);
  const [transcript, setTranscript] = useState("");
  const [words, setWords] = useState<WordPrediction[]>([]);
  const [bufferFrames, setBufferFrames] = useState(0);
  const [latencyMs, setLatencyMs] = useState(0);
  const [fps, setFps] = useState(0);
  const [history, setHistory] = useState<ConversationEntry[]>([]);
  const [lastError, setLastError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const lastSendRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Mirrors transcript/words in a ref too, so startNewConversation (called
  // from a click handler, not a render) always reads the latest values
  // without needing to be re-created every time transcript/words change.
  const latestRef = useRef({ transcript: "", words: [] as WordPrediction[] });
  latestRef.current = { transcript, words };

  const send = useCallback((message: StreamClientMessage) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  }, []);

  useEffect(() => {
    if (!enabled) {
      wsRef.current?.close();
      wsRef.current = null;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      setStatus("closed");
      return;
    }

    let cancelled = false;

    function connect() {
      if (cancelled) return;
      setStatus("connecting");
      const ws = new WebSocket(getStreamWebSocketUrl());
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus("open");
        setLastError(null);
      };

      ws.onmessage = (event) => {
        let message: StreamServerMessage;
        try {
          message = JSON.parse(event.data);
        } catch {
          return; // ignore a malformed frame rather than crashing the UI
        }

        switch (message.type) {
          case "ready":
            setReadyInfo(message);
            break;
          case "update":
            setTranscript(message.transcript);
            setWords(message.words);
            setBufferFrames(message.buffer_frames);
            setLatencyMs(message.latency_ms);
            setFps(message.fps);
            break;
          case "error":
            setLastError(message.message);
            break;
          case "pong":
            break;
        }
      };

      ws.onerror = () => {
        setStatus("error");
      };

      ws.onclose = () => {
        wsRef.current = null;
        if (cancelled) return;
        setStatus("closed");
        reconnectTimerRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
      };
    }

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [enabled]);

  const sendFrame = useCallback(
    (features: Float32Array) => {
      const now = performance.now();
      if (now - lastSendRef.current < SEND_INTERVAL_MS) return; // throttle
      lastSendRef.current = now;
      send({ type: "frame", features: Array.from(features), timestamp_ms: Math.round(now) });
    },
    [send]
  );

  const startNewConversation = useCallback(() => {
    const { transcript: currentTranscript, words: currentWords } = latestRef.current;
    if (currentTranscript) {
      setHistory((prev) => [
        ...prev,
        { id: `${Date.now()}`, transcript: currentTranscript, words: currentWords, endedAt: Date.now() },
      ]);
    }
    setTranscript("");
    setWords([]);
    send({ type: "reset" });
  }, [send]);

  return {
    status,
    readyInfo,
    transcript,
    words,
    bufferFrames,
    latencyMs,
    fps,
    history,
    lastError,
    sendFrame,
    startNewConversation,
  };
}
