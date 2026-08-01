import axios, { AxiosError } from "axios";
import type { HealthStatus, HistoryFilters, Metrics, Prediction } from "@/types/prediction";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
});

/** Human-readable message for a failed API call, distinguishing the
 * "model not trained yet" case (503) from generic failures so the UI can
 * show useful guidance instead of a bare "Error". */
export function describeApiError(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<{ detail?: string }>;
    if (axiosError.response?.status === 503) {
      return "The model isn't loaded yet -- train it and place a checkpoint, or check back once deployment finishes.";
    }
    if (axiosError.response?.data?.detail) {
      return axiosError.response.data.detail;
    }
    if (axiosError.code === "ECONNABORTED") {
      return "The request timed out. Is the backend running?";
    }
    if (!axiosError.response) {
      return "Couldn't reach the backend. Is it running at " + API_BASE_URL + "?";
    }
  }
  return "Something went wrong. Please try again.";
}

export async function getHealth(): Promise<HealthStatus> {
  const { data } = await client.get<HealthStatus>("/health");
  return data;
}

export async function predictUpload(file: File): Promise<Prediction> {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await client.post<Prediction>("/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function predictWebcamFrame(imageBase64: string): Promise<Prediction> {
  const { data } = await client.post<Prediction>("/predict-webcam", {
    image_base64: imageBase64,
  });
  return data;
}

export async function getMetrics(): Promise<Metrics> {
  const { data } = await client.get<Metrics>("/metrics");
  return data;
}

export async function getHistory(filters: HistoryFilters = {}): Promise<Prediction[]> {
  const { data } = await client.get<Prediction[]>("/history", { params: filters });
  return data;
}
