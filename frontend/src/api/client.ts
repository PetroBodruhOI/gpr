import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "/api";

export const api = axios.create({ baseURL: API_BASE });

export interface TaskStatus {
  task_id: string;
  status: "pending" | "processing" | "done" | "error";
  progress: number;
  message: string;
  result?: PredictResult;
}

export interface ChunkResult {
  chunk_idx: number;
  time_start: number;
  time_end: number;
  label: string;
  confidence: number;
  probs: Record<string, number>;
}

export interface PredictResult {
  final_label: string;
  final_conf: number;
  mean_probs: Record<string, number>;
  chunks: ChunkResult[];
  n_chunks: number;
}

export async function predictUrl(url: string,
                                 start_sec?: number,
                                 duration_sec?: number): Promise<string> {
  const { data } = await api.post("/predict/url", { url, start_sec, duration_sec });
  return data.task_id;
}

export async function predictFile(file: File): Promise<string> {
  const fd = new FormData();
  fd.append("file", file);
  const { data } = await api.post("/predict/file", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data.task_id;
}

export async function getTask(taskId: string): Promise<TaskStatus> {
  const { data } = await api.get(`/tasks/${taskId}`);
  return data;
}

// frontend/src/api/client.ts
/** Polling кожні 1.5 сек, поки статус не done/error. */
export async function pollTask(
  taskId: string,
  onUpdate: (t: TaskStatus) => void,
  intervalMs = 1500,
): Promise<TaskStatus> {
  while (true) {
    const t = await getTask(taskId);
    onUpdate(t);
    if (t.status === "done" || t.status === "error") return t;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

export async function submitFeedback(taskId: string,
                                     rating: "good" | "bad"): Promise<void> {
  await api.post(`/feedback/${taskId}`, { rating });
}
