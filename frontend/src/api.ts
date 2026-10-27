import type {
  AnomalyResponse,
  FailureResponse,
  EvaluationResponse,
  DatasetAnalysis,
  DatasetProfile,
  HealthResponse,
  ModelsResponse,
  ModelComparisonsResponse,
  NasaDatasetOverview,
  NasaEngineHistory,
  NasaRulEvaluation,
  NasaRulPrediction,
  TelemetryResponse
} from "./types";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8011").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: init?.method === "POST"
      ? { "Content-Type": "application/json", ...init.headers }
      : init?.headers
  });
  if (!response.ok) {
    let detail = `Request failed with HTTP ${response.status}`;
    try {
      const body = await response.json() as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // Keep the HTTP status when the server does not return JSON.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  models: () => request<ModelsResponse>("/models"),
  evaluation: () => request<EvaluationResponse>("/evaluation"),
  modelComparisons: (dataset: string) =>
    request<ModelComparisonsResponse>(`/model-comparisons/${encodeURIComponent(dataset)}`),
  datasetProfile: (dataset: string) => request<DatasetProfile>(`/datasets/${encodeURIComponent(dataset)}`),
  analyzeDataset: (dataset: string) => request<DatasetAnalysis>(`/datasets/${encodeURIComponent(dataset)}/analysis`),
  nasaOverview: (dataset: string) => request<NasaDatasetOverview>(`/nasa/${encodeURIComponent(dataset)}/overview`),
  nasaEngine: (dataset: string, engineId: number, split: "train" | "test") =>
    request<NasaEngineHistory>(`/nasa/${encodeURIComponent(dataset)}/engines/${engineId}?split=${split}`),
  nasaRulEvaluation: (dataset: string) => request<NasaRulEvaluation>(`/nasa/${encodeURIComponent(dataset)}/rul`),
  nasaRulPredict: (dataset: string, engineId: number) =>
    request<NasaRulPrediction>(`/nasa/${encodeURIComponent(dataset)}/rul/predict`, {
      method: "POST",
      body: JSON.stringify({ engine_id: engineId })
    }),
  cycleTelemetry: () => request<TelemetryResponse>("/datasets/cycle001/telemetry"),
  scoreAnomaly: (features: Record<string, number>) =>
    request<AnomalyResponse>("/anomaly/score", {
      method: "POST",
      body: JSON.stringify({ features })
    }),
  predictFailure: (payload: {
    Type: string;
    air_temperature: number;
    process_temperature: number;
    rotational_speed: number;
    torque: number;
    tool_wear: number;
  }) =>
    request<FailureResponse>("/failure/predict", {
      method: "POST",
      body: JSON.stringify(payload)
    })
};

export const API_BASE = API_BASE_URL;
