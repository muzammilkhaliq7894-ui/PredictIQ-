import type { EvaluationResponse } from "./types";

export type Dataset = "cycle001" | "ai4i2020" | "nasa_fd001" | "nasa_fd002" | "nasa_fd003" | "nasa_fd004";
export type DatasetTask = "anomaly" | "failure";

export const DATASETS: {
  cycle001: { id: "cycle001"; label: "Cycle 001" };
  ai4i2020: { id: "ai4i2020"; label: "AI4I 2020" };
  nasa_fd001: { id: "nasa_fd001"; label: "NASA C-MAPSS FD001" };
  nasa_fd002: { id: "nasa_fd002"; label: "NASA C-MAPSS FD002" };
  nasa_fd003: { id: "nasa_fd003"; label: "NASA C-MAPSS FD003" };
  nasa_fd004: { id: "nasa_fd004"; label: "NASA C-MAPSS FD004" };
};

export function getDatasetSummary(evaluation: EvaluationResponse, dataset: "cycle001"): {
  dataset: "cycle001";
  rows: number;
  features: number;
  alerts: number;
  alertRate: number;
  flagged: EvaluationResponse["cycle_flagged"];
};

export function getDatasetSummary(evaluation: EvaluationResponse, dataset: "ai4i2020"): {
  dataset: "ai4i2020";
  rows: number;
  failures: number;
  positiveRate: number;
  metrics: EvaluationResponse["metrics"];
  confusionMatrices: EvaluationResponse["confusion_matrices"];
};

export function supportsDataset(dataset: Dataset, task: DatasetTask): boolean;
