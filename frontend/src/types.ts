export type Page = "overview" | "explorer" | "anomaly" | "failure" | "evaluation" | "nasa-rul";

export interface DatasetColumn {
  name: string;
  dtype: string;
  missing_count: number;
  missing_percent: number;
}

export interface DatasetProfile {
  id: string;
  name: string;
  row_count: number;
  column_count: number;
  columns: DatasetColumn[];
  preview: Array<Record<string, string | number | boolean | null>>;
  preview_rows: number;
}

export interface DatasetStatistic {
  column: string;
  count: number;
  mean: number;
  std: number;
  min: number;
  max: number;
}

export interface DatasetDistribution {
  column: string;
  bins: Array<{ range: string; count: number }>;
}

export interface DatasetCategoryCounts {
  column: string;
  values: Array<{ value: string; count: number }>;
}

export interface DatasetAnalysis {
  row_count: number;
  sample_size: number;
  statistics: DatasetStatistic[];
  numeric_distributions: DatasetDistribution[];
  categorical_counts: DatasetCategoryCounts[];
}

export interface ModelInfo {
  name: string;
  version: string;
  type: string;
  feature_names: string[];
  limitations: string[];
}

export interface HealthResponse {
  status: string;
  model_version: string;
}

export interface ModelsResponse {
  models: ModelInfo[];
}

export interface AnomalyResponse {
  anomaly_score: number;
  alert: boolean;
  model_version: string;
  limitations: string;
}

export interface FailureResponse {
  failure_probability: number;
  predicted_failure: boolean;
  model_version: string;
  limitations: string;
}

export interface TelemetryRow {
  timestamp: string;
  values: Record<string, number>;
}

export interface TelemetryResponse {
  dataset: "cycle001";
  rows: TelemetryRow[];
}

export interface AnomalyPoint {
  timestamp: string;
  score: number;
  alert: boolean;
}

export interface MetricRow {
  model: string;
  test_rows: number;
  positive_rate: number;
  precision: number;
  recall: number;
  f1: number;
  average_precision: number;
}

export interface ConfusionRow {
  model: string;
  actual: string;
  predicted_0: number;
  predicted_1: number;
}

export interface EvaluationResponse {
  metrics: MetricRow[];
  confusion_matrices: ConfusionRow[];
  cycle_summary: {
    contamination: number;
    rows: number;
    features_used: number;
    alerts: number;
    alert_rate: number;
    interpretation: string;
  };
  class_balance: Array<{ label: number; rows: number; rate: number }>;
  cycle_flagged: Array<{
    timestamp: string;
    machine_id: number;
    anomaly_score: number;
    alert: boolean;
  }>;
  source: string;
}

export interface ModelComparisonRow {
  model: string;
  role: string;
  precision?: number;
  precision_std?: number;
  recall?: number;
  recall_std?: number;
  f1?: number;
  f1_std?: number;
  average_precision?: number;
  average_precision_std?: number;
  mae?: number;
  mae_std?: number;
  rmse?: number;
  rmse_std?: number;
  r2?: number;
  r2_std?: number;
  nasa_score?: number;
  nasa_score_std?: number;
  rows?: number;
  features?: number;
  alerts?: number;
  alert_rate?: number;
  threshold?: number;
  overlap_with_current?: number;
  jaccard_with_current?: number;
  parameters?: Record<string, string | number | boolean>;
  limitations?: string;
}

export interface ModelComparisonsResponse {
  dataset: string;
  model_type: string;
  model_comparisons: ModelComparisonRow[];
  metadata: Record<string, string | number | boolean | string[] | Record<string, string | number | boolean> | undefined>;
  recommendation: string;
  official_test_current?: NasaRulMetrics;
  official_test_note?: string;
  limitations?: string;
}

export interface NasaDatasetSplitSummary {
  rows: number;
  columns: number;
  engine_count: number;
  engine_ids: number[];
  numeric_summary: DatasetStatistic[];
}

export interface NasaDatasetOverview {
  id: string;
  name: string;
  columns: string[];
  settings: string[];
  sensors: string[];
  training: NasaDatasetSplitSummary;
  test: NasaDatasetSplitSummary;
  rul_target: string;
}

export interface NasaEngineHistory {
  dataset: string;
  split: "train" | "test";
  engine_id: number;
  columns: string[];
  rows: Array<Record<string, number>>;
}

export interface NasaRulMetrics {
  mae: number;
  rmse: number;
  r2: number;
  nasa_score: number;
  engine_count: number;
  row_count: number;
  split: string;
}

export interface NasaRulEvaluation {
  dataset: string;
  validation: NasaRulMetrics;
  official_test: NasaRulMetrics;
  target_definition: string;
  features: string[];
  limitations: string;
}

export interface NasaRulPrediction {
  dataset: string;
  engine_id: number;
  last_observed_cycle: number;
  predicted_rul: number;
  benchmark_rul: number;
  evaluation: NasaRulMetrics;
  limitations: string;
}
