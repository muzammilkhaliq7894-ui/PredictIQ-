export const DATASETS = {
  cycle001: { id: "cycle001", label: "Cycle 001" },
  ai4i2020: { id: "ai4i2020", label: "AI4I 2020" },
  nasa_fd001: { id: "nasa_fd001", label: "NASA C-MAPSS FD001" },
  nasa_fd002: { id: "nasa_fd002", label: "NASA C-MAPSS FD002" },
  nasa_fd003: { id: "nasa_fd003", label: "NASA C-MAPSS FD003" },
  nasa_fd004: { id: "nasa_fd004", label: "NASA C-MAPSS FD004" }
};

export function getDatasetSummary(evaluation, dataset) {
  if (dataset === "cycle001") {
    return {
      dataset,
      rows: evaluation.cycle_summary.rows,
      features: evaluation.cycle_summary.features_used,
      alerts: evaluation.cycle_summary.alerts,
      alertRate: evaluation.cycle_summary.alert_rate,
      flagged: evaluation.cycle_flagged
    };
  }

  const failures = evaluation.class_balance.find(row => row.label === 1);
  return {
    dataset,
    rows: evaluation.class_balance.reduce((sum, row) => sum + row.rows, 0),
    failures: failures?.rows ?? 0,
    positiveRate: failures?.rate ?? 0,
    metrics: evaluation.metrics,
    confusionMatrices: evaluation.confusion_matrices
  };
}

export function supportsDataset(dataset, task) {
  return (task === "anomaly" && dataset === "cycle001")
    || (task === "failure" && dataset === "ai4i2020");
}
