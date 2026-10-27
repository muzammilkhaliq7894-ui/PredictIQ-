import assert from "node:assert/strict";
import test from "node:test";
import { DATASETS, getDatasetSummary, supportsDataset } from "../src/datasets.mjs";

const evaluation = {
  metrics: [{ model: "random_forest_balanced" }],
  confusion_matrices: [{ model: "random_forest_balanced" }],
  cycle_summary: {
    rows: 10132,
    features_used: 42,
    alerts: 102,
    alert_rate: 0.0101
  },
  class_balance: [
    { label: 0, rows: 9661, rate: 0.9661 },
    { label: 1, rows: 339, rate: 0.0339 }
  ],
  cycle_flagged: [{ timestamp: "cycle-row", machine_id: 1, anomaly_score: 0.1, alert: true }]
};

test("dataset selector offers only implemented datasets", () => {
  assert.deepEqual(Object.keys(DATASETS), ["cycle001", "ai4i2020", "nasa_fd001", "nasa_fd002", "nasa_fd003", "nasa_fd004"]);
  assert.equal(DATASETS.cycle001.label, "Cycle 001");
  assert.equal(DATASETS.ai4i2020.label, "AI4I 2020");
  assert.equal(DATASETS.nasa_fd004.label, "NASA C-MAPSS FD004");
});

test("Cycle 001 summary uses only its anomaly report", () => {
  assert.deepEqual(getDatasetSummary(evaluation, "cycle001"), {
    dataset: "cycle001",
    rows: 10132,
    features: 42,
    alerts: 102,
    alertRate: 0.0101,
    flagged: evaluation.cycle_flagged
  });
});

test("AI4I summary uses only its supervised evaluation reports", () => {
  assert.deepEqual(getDatasetSummary(evaluation, "ai4i2020"), {
    dataset: "ai4i2020",
    rows: 10000,
    failures: 339,
    positiveRate: 0.0339,
    metrics: evaluation.metrics,
    confusionMatrices: evaluation.confusion_matrices
  });
});

test("each prediction task is restricted to its trained dataset", () => {
  assert.equal(supportsDataset("cycle001", "anomaly"), true);
  assert.equal(supportsDataset("ai4i2020", "anomaly"), false);
  assert.equal(supportsDataset("ai4i2020", "failure"), true);
  assert.equal(supportsDataset("cycle001", "failure"), false);
  assert.equal(supportsDataset("nasa", "anomaly"), false);
  assert.equal(supportsDataset("nasa", "failure"), false);
  for (const dataset of ["nasa_fd001", "nasa_fd002", "nasa_fd003", "nasa_fd004"]) {
    assert.equal(supportsDataset(dataset, "anomaly"), false);
    assert.equal(supportsDataset(dataset, "failure"), false);
  }
});
