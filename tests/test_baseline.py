from pathlib import Path

import pandas as pd
import pytest

from data.loaders import load_ai4i2020, load_cycle001
from models.baseline import (
    ai4i_label_inconsistencies,
    cycle_numeric_matrix,
    run_ai4i_baselines,
    run_ai4i_model_comparison,
    run_cycle_baseline,
)

ROOT = Path(__file__).resolve().parents[1]


def test_cycle_matrix_excludes_identifiers_and_constants():
    frame = load_cycle001(ROOT / "docs" / "cycle_001_machine_data.csv")
    matrix, audit, correlated = cycle_numeric_matrix(frame)
    assert "Datum/Zeit" not in matrix
    assert "Maschinennummer" not in matrix
    assert all(matrix[column].nunique() > 1 for column in matrix)
    assert {"feature_a", "feature_b", "absolute_correlation"} <= set(correlated.columns)
    assert (audit["reason"] == "constant").any()


def test_cycle_baseline_is_configurable_and_flags_only_alerts():
    frame = load_cycle001(ROOT / "docs" / "cycle_001_machine_data.csv")
    results = run_cycle_baseline(frame, contamination=0.02)
    assert results["summary"].iloc[0]["contamination"] == pytest.approx(0.02)
    assert results["scores"]["alert"].sum() > 0
    assert set(results["scores"].columns) == {"timestamp", "machine_id", "anomaly_score", "alert"}


def test_ai4i_inconsistencies_are_reported_without_label_changes():
    frame = load_ai4i2020(ROOT / "docs" / "ai4i2020.csv")
    inconsistencies = ai4i_label_inconsistencies(frame)
    assert len(inconsistencies) == 27
    assert frame["Machine failure"].sum() == 339


def test_ai4i_baselines_use_separate_stratified_evaluation():
    frame = load_ai4i2020(ROOT / "docs" / "ai4i2020.csv")
    results = run_ai4i_baselines(frame)
    assert list(results["metrics"]["model"]) == [
        "dummy_stratified",
        "logistic_regression_balanced",
        "random_forest_balanced",
        "random_forest_balanced_leaf2",
    ]
    assert set(["precision", "recall", "f1", "average_precision"]) <= set(results["metrics"])
    assert results["class_balance"]["rows"].sum() == len(frame)


def test_ai4i_candidate_improves_current_model_across_repeated_splits():
    comparison = run_ai4i_model_comparison(
        load_ai4i2020(ROOT / "docs" / "ai4i2020.csv")
    )
    rows = {row["role"]: row for row in comparison["model_comparisons"]}
    current = rows["current"]
    candidate = rows["candidate"]

    assert comparison["metadata"]["random_seeds"] == [42, 17, 91]
    assert "Machine failure" not in comparison["features"]
    assert all(
        name not in comparison["features"]
        for name in ("UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF")
    )
    assert candidate["f1"] > current["f1"]
    assert candidate["average_precision"] > current["average_precision"]
    assert all(
        candidate_row["f1"] > current_row["f1"]
        and candidate_row["average_precision"] > current_row["average_precision"]
        for candidate_row, current_row in zip(candidate["seed_results"], current["seed_results"])
    )
