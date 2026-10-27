"""Leakage-conscious baseline models for the independent Phase 2 datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sklearn

RANDOM_STATE = 42
AI4I_VALIDATION_SEEDS = (42, 17, 91)
AI4I_CURRENT_RF_PARAMS = {
    "n_estimators": 300,
    "class_weight": "balanced",
}
AI4I_CURRENT_MODEL_VERSION = "phase3-v1"
AI4I_CANDIDATE_MODEL_VERSION = "phase4-v2"
AI4I_CANDIDATE_RF_PARAMS = {
    "n_estimators": 400,
    "class_weight": "balanced",
    "max_features": 0.8,
    "min_samples_leaf": 2,
}
AI4I_TARGET = "Machine failure"
AI4I_EXCLUDED = {
    "UDI",
    "Product ID",
    AI4I_TARGET,
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
}


def cycle_numeric_matrix(
    frame: pd.DataFrame, correlation_threshold: float = 0.95
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return usable telemetry, an exclusion audit, and highly correlated pairs."""
    audit: list[dict[str, Any]] = []
    usable: list[str] = []
    for column in frame.columns:
        series = frame[column]
        numeric = pd.api.types.is_numeric_dtype(series)
        excluded = (
            "timestamp_or_identifier"
            if column in {"Datum/Zeit", "Maschinennummer"}
            else "non_numeric"
            if not numeric
            else "empty"
            if series.isna().all()
            else "constant"
            if series.nunique(dropna=True) <= 1
            else ""
        )
        audit.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "unique_count": int(series.nunique(dropna=True)),
                "missing_count": int(series.isna().sum()),
                "status": "excluded" if excluded else "included",
                "reason": excluded or "telemetry_feature",
            }
        )
        if not excluded:
            usable.append(column)
    matrix = frame[usable].copy()
    correlations = matrix.corr().abs()
    pairs = [
        {
            "feature_a": left,
            "feature_b": right,
            "absolute_correlation": float(correlations.loc[left, right]),
        }
        for index, left in enumerate(usable)
        for right in usable[index + 1 :]
        if correlations.loc[left, right] >= correlation_threshold
    ]
    return matrix, pd.DataFrame(audit), pd.DataFrame(pairs)


def run_cycle_model_comparison(
    frame: pd.DataFrame, model: Any, model_metadata: dict[str, Any]
) -> dict[str, Any]:
    """Compare unlabeled Cycle 001 alert behavior without inventing target metrics."""
    matrix, _, _ = cycle_numeric_matrix(frame)
    if list(matrix.columns) != model_metadata["feature_names"]:
        raise ValueError("Cycle 001 features do not match the loaded anomaly model schema")
    current_alert = model.predict(matrix) == -1
    median = matrix.median()
    mad = (matrix - median).abs().median()
    usable_mad = mad[mad > 0]
    if usable_mad.empty:
        raise ValueError("No Cycle 001 features have non-zero median absolute deviation")
    robust_score = (
        (matrix[usable_mad.index] - median[usable_mad.index])
        .abs()
        .div(usable_mad, axis="columns")
        .max(axis=1)
    )
    threshold = float(robust_score.quantile(0.99))
    simple_alert = robust_score >= threshold
    intersection = int(np.logical_and(current_alert, simple_alert).sum())
    union = int(np.logical_or(current_alert, simple_alert).sum())
    return {
        "dataset": "cycle001",
        "model_type": "unsupervised_anomaly_detection",
        "model_comparisons": [
            {
                "model": "isolation_forest_current",
                "role": "current",
                "rows": int(len(matrix)),
                "features": int(matrix.shape[1]),
                "alerts": int(current_alert.sum()),
                "alert_rate": float(current_alert.mean()),
                "limitations": "Unlabeled in-sample descriptive output; no supervised performance metrics.",
            },
            {
                "model": "robust_mad_top_1pct",
                "role": "simple_baseline",
                "rows": int(len(matrix)),
                "features": int(len(usable_mad)),
                "alerts": int(simple_alert.sum()),
                "alert_rate": float(simple_alert.mean()),
                "threshold": threshold,
                "overlap_with_current": intersection,
                "jaccard_with_current": float(intersection / union) if union else 1.0,
                "limitations": "Unlabeled descriptive comparator only; not a validated detector.",
            },
        ],
        "metadata": {
            "model_version": model_metadata["model_version"],
            "model_type": model_metadata["model_type"],
            "random_state": model_metadata["random_state"],
            "contamination": model_metadata["contamination"],
            "feature_names": model_metadata["feature_names"],
            "split_strategy": "No supervised split or target exists for Cycle 001.",
            "metric_note": "Precision, recall, F1, and average precision are not defined without verified labels.",
        },
        "recommendation": "Retain the current Isolation Forest; no labeled evidence supports a performance-based replacement.",
    }


def run_cycle_baseline(
    frame: pd.DataFrame, contamination: float = 0.01, random_state: int = RANDOM_STATE
) -> dict[str, pd.DataFrame]:
    """Fit an Isolation Forest and return scores, audit tables, and alert summary."""
    if not 0 < contamination <= 0.5:
        raise ValueError("contamination must be greater than 0 and at most 0.5")
    matrix, audit, correlated = cycle_numeric_matrix(frame)
    model = Pipeline(
        [
            (
                "isolation_forest",
                IsolationForest(
                    contamination=contamination,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(matrix)
    estimator = model.named_steps["isolation_forest"]
    result = pd.DataFrame(
        {
            "timestamp": frame["Datum/Zeit"],
            "machine_id": frame["Maschinennummer"],
            "anomaly_score": -estimator.decision_function(matrix),
            "alert": estimator.predict(matrix) == -1,
        }
    )
    summary = pd.DataFrame(
        [
            {
                "contamination": contamination,
                "rows": len(result),
                "features_used": matrix.shape[1],
                "alerts": int(result["alert"].sum()),
                "alert_rate": float(result["alert"].mean()),
                "interpretation": "Unsupervised anomaly alerts; not confirmed failures",
            }
        ]
    )
    return {
        "feature_audit": audit,
        "correlated_sensors": correlated,
        "scores": result,
        "summary": summary,
    }


def ai4i_feature_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe the explicit AI4I leakage boundary without modifying labels."""
    return pd.DataFrame(
        [
            {
                "column": column,
                "status": "excluded" if column in AI4I_EXCLUDED else "feature",
                "reason": (
                    "target_or_identifier_or_failure_subtype_metadata"
                    if column in AI4I_EXCLUDED
                    else "candidate_predictor"
                ),
            }
            for column in frame.columns
        ]
    )


def ai4i_label_inconsistencies(frame: pd.DataFrame) -> pd.DataFrame:
    subtype_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    subtype_union = (frame[subtype_columns].sum(axis=1) > 0).astype(int)
    return frame.loc[frame[AI4I_TARGET] != subtype_union, [
        "UDI", "Product ID", AI4I_TARGET, *subtype_columns
    ]].assign(subtype_union=subtype_union[frame[AI4I_TARGET] != subtype_union].to_numpy())


def _ai4i_pipeline(model: Any, numeric: list[str], categorical: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def _classification_metrics(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict[str, Any], pd.DataFrame]:
    predicted = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
    metrics = {
        "model": name,
        "test_rows": len(y_test),
        "positive_rate": float(y_test.mean()),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
    }
    matrix_frame = pd.DataFrame(
        matrix, index=["actual_0", "actual_1"], columns=["predicted_0", "predicted_1"]
    ).reset_index(names="actual")
    matrix_frame.insert(0, "model", name)
    return metrics, matrix_frame


def run_ai4i_baselines(
    frame: pd.DataFrame, random_state: int = RANDOM_STATE
) -> dict[str, pd.DataFrame]:
    """Evaluate Dummy, logistic, then Random Forest on the supplied labels."""
    features = [column for column in frame.columns if column not in AI4I_EXCLUDED]
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in features if column not in numeric]
    x_train, x_test, y_train, y_test = train_test_split(
        frame[features],
        frame[AI4I_TARGET],
        test_size=0.25,
        stratify=frame[AI4I_TARGET],
        random_state=random_state,
    )
    models = [
        (
            "dummy_stratified",
            _ai4i_pipeline(DummyClassifier(strategy="stratified", random_state=random_state), numeric, categorical),
        ),
        (
            "logistic_regression_balanced",
            _ai4i_pipeline(
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state),
                numeric,
                categorical,
            ),
        ),
        (
            "random_forest_balanced",
            _ai4i_pipeline(
                RandomForestClassifier(
                    **AI4I_CURRENT_RF_PARAMS,
                    random_state=random_state,
                    n_jobs=-1,
                ),
                numeric,
                categorical,
            ),
        ),
        (
            "random_forest_balanced_leaf2",
            _ai4i_pipeline(
                RandomForestClassifier(
                    **AI4I_CANDIDATE_RF_PARAMS,
                    random_state=random_state,
                    n_jobs=-1,
                ),
                numeric,
                categorical,
            ),
        ),
    ]
    metric_rows: list[dict[str, Any]] = []
    matrices: list[pd.DataFrame] = []
    for name, model in models:
        model.fit(x_train, y_train)
        metrics, matrix = _classification_metrics(name, model, x_test, y_test)
        metric_rows.append(metrics)
        matrices.append(matrix)
    return {
        "feature_audit": ai4i_feature_audit(frame),
        "label_inconsistencies": ai4i_label_inconsistencies(frame),
        "class_balance": frame[AI4I_TARGET].value_counts().sort_index().rename_axis("label").reset_index(name="rows").assign(rate=lambda table: table["rows"] / len(frame)),
        "metrics": pd.DataFrame(metric_rows),
        "confusion_matrices": pd.concat(matrices, ignore_index=True),
    }


def run_ai4i_model_comparison(frame: pd.DataFrame) -> dict[str, Any]:
    """Compare fixed classifiers over repeated stratified holdouts without test leakage."""
    features = [column for column in frame.columns if column not in AI4I_EXCLUDED]
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in features if column not in numeric]
    candidates: list[tuple[str, str, dict[str, Any]]] = [
        ("dummy_prior", "simple_baseline", {"kind": "dummy", "parameters": {"strategy": "prior"}}),
        (
            "logistic_regression_balanced",
            "baseline",
            {"kind": "logistic", "parameters": {"class_weight": "balanced", "max_iter": 2000}},
        ),
        (
            "random_forest_balanced",
            "current",
            {"kind": "current_rf", "parameters": AI4I_CURRENT_RF_PARAMS},
        ),
        (
            "random_forest_balanced_leaf2",
            "candidate",
            {"kind": "candidate_rf", "parameters": AI4I_CANDIDATE_RF_PARAMS},
        ),
    ]
    seed_results: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in candidates}

    for seed in AI4I_VALIDATION_SEEDS:
        train_indices, test_indices = train_test_split(
            np.arange(len(frame)),
            test_size=0.25,
            stratify=frame[AI4I_TARGET],
            random_state=seed,
        )
        x_train = frame.iloc[train_indices][features]
        x_test = frame.iloc[test_indices][features]
        y_train = frame.iloc[train_indices][AI4I_TARGET]
        y_test = frame.iloc[test_indices][AI4I_TARGET]
        for name, _, specification in candidates:
            kind = specification["kind"]
            if kind == "dummy":
                estimator = DummyClassifier(strategy="prior")
            elif kind == "logistic":
                estimator = LogisticRegression(
                    class_weight="balanced", max_iter=2000, random_state=seed
                )
            elif kind == "current_rf":
                estimator = RandomForestClassifier(
                    **AI4I_CURRENT_RF_PARAMS, random_state=seed, n_jobs=-1
                )
            else:
                estimator = RandomForestClassifier(
                    **AI4I_CANDIDATE_RF_PARAMS, random_state=seed, n_jobs=-1
                )
            model = _ai4i_pipeline(estimator, numeric, categorical)
            model.fit(x_train, y_train)
            metrics, _ = _classification_metrics(name, model, x_test, y_test)
            seed_results[name].append(
                {
                    "seed": seed,
                    **{
                        metric: float(metrics[metric])
                        for metric in ("precision", "recall", "f1", "average_precision")
                    },
                }
            )

    model_rows = []
    for name, role, specification in candidates:
        results = seed_results[name]
        model_rows.append(
            {
                "model": name,
                "role": role,
                "model_version": (
                    AI4I_CURRENT_MODEL_VERSION
                    if role == "current"
                    else AI4I_CANDIDATE_MODEL_VERSION
                    if role == "candidate"
                    else "comparison-baseline-v1"
                ),
                "parameters": specification["parameters"],
                **{
                    metric: float(np.mean([result[metric] for result in results]))
                    for metric in ("precision", "recall", "f1", "average_precision")
                },
                **{
                    f"{metric}_std": float(np.std([result[metric] for result in results]))
                    for metric in ("precision", "recall", "f1", "average_precision")
                },
                "seed_results": results,
                "test_rows": int(len(frame) * 0.25),
                "positive_rate": float(frame[AI4I_TARGET].mean()),
            }
        )
    current = next(row for row in model_rows if row["role"] == "current")
    candidate = next(row for row in model_rows if row["role"] == "candidate")
    promoted = (
        candidate["average_precision"] > current["average_precision"]
        and candidate["f1"] > current["f1"]
        and all(
            candidate_result["average_precision"] > current_result["average_precision"]
            and candidate_result["f1"] > current_result["f1"]
            for candidate_result, current_result in zip(
                candidate["seed_results"], current["seed_results"]
            )
        )
    )
    return {
        "dataset": "ai4i2020",
        "target": AI4I_TARGET,
        "features": features,
        "excluded_columns": sorted(AI4I_EXCLUDED),
        "model_comparisons": model_rows,
        "metadata": {
            "split_strategy": "Repeated stratified 75/25 holdouts",
            "random_seeds": list(AI4I_VALIDATION_SEEDS),
            "metric_aggregation": "mean and population standard deviation across holdouts",
            "threshold": 0.5,
            "primary_metric": "average_precision",
            "feature_names": features,
            "excluded_columns": sorted(AI4I_EXCLUDED),
            "current_model_parameters": AI4I_CURRENT_RF_PARAMS,
            "candidate_model_parameters": AI4I_CANDIDATE_RF_PARAMS,
            "current_model_version": AI4I_CURRENT_MODEL_VERSION,
            "candidate_model_version": AI4I_CANDIDATE_MODEL_VERSION,
            "scikit_learn_version": sklearn.__version__,
        },
        "recommendation": (
            "Promote the min_samples_leaf=2 Random Forest candidate: it improves "
            "average precision and F1 over the current Random Forest on every recorded seed."
            if promoted
            else "Retain the current Random Forest; no candidate showed a consistent improvement."
        ),
    }


def save_confusion_plot(matrices: pd.DataFrame, output: Path) -> None:
    models = matrices["model"].unique()
    figure, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.5), squeeze=False)
    for axis, model in zip(axes[0], models):
        table = matrices[matrices["model"] == model].set_index("actual").loc[["actual_0", "actual_1"], ["predicted_0", "predicted_1"]]
        axis.imshow(table.to_numpy(), cmap="Blues")
        axis.set_title(model.replace("_", "\n"))
        axis.set_xticks([0, 1], ["pred 0", "pred 1"])
        axis.set_yticks([0, 1], ["actual 0", "actual 1"])
        for row in range(2):
            for column in range(2):
                axis.text(column, row, int(table.iloc[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def save_cycle_plot(scores: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(scores["timestamp"], scores["anomaly_score"], linewidth=0.7)
    flagged = scores[scores["alert"]]
    axis.scatter(flagged["timestamp"], flagged["anomaly_score"], color="crimson", s=10, label="anomaly alert")
    axis.set(title="Cycle 001 Isolation Forest anomaly scores", xlabel="Timestamp", ylabel="Higher = more anomalous")
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
