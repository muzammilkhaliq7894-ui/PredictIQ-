"""Reproducible model training and restricted loading of local artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.pipeline import Pipeline

from data.loaders import load_ai4i2020, load_cycle001
from data.validation import validate_ai4i2020, validate_cycle001
from models.baseline import (
    AI4I_EXCLUDED,
    AI4I_CANDIDATE_RF_PARAMS,
    AI4I_CANDIDATE_MODEL_VERSION,
    RANDOM_STATE,
    _ai4i_pipeline,
    cycle_numeric_matrix,
)

ARTIFACT_VERSION = "phase3-v1"
FAILURE_ARTIFACT_VERSION = AI4I_CANDIDATE_MODEL_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_artifact(model: Any, metadata: dict[str, Any], output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / f"{name}.joblib"
    metadata_path = output / f"{name}.json"
    joblib.dump(model, model_path)
    metadata = {**metadata, "artifact": model_path.name}
    metadata["sha256"] = _sha256(model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_cycle_artifact(frame: pd.DataFrame, output: Path, contamination: float = 0.01) -> dict[str, Any]:
    validate_cycle001(frame)
    matrix, _, _ = cycle_numeric_matrix(frame)
    model = Pipeline(
        [
            ("isolation_forest", IsolationForest(
                contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]
    )
    model.fit(matrix)
    metadata = {
        "model_name": "cycle001_anomaly",
        "model_version": ARTIFACT_VERSION,
        "model_type": "isolation_forest",
        "feature_names": list(matrix.columns),
        "contamination": contamination,
        "random_state": RANDOM_STATE,
        "limitations": [
            "Unsupervised anomaly score, not a failure probability.",
            "Cycle 001 has no verified failure labels.",
            "An alert does not establish failure, causation, or advance warning.",
        ],
    }
    _write_artifact(model, metadata, output, "cycle001_anomaly")
    return metadata


def train_failure_artifact(frame: pd.DataFrame, output: Path) -> dict[str, Any]:
    validate_ai4i2020(frame)
    features = [column for column in frame.columns if column not in AI4I_EXCLUDED]
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in features if column not in numeric]
    model = _ai4i_pipeline(
        RandomForestClassifier(
            **AI4I_CANDIDATE_RF_PARAMS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        numeric,
        categorical,
    )
    model.fit(frame[features], frame["Machine failure"])
    metadata = {
        "model_name": "ai4i_failure",
        "model_version": FAILURE_ARTIFACT_VERSION,
        "model_type": "random_forest_classifier",
        "feature_names": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "target": "Machine failure",
        "random_state": RANDOM_STATE,
        "model_parameters": AI4I_CANDIDATE_RF_PARAMS,
        "training_scope": "All supplied AI4I training rows after repeated stratified evaluation.",
        "scikit_learn_version": sklearn.__version__,
        "limitations": [
            "Probability is a model estimate for the AI4I-style input, not a guarantee.",
            "AI4I is a static benchmark and not a temporal maintenance dataset.",
            "The supplied labels contain 27 disagreements with subtype metadata; labels were not corrected.",
        ],
    }
    _write_artifact(model, metadata, output, "ai4i_failure")
    return metadata


def load_trusted_artifact(directory: Path, name: str) -> tuple[Any, dict[str, Any]]:
    """Load only a locally generated, checksum-verified artifact pair.

    The API never accepts an artifact path from a request. Callers must provide a
    configured directory containing metadata produced by ``train_models.py``.
    """
    root = directory.resolve()
    metadata_path = (root / f"{name}.json").resolve()
    if metadata_path.parent != root or not metadata_path.is_file():
        raise FileNotFoundError(f"Trusted metadata not found for {name}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("model_name") != name:
        raise ValueError(f"Invalid model metadata for {name}")
    artifact_name = metadata.get("artifact")
    if artifact_name != f"{name}.joblib":
        raise ValueError(f"Invalid artifact name in {metadata_path.name}")
    model_path = (root / artifact_name).resolve()
    if model_path.parent != root or not model_path.is_file():
        raise FileNotFoundError(f"Trusted model not found for {name}")
    if metadata.get("sha256") != _sha256(model_path):
        raise ValueError(f"Checksum mismatch for {model_path.name}")
    return joblib.load(model_path), metadata
