"""Validated FastAPI surface for the independent Phase 3 models."""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from models.artifacts import load_trusted_artifact
from data.explorer import analyze_dataset, dataset_profile
from data.loaders import load_ai4i2020, load_cycle001
from data.nasa_cmapss import (
    engine_history,
    load_nasa_dataset,
    dataset_overview as nasa_dataset_overview,
    validate_dataset_id,
)
from models.baseline import (
    cycle_numeric_matrix,
    run_ai4i_model_comparison,
    run_cycle_model_comparison,
)
from models.nasa_rul import FEATURES as NASA_RUL_FEATURES, get_nasa_rul_model

ROOT = Path(__file__).resolve().parents[2]
LIMITATIONS = {
    "anomaly": "Anomaly score is not a failure probability; Cycle 001 has no verified failure labels.",
    "failure": "Failure probability is a model estimate for AI4I-style data, not advance warning or a guarantee.",
}


class AnomalyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, Any]
    timestamp: str | None = None
    machine_id: int | None = None


class FailureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    Type: str
    air_temperature: Any
    process_temperature: Any
    rotational_speed: Any
    torque: Any
    tool_wear: Any


class NasaRulPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    engine_id: int


def _numeric_features(values: dict[str, Any], expected: list[str]) -> dict[str, float]:
    missing = [name for name in expected if name not in values]
    extra = [name for name in values if name not in expected]
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing features: {missing}")
        if extra:
            detail.append(f"unexpected features: {extra}")
        raise HTTPException(status_code=422, detail="; ".join(detail))
    result: dict[str, float] = {}
    for name in expected:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise HTTPException(status_code=422, detail=f"{name} must be a finite number")
        result[name] = float(value)
    return result


def create_app(artifact_dir: Path | None = None) -> FastAPI:
    directory = (artifact_dir or ROOT / "artifacts").resolve()
    try:
        anomaly_model, anomaly_meta = load_trusted_artifact(directory, "cycle001_anomaly")
        failure_model, failure_meta = load_trusted_artifact(directory, "ai4i_failure")
    except (FileNotFoundError, ValueError, OSError, TypeError) as exc:
        raise RuntimeError(
            f"Unable to load trusted Phase 3 artifacts from {directory}: {exc}"
        ) from exc

    app = FastAPI(title="Predictive Maintenance API", version="phase3")

    @lru_cache(maxsize=4)
    def nasa_frames(dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        return load_nasa_dataset(dataset_id)

    @lru_cache(maxsize=1)
    def ai4i_comparisons() -> dict[str, Any]:
        frame = load_ai4i2020(ROOT / "docs" / "ai4i2020.csv")
        return run_ai4i_model_comparison(frame)

    cors_origins = [
        origin.strip()
        for origin in os.getenv(
            "PREDICTIVE_MAINTENANCE_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://127.0.0.1:5174,http://localhost:5174",
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_version": anomaly_meta["model_version"]}

    @app.get("/models")
    def models() -> dict[str, list[dict[str, Any]]]:
        return {"models": [
            {
                "name": anomaly_meta["model_name"],
                "version": anomaly_meta["model_version"],
                "type": anomaly_meta["model_type"],
                "feature_names": anomaly_meta["feature_names"],
                "limitations": anomaly_meta["limitations"],
            },
            {
                "name": failure_meta["model_name"],
                "version": failure_meta["model_version"],
                "type": failure_meta["model_type"],
                "feature_names": failure_meta["feature_names"],
                "limitations": failure_meta["limitations"],
            },
        ]}

    @app.get("/evaluation")
    def evaluation() -> dict[str, Any]:
        report_dir = ROOT / "reports" / "baseline"
        try:
            metrics = pd.read_csv(report_dir / "ai4i2020_metrics.csv")
            confusion = pd.read_csv(report_dir / "ai4i2020_confusion_matrices.csv")
            cycle = pd.read_csv(report_dir / "cycle001_summary.csv")
            class_balance = pd.read_csv(report_dir / "ai4i2020_class_balance.csv")
            cycle_flagged = pd.read_csv(report_dir / "cycle001_flagged_timestamps.csv")
        except (OSError, pd.errors.ParserError) as exc:
            raise HTTPException(status_code=503, detail=f"Saved evaluation reports are unavailable: {exc}") from exc
        return {
            "metrics": json.loads(metrics.to_json(orient="records", double_precision=15)),
            "confusion_matrices": json.loads(confusion.to_json(orient="records", double_precision=15)),
            "cycle_summary": json.loads(cycle.to_json(orient="records", double_precision=15))[0],
            "class_balance": json.loads(class_balance.to_json(orient="records", double_precision=15)),
            "cycle_flagged": json.loads(cycle_flagged.head(20).to_json(orient="records", double_precision=15)),
            "source": "reports/baseline",
        }

    @app.get("/model-comparisons/{dataset_id}")
    def model_comparisons(dataset_id: str) -> dict[str, Any]:
        if dataset_id == "ai4i2020":
            try:
                return ai4i_comparisons()
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                raise HTTPException(status_code=503, detail=f"AI4I model comparisons are unavailable: {exc}") from exc

        if dataset_id == "cycle001":
            try:
                frame = load_cycle001(ROOT / "docs" / "cycle_001_machine_data.csv")
                return run_cycle_model_comparison(frame, anomaly_model, anomaly_meta)
            except HTTPException:
                raise
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                raise HTTPException(status_code=503, detail=f"Cycle 001 comparisons are unavailable: {exc}") from exc

        if dataset_id.startswith("nasa_"):
            try:
                nasa_id = validate_dataset_id(dataset_id.removeprefix("nasa_"))
                metrics, _ = get_nasa_rul_model(nasa_id)
                return {
                    "dataset": dataset_id,
                    "model_type": "remaining_useful_life_regression",
                    "model_comparisons": metrics["comparisons"],
                    "metadata": metrics["metadata"],
                    "recommendation": metrics["recommendation"],
                    "official_test_current": metrics["official_test"],
                    "official_test_note": "Official test labels are reported for the current model only and never used to choose candidates.",
                    "limitations": metrics["limitations"],
                }
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

        raise HTTPException(
            status_code=404,
            detail="Unsupported comparison dataset. Choose cycle001, ai4i2020, or nasa_fd001 through nasa_fd004.",
        )

    @app.get("/datasets/{dataset_id}")
    def explore_dataset(dataset_id: str) -> dict[str, Any]:
        try:
            return dataset_profile(dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            raise HTTPException(status_code=503, detail=f"Dataset could not be read: {exc}") from exc

    @app.get("/datasets/{dataset_id}/analysis")
    def explore_dataset_analysis(dataset_id: str) -> dict[str, Any]:
        try:
            profile = dataset_profile(dataset_id)
            return analyze_dataset(dataset_id, profile["row_count"])
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            raise HTTPException(status_code=503, detail=f"Dataset analysis failed: {exc}") from exc

    @app.get("/datasets/{dataset_id}/telemetry")
    def cycle001_telemetry(
        dataset_id: str, limit: int = Query(default=20, ge=1, le=20)
    ) -> dict[str, Any]:
        if dataset_id != "cycle001":
            raise HTTPException(
                status_code=400,
                detail="Built-in anomaly telemetry is available only for Cycle 001.",
            )
        try:
            frame = load_cycle001(ROOT / "docs" / "cycle_001_machine_data.csv")
            matrix, _, _ = cycle_numeric_matrix(frame)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise HTTPException(status_code=503, detail=f"Cycle 001 telemetry is unavailable: {exc}") from exc

        expected = anomaly_meta["feature_names"]
        if list(matrix.columns) != expected:
            raise HTTPException(
                status_code=503,
                detail="Cycle 001 dataset features do not match the loaded anomaly model schema.",
            )
        complete = matrix.replace([float("inf"), float("-inf")], pd.NA).dropna()
        selected = complete.head(limit)
        return {
            "dataset": "cycle001",
            "rows": [
                {
                    "timestamp": frame.loc[index, "Datum/Zeit"].isoformat(),
                    "values": row.to_dict(),
                }
                for index, row in selected.iterrows()
            ],
        }

    @app.get("/nasa/{dataset_id}/overview")
    def nasa_overview(dataset_id: str) -> dict[str, Any]:
        try:
            dataset_id = validate_dataset_id(dataset_id)
            train, test, _ = nasa_frames(dataset_id)
            return nasa_dataset_overview(dataset_id, train, test)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            raise HTTPException(status_code=503, detail=f"NASA C-MAPSS data could not be read: {exc}") from exc

    @app.get("/nasa/{dataset_id}/engines/{engine_id}")
    def nasa_engine_history(
        dataset_id: str,
        engine_id: int,
        split: str = Query(default="test"),
    ) -> dict[str, Any]:
        if split not in {"train", "test"}:
            raise HTTPException(status_code=422, detail="split must be either train or test")
        try:
            dataset_id = validate_dataset_id(dataset_id)
            train, test, _ = nasa_frames(dataset_id)
            frame = train if split == "train" else test
            return {
                "dataset": dataset_id,
                "split": split,
                "engine_id": engine_id,
                "columns": NASA_RUL_FEATURES,
                "rows": engine_history(frame, engine_id),
            }
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            raise HTTPException(status_code=503, detail=f"NASA C-MAPSS engine history is unavailable: {exc}") from exc

    @app.get("/nasa/{dataset_id}/rul")
    def nasa_rul_evaluation(dataset_id: str) -> dict[str, Any]:
        try:
            dataset_id = validate_dataset_id(dataset_id)
            metrics, _ = get_nasa_rul_model(dataset_id)
            return {"dataset": dataset_id, **metrics}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            raise HTTPException(status_code=503, detail=f"NASA RUL evaluation is unavailable: {exc}") from exc

    @app.post("/nasa/{dataset_id}/rul/predict")
    def nasa_rul_predict(
        dataset_id: str, request: NasaRulPredictionRequest
    ) -> dict[str, Any]:
        try:
            dataset_id = validate_dataset_id(dataset_id)
            _, test, official_rul = nasa_frames(dataset_id)
            metrics, model = get_nasa_rul_model(dataset_id)
            if request.engine_id not in set(test["engine_id"].unique()):
                raise HTTPException(
                    status_code=404,
                    detail="RUL prediction supports test engines only; select an engine from the NASA test split.",
                )
            latest = test.loc[test["engine_id"] == request.engine_id].sort_values("cycle").tail(1)
            prediction = max(0.0, float(model.predict(latest[NASA_RUL_FEATURES])[0]))
            return {
                "dataset": dataset_id,
                "engine_id": request.engine_id,
                "last_observed_cycle": int(latest["cycle"].iloc[0]),
                "predicted_rul": prediction,
                "benchmark_rul": int(official_rul.loc[request.engine_id]),
                "evaluation": metrics["official_test"],
                "limitations": metrics["limitations"],
            }
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            raise HTTPException(status_code=503, detail=f"NASA RUL prediction is unavailable: {exc}") from exc

    @app.post("/anomaly/score")
    def anomaly_score(request: AnomalyRequest) -> dict[str, Any]:
        values = _numeric_features(request.features, anomaly_meta["feature_names"])
        frame = pd.DataFrame([values], columns=anomaly_meta["feature_names"])
        estimator = anomaly_model.named_steps.get("isolation_forest", anomaly_model)
        score = float(-estimator.decision_function(frame)[0])
        alert = bool(estimator.predict(frame)[0] == -1)
        return {
            "anomaly_score": score,
            "alert": alert,
            "model_version": anomaly_meta["model_version"],
            "limitations": LIMITATIONS["anomaly"],
        }

    @app.post("/failure/predict")
    def failure_predict(request: FailureRequest) -> dict[str, Any]:
        values = {
            "Type": request.Type,
            "Air temperature [K]": request.air_temperature,
            "Process temperature [K]": request.process_temperature,
            "Rotational speed [rpm]": request.rotational_speed,
            "Torque [Nm]": request.torque,
            "Tool wear [min]": request.tool_wear,
        }
        if request.Type not in {"L", "M", "H"}:
            raise HTTPException(status_code=422, detail="Type must be one of L, M, or H")
        numeric = _numeric_features(
            {key: value for key, value in values.items() if key != "Type"},
            failure_meta["numeric_features"],
        )
        row = pd.DataFrame([{**numeric, "Type": request.Type}], columns=failure_meta["feature_names"])
        probability = float(failure_model.predict_proba(row)[0, 1])
        return {
            "failure_probability": probability,
            "predicted_failure": bool(failure_model.predict(row)[0]),
            "model_version": failure_meta["model_version"],
            "limitations": LIMITATIONS["failure"],
        }

    return app


app = None
