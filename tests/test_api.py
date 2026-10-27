import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from data.loaders import load_ai4i2020, load_cycle001
from models.artifacts import load_trusted_artifact, train_cycle_artifact, train_failure_artifact
from models.baseline import cycle_numeric_matrix

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    artifact_dir = tmp_path_factory.mktemp("artifacts")
    train_cycle_artifact(load_cycle001(ROOT / "docs" / "cycle_001_machine_data.csv"), artifact_dir)
    train_failure_artifact(load_ai4i2020(ROOT / "docs" / "ai4i2020.csv"), artifact_dir)
    from api.app import create_app

    return TestClient(create_app(artifact_dir))


def test_health_and_models(client):
    assert client.get("/health").json()["status"] == "ok"
    models = client.get("/models").json()["models"]
    assert len(models) == 2
    failure = next(model for model in models if model["name"] == "ai4i_failure")
    assert failure["version"] == "phase4-v2"


def test_dashboard_origin_is_allowed_for_api_requests(client):
    response = client.options(
        "/failure/predict",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_dashboard_fallback_origin_is_allowed_for_api_requests(client):
    response = client.options(
        "/failure/predict",
        headers={
            "Origin": "http://127.0.0.1:5174",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5174"


def test_evaluation_serves_saved_phase2_reports(client):
    response = client.get("/evaluation")
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "reports/baseline"
    assert body["metrics"][2]["model"] == "random_forest_balanced"
    assert body["metrics"][2]["average_precision"] == pytest.approx(0.7523515427)
    assert body["confusion_matrices"][5]["predicted_1"] == 38
    assert body["cycle_summary"]["alerts"] == 102
    assert body["class_balance"][1]["rows"] == 339
    assert len(body["cycle_flagged"]) == 20
    assert body["cycle_flagged"][0]["alert"] is True


def test_model_comparisons_report_ai4i_candidate_and_cycle_unlabeled_limitations(client):
    ai4i = client.get("/model-comparisons/ai4i2020")
    assert ai4i.status_code == 200
    result = ai4i.json()
    assert result["metadata"]["random_seeds"] == [42, 17, 91]
    roles = {row["role"]: row for row in result["model_comparisons"]}
    assert roles["candidate"]["average_precision"] > roles["current"]["average_precision"]
    assert roles["candidate"]["f1"] > roles["current"]["f1"]

    cycle = client.get("/model-comparisons/cycle001")
    assert cycle.status_code == 200
    cycle_result = cycle.json()
    assert cycle_result["model_comparisons"][0]["alerts"] == 102
    assert "not defined without verified labels" in cycle_result["metadata"]["metric_note"]
    assert "precision" not in cycle_result["model_comparisons"][0]


def test_dataset_explorer_returns_bounded_cycle_profile(client):
    response = client.get("/datasets/cycle001")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Cycle 001"
    assert body["row_count"] == 10132
    assert body["column_count"] == 60
    assert len(body["preview"]) == body["preview_rows"] == 10
    assert len(body["columns"]) == body["column_count"]
    assert body["columns"][0]["name"] == "Datum/Zeit"
    assert all(column["missing_count"] == 0 for column in body["columns"])


def test_dataset_explorer_profiles_ai4i_schema_and_missing_counts(client):
    response = client.get("/datasets/ai4i2020")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "AI4I 2020"
    assert body["row_count"] == 10000
    assert body["column_count"] == 14
    assert len(body["preview"]) == body["preview_rows"] == 10
    assert all(column["missing_count"] == 0 for column in body["columns"])


def test_dataset_explorer_analysis_profiles_real_ai4i_rows(client):
    response = client.get("/datasets/ai4i2020/analysis")
    assert response.status_code == 200
    body = response.json()
    assert body["row_count"] == 10000
    assert body["sample_size"] == 5000
    failure_stats = next(row for row in body["statistics"] if row["column"] == "Machine failure")
    assert failure_stats["mean"] == pytest.approx(0.0339)
    assert failure_stats["min"] == 0
    assert failure_stats["max"] == 1
    assert body["numeric_distributions"]
    assert body["categorical_counts"]
    failure_histogram = next(
        chart for chart in body["numeric_distributions"] if chart["column"] == "Machine failure"
    )
    assert sum(item["count"] for item in failure_histogram["bins"]) == body["sample_size"]


def test_dataset_explorer_rejects_unsupported_dataset(client):
    assert client.get("/datasets/nasa").status_code == 404
    assert client.get("/datasets/nasa/analysis").status_code == 404


def test_builtin_cycle_telemetry_matches_anomaly_schema_and_scores(client):
    models = client.get("/models").json()["models"]
    anomaly = next(model for model in models if model["name"] == "cycle001_anomaly")

    response = client.get("/datasets/cycle001/telemetry")

    assert response.status_code == 200
    body = response.json()
    assert body["dataset"] == "cycle001"
    assert len(body["rows"]) == 20
    first = body["rows"][0]
    assert list(first["values"]) == anomaly["feature_names"]
    scored = client.post("/anomaly/score", json={"features": first["values"]})
    assert scored.status_code == 200
    assert isinstance(scored.json()["anomaly_score"], float)


def test_builtin_ai4i_telemetry_is_not_sent_to_cycle_anomaly_model(client):
    response = client.get("/datasets/ai4i2020/telemetry")
    assert response.status_code == 400
    assert "only for Cycle 001" in response.json()["detail"]


def test_ai4i_numeric_predictors_are_rejected_by_cycle_anomaly_schema(client):
    ai4i = load_ai4i2020(ROOT / "docs" / "ai4i2020.csv")
    features = ai4i.loc[
        0,
        [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ],
    ].to_dict()
    response = client.post("/anomaly/score", json={"features": features})
    assert response.status_code == 422
    assert "missing features" in response.json()["detail"]


def test_anomaly_score_requires_exact_numeric_features(client):
    response = client.post("/anomaly/score", json={"features": {"wrong": 1}})
    assert response.status_code == 422
    assert "missing features" in response.json()["detail"]


def test_anomaly_score_returns_score_separately(client):
    frame = load_cycle001(ROOT / "docs" / "cycle_001_machine_data.csv")
    matrix, _, _ = cycle_numeric_matrix(frame)
    response = client.post("/anomaly/score", json={"features": matrix.iloc[0].to_dict()})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["anomaly_score"], float)
    assert isinstance(body["alert"], bool)
    assert body["model_version"] == "phase3-v1"
    assert "failure_probability" not in body


def test_anomaly_score_rejects_non_numeric_features(client):
    frame = load_cycle001(ROOT / "docs" / "cycle_001_machine_data.csv")
    matrix, _, _ = cycle_numeric_matrix(frame)
    features = matrix.iloc[0].to_dict()
    features[next(iter(features))] = "not-a-number"
    response = client.post("/anomaly/score", json={"features": features})
    assert response.status_code == 422


def test_failure_prediction_separates_probability_and_limitations(client):
    response = client.post(
        "/failure/predict",
        json={
            "Type": "L",
            "air_temperature": 298.1,
            "process_temperature": 308.6,
            "rotational_speed": 1551,
            "torque": 42.8,
            "tool_wear": 0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["failure_probability"] <= 1
    assert "anomaly_score" not in body
    assert "limitations" in body


def test_invalid_failure_input_is_not_substituted(client):
    response = client.post(
        "/failure/predict",
        json={
            "Type": "X",
            "air_temperature": 298.1,
            "process_temperature": 308.6,
            "rotational_speed": 1551,
            "torque": None,
            "tool_wear": 0,
        },
    )
    assert response.status_code == 422


def test_missing_model_artifacts_fail_startup_clearly(tmp_path):
    from api.app import create_app

    with pytest.raises(RuntimeError, match="Unable to load trusted Phase 3 artifacts"):
        create_app(tmp_path)


def test_model_artifact_checksum_mismatch_is_rejected(tmp_path):
    (tmp_path / "cycle001_anomaly.joblib").write_bytes(b"modified artifact")
    (tmp_path / "cycle001_anomaly.json").write_text(
        json.dumps({
            "model_name": "cycle001_anomaly",
            "artifact": "cycle001_anomaly.joblib",
            "sha256": "0" * 64,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_trusted_artifact(tmp_path, "cycle001_anomaly")
