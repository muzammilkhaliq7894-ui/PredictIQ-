from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from data.nasa_cmapss import COLUMNS, load_nasa_dataset, load_nasa_rul, load_nasa_trajectories
from data.loaders import load_ai4i2020, load_cycle001
from models.artifacts import train_cycle_artifact, train_failure_artifact
from models.nasa_rul import _metrics

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    artifact_dir = tmp_path_factory.mktemp("nasa-api-artifacts")
    train_cycle_artifact(load_cycle001(ROOT / "docs" / "cycle_001_machine_data.csv"), artifact_dir)
    train_failure_artifact(load_ai4i2020(ROOT / "docs" / "ai4i2020.csv"), artifact_dir)
    from api.app import create_app

    return TestClient(create_app(artifact_dir))


@pytest.mark.parametrize(
    ("dataset_id", "train_rows", "test_rows", "train_engines", "test_engines"),
    [
        ("FD001", 20631, 13096, 100, 100),
        ("FD002", 53759, 33991, 260, 259),
        ("FD003", 24720, 16596, 100, 100),
        ("FD004", 61249, 41214, 249, 248),
    ],
)
def test_nasa_loader_preserves_independent_files_and_expected_schema(
    dataset_id, train_rows, test_rows, train_engines, test_engines
):
    train, test, rul = load_nasa_dataset(dataset_id)

    assert list(train.columns) == COLUMNS
    assert list(test.columns) == COLUMNS
    assert train.shape == (train_rows, 26)
    assert test.shape == (test_rows, 26)
    assert train["engine_id"].nunique() == train_engines
    assert test["engine_id"].nunique() == len(rul) == test_engines
    assert train["cycle"].dtype == test["cycle"].dtype == np.dtype("int64")
    assert len([column for column in COLUMNS if column.startswith("sensor_")]) == 21
    assert (rul >= 0).all()


def test_nasa_loader_rejects_malformed_column_counts(tmp_path):
    malformed = tmp_path / "train_FD001.txt"
    malformed.write_text("1 1 0.1 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fields; expected"):
        load_nasa_trajectories(malformed)
    malformed.write_text(" ".join(["1"] * 27), encoding="utf-8")
    with pytest.raises(ValueError, match="27 fields; expected 26"):
        load_nasa_trajectories(malformed)
    malformed.write_text(" ".join(["1"] * 26) + "\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="0 fields; expected 26"):
        load_nasa_trajectories(malformed)


def test_nasa_rul_loader_requires_one_integer_per_engine(tmp_path):
    path = tmp_path / "RUL_FD001.txt"
    path.write_text("8\nnot-a-number\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not numeric"):
        load_nasa_rul(path, expected_engines=2)
    path.write_text("8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="for 2 test engines"):
        load_nasa_rul(path, expected_engines=2)


def test_nasa_asymmetric_score_penalizes_late_estimates_more():
    score = _metrics(np.array([100.0, 100.0]), np.array([90.0, 110.0]))["nasa_score"]
    expected = (np.exp(10.0 / 13.0) - 1) + (np.exp(10.0 / 10.0) - 1)

    assert score == pytest.approx(expected)


def test_nasa_api_exposes_profile_history_and_separate_rul_workflow(client: TestClient):
    overview = client.get("/nasa/FD001/overview")
    assert overview.status_code == 200
    profile = overview.json()
    assert profile["training"]["rows"] == 20631
    assert profile["training"]["engine_count"] == 100
    assert profile["test"]["engine_count"] == 100
    assert len(profile["sensors"]) == 21
    assert len(profile["training"]["numeric_summary"]) == 26

    history = client.get("/nasa/FD001/engines/1?split=test")
    assert history.status_code == 200
    assert history.json()["rows"][0]["cycle"] == 1
    assert len(history.json()["rows"]) > 1
    assert client.get("/nasa/FD001/engines/999?split=test").status_code == 404
    assert client.get("/nasa/FD001/engines/1?split=unknown").status_code == 422

    evaluation = client.get("/nasa/FD001/rul")
    assert evaluation.status_code == 200
    result = evaluation.json()
    assert result["validation"]["engine_count"] == 20
    assert "GroupShuffleSplit runs by engine_id" in result["validation"]["split"]
    assert result["official_test"]["engine_count"] == 100
    assert result["validation"]["mae"] >= 0
    assert result["official_test"]["rmse"] >= 0
    assert result["metadata"]["validation_seeds"] == [42, 17]
    assert result["metadata"]["censor_fractions"] == [0.65, 0.75, 0.85, 0.9]
    assert result["validation"]["row_count"] == result["validation"]["engine_count"] * 4
    assert len(result["comparisons"]) == 4
    prediction = client.post("/nasa/FD001/rul/predict", json={"engine_id": 1})
    assert prediction.status_code == 200
    assert prediction.json()["engine_id"] == 1
    assert prediction.json()["last_observed_cycle"] > 0
    assert prediction.json()["predicted_rul"] >= 0
    assert prediction.json()["benchmark_rul"] >= 0
    comparisons = client.get("/model-comparisons/nasa_fd001")
    assert comparisons.status_code == 200
    assert len(comparisons.json()["model_comparisons"]) == 4
    assert "never used to choose candidates" in comparisons.json()["official_test_note"]


def test_nasa_api_does_not_expose_nasa_through_existing_dataset_or_model_routes(client):
    assert client.get("/datasets/FD001").status_code == 404
    assert client.get("/nasa/FD005/overview").status_code == 404
    assert len(client.get("/models").json()["models"]) == 2
    assert client.get("/nasa/FD001/rul").status_code == 200
    assert client.post("/nasa/FD001/rul/predict", json={"engine_id": 101}).status_code == 404
