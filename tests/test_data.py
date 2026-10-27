from pathlib import Path

import pandas as pd
import pytest

from data.loaders import load_ai4i2020, load_cycle001
from data import explorer
from data.validation import ValidationError, validate_ai4i2020, validate_cycle001


ROOT = Path(__file__).resolve().parents[1]


def test_cycle_loader_removes_empty_trailing_columns_and_parses_values():
    frame = load_cycle001(ROOT / "docs" / "cycle_001_machine_data.csv")
    assert frame.shape == (10132, 60)
    assert frame["Datum/Zeit"].dtype == "datetime64[ns]"
    assert frame["CycleTime.sv_CycleTime.dActval"].iloc[0] == pytest.approx(0.01)
    assert not any(column.startswith("Unnamed:") for column in frame.columns)


def test_cycle_validation():
    frame = load_cycle001(ROOT / "docs" / "cycle_001_machine_data.csv")
    summary = validate_cycle001(frame)
    assert summary.missing_values == 0
    assert summary.duplicate_rows == 0


def test_ai4i_schema_and_validation():
    frame = load_ai4i2020(ROOT / "docs" / "ai4i2020.csv")
    summary = validate_ai4i2020(frame)
    assert frame.shape == (10000, 14)
    assert summary.missing_values == 0
    assert int(frame["Machine failure"].sum()) == 339


def test_ai4i_rejects_non_binary_label():
    frame = load_ai4i2020(ROOT / "docs" / "ai4i2020.csv")
    frame.loc[0, "Machine failure"] = 2
    with pytest.raises(ValidationError):
        validate_ai4i2020(frame)


def test_dataset_explorer_handles_header_only_files_as_empty(tmp_path, monkeypatch):
    source = tmp_path / "empty.csv"
    source.write_text("feature_a,feature_b\n", encoding="utf-8")
    monkeypatch.setitem(
        explorer.DATASETS,
        "cycle001",
        {"name": "Cycle 001", "path": source, "read_options": {"encoding": "utf-8"}},
    )

    profile = explorer.dataset_profile("cycle001")
    analysis = explorer.analyze_dataset("cycle001", profile["row_count"])

    assert profile["row_count"] == 0
    assert profile["column_count"] == 2
    assert profile["preview"] == []
    assert all(column["dtype"] == "unknown" for column in profile["columns"])
    assert analysis["sample_size"] == 0
    assert analysis["statistics"] == []
