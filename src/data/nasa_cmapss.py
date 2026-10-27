"""Strict loaders and descriptive views for the NASA C-MAPSS benchmark files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NASA_ROOT = ROOT / "docs" / "nasa_cmaps"
SETTINGS = [f"operational_setting_{number}" for number in range(1, 4)]
SENSORS = [f"sensor_{number}" for number in range(1, 22)]
COLUMNS = ["engine_id", "cycle", *SETTINGS, *SENSORS]
DATASETS = ("FD001", "FD002", "FD003", "FD004")


def validate_dataset_id(dataset_id: str) -> str:
    normalized = dataset_id.upper()
    if normalized not in DATASETS:
        raise ValueError(f"Unsupported NASA C-MAPSS dataset: {dataset_id}")
    return normalized


def load_nasa_trajectories(path: str | Path) -> pd.DataFrame:
    """Read one headerless whitespace file and reject malformed rows or values."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"NASA C-MAPSS file is unavailable: {source.name}")
    line_count = 0
    try:
        with source.open(encoding="utf-8") as handle:
            for line_count, line in enumerate(handle, start=1):
                field_count = len(line.split())
                if field_count != len(COLUMNS):
                    raise ValueError(
                        f"NASA C-MAPSS file {source.name} line {line_count} has "
                        f"{field_count} fields; expected {len(COLUMNS)}"
                    )
    except UnicodeError as exc:
        raise ValueError(f"NASA C-MAPSS trajectory file {source.name} is not valid UTF-8") from exc
    if line_count == 0:
        raise ValueError(f"NASA C-MAPSS trajectory file {source.name} has no data rows")
    try:
        frame = pd.read_csv(
            source,
            sep=r"\s+",
            header=None,
            names=COLUMNS,
            dtype=np.float64,
            engine="python",
            skip_blank_lines=False,
        )
    except (pd.errors.ParserError, ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid NASA C-MAPSS trajectory file {source.name}: {exc}") from exc

    if frame.empty:
        raise ValueError(f"NASA C-MAPSS trajectory file {source.name} has no data rows")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"NASA C-MAPSS trajectory file {source.name} contains missing or non-finite values")
    if (frame[["engine_id", "cycle"]] <= 0).any().any():
        raise ValueError(f"NASA C-MAPSS trajectory file {source.name} has non-positive engine IDs or cycles")
    for column in ("engine_id", "cycle"):
        if not np.equal(frame[column], np.floor(frame[column])).all():
            raise ValueError(f"NASA C-MAPSS {column} values must be integers in {source.name}")
        frame[column] = frame[column].astype("int64")
    if frame.duplicated(["engine_id", "cycle"]).any():
        raise ValueError(f"NASA C-MAPSS file {source.name} has duplicate engine/cycle rows")

    engine_ids = sorted(frame["engine_id"].unique().tolist())
    if engine_ids != list(range(1, len(engine_ids) + 1)):
        raise ValueError(f"NASA C-MAPSS engine IDs in {source.name} must be contiguous and start at 1")
    for engine_id, trajectory in frame.groupby("engine_id", sort=False):
        cycles = trajectory["cycle"]
        if not cycles.is_monotonic_increasing:
            raise ValueError(f"NASA C-MAPSS cycles for engine {engine_id} are not increasing in {source.name}")
    return frame


def load_nasa_rul(path: str | Path, expected_engines: int) -> pd.Series:
    """Read the separate, one-integer-per-test-engine RUL target file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"NASA C-MAPSS file is unavailable: {source.name}")
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except UnicodeError as exc:
        raise ValueError(f"NASA C-MAPSS RUL file {source.name} could not be read: {exc}") from exc

    values: list[int] = []
    for line_number, line in enumerate(lines, start=1):
        tokens = line.split()
        if len(tokens) != 1:
            raise ValueError(f"NASA C-MAPSS RUL file {source.name} line {line_number} must contain one value")
        try:
            value = float(tokens[0])
        except ValueError as exc:
            raise ValueError(f"NASA C-MAPSS RUL file {source.name} line {line_number} is not numeric") from exc
        if not np.isfinite(value) or value < 0 or value != np.floor(value):
            raise ValueError(f"NASA C-MAPSS RUL file {source.name} line {line_number} must be a non-negative integer")
        values.append(int(value))
    if len(values) != expected_engines:
        raise ValueError(
            f"NASA C-MAPSS RUL file {source.name} contains {len(values)} labels "
            f"for {expected_engines} test engines"
        )
    return pd.Series(values, index=pd.RangeIndex(1, len(values) + 1, name="engine_id"), name="rul")


def load_nasa_dataset(dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    dataset_id = validate_dataset_id(dataset_id)
    train = load_nasa_trajectories(NASA_ROOT / f"train_{dataset_id}.txt")
    test = load_nasa_trajectories(NASA_ROOT / f"test_{dataset_id}.txt")
    rul = load_nasa_rul(NASA_ROOT / f"RUL_{dataset_id}.txt", test["engine_id"].nunique())
    return train, test, rul


def _numeric_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    summary = frame.describe().T
    return [
        {
            "column": column,
            "count": int(row["count"]),
            "mean": float(row["mean"]),
            "std": float(row["std"]) if pd.notna(row["std"]) else 0.0,
            "min": float(row["min"]),
            "max": float(row["max"]),
        }
        for column, row in summary.iterrows()
    ]


def dataset_overview(
    dataset_id: str, train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, Any]:
    dataset_id = validate_dataset_id(dataset_id)
    return {
        "id": dataset_id,
        "name": f"NASA C-MAPSS {dataset_id}",
        "columns": COLUMNS,
        "settings": SETTINGS,
        "sensors": SENSORS,
        "training": {
            "rows": len(train),
            "columns": len(train.columns),
            "engine_count": int(train["engine_id"].nunique()),
            "engine_ids": sorted(int(value) for value in train["engine_id"].unique()),
            "numeric_summary": _numeric_summary(train),
        },
        "test": {
            "rows": len(test),
            "columns": len(test.columns),
            "engine_count": int(test["engine_id"].nunique()),
            "engine_ids": sorted(int(value) for value in test["engine_id"].unique()),
            "numeric_summary": _numeric_summary(test),
        },
        "rul_target": "Separate RUL file has one non-negative integer label per test engine.",
    }


def engine_history(frame: pd.DataFrame, engine_id: int) -> list[dict[str, Any]]:
    trajectory = frame.loc[frame["engine_id"] == engine_id].sort_values("cycle")
    if trajectory.empty:
        raise ValueError(f"Engine {engine_id} is not present in the selected NASA C-MAPSS split")
    return trajectory.to_dict(orient="records")
