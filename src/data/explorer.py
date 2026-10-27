"""Bounded dataset inspection and profiling for the supplied datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PREVIEW_ROWS = 10
CHUNK_SIZE = 2000
SAMPLE_SIZE = 5000
HISTOGRAM_BINS = 20

DATASETS = {
    "cycle001": {
        "name": "Cycle 001",
        "path": ROOT / "docs" / "cycle_001_machine_data.csv",
        "read_options": {
            "sep": ";",
            "decimal": ",",
            "encoding": "utf-8-sig",
            "parse_dates": ["Datum/Zeit"],
        },
    },
    "ai4i2020": {
        "name": "AI4I 2020",
        "path": ROOT / "docs" / "ai4i2020.csv",
        "read_options": {"encoding": "utf-8-sig"},
    },
}


def _dataset(dataset_id: str) -> dict[str, Any]:
    try:
        return DATASETS[dataset_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset_id}") from exc


def _chunks(dataset: dict[str, Any]):
    return pd.read_csv(
        dataset["path"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
        **dataset["read_options"],
    )


def _clean_chunk(frame: pd.DataFrame) -> pd.DataFrame:
    empty_columns = [column for column in frame.columns if column.startswith("Unnamed:")]
    return frame.drop(columns=empty_columns) if empty_columns else frame


def dataset_profile(dataset_id: str) -> dict[str, Any]:
    dataset = _dataset(dataset_id)
    if not dataset["path"].is_file():
        raise FileNotFoundError(f"Dataset file is unavailable: {dataset['path'].name}")

    row_count = 0
    missing: dict[str, int] = {}
    dtypes: dict[str, str] = {}
    preview: list[dict[str, Any]] = []
    for frame in _chunks(dataset):
        frame = _clean_chunk(frame)
        if frame.empty:
            continue
        if not dtypes:
            dtypes = {column: str(dtype) for column, dtype in frame.dtypes.items()}
            missing = {column: 0 for column in frame.columns}
        row_count += len(frame)
        for column in frame.columns:
            missing[column] += int(frame[column].isna().sum())
        if len(preview) < PREVIEW_ROWS:
            preview.extend(
                frame.head(PREVIEW_ROWS - len(preview)).to_dict(orient="records")
            )

    if not dtypes:
        headers = _clean_chunk(
            pd.read_csv(dataset["path"], nrows=0, **dataset["read_options"])
        )
        dtypes = {column: "unknown" for column in headers.columns}
        missing = {column: 0 for column in headers.columns}
    return {
        "id": dataset_id,
        "name": dataset["name"],
        "row_count": row_count,
        "column_count": len(dtypes),
        "preview": json_records(preview),
        "columns": [
            {
                "name": column,
                "dtype": dtypes[column],
                "missing_count": missing[column],
                "missing_percent": (missing[column] / row_count * 100) if row_count else 0,
            }
            for column in dtypes
        ],
        "preview_rows": len(preview),
    }


def json_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    return json.loads(pd.DataFrame(rows).to_json(orient="records", date_format="iso"))


def analyze_dataset(dataset_id: str, row_count: int) -> dict[str, Any]:
    dataset = _dataset(dataset_id)
    if not dataset["path"].is_file():
        raise FileNotFoundError(f"Dataset file is unavailable: {dataset['path'].name}")
    sample_size = min(SAMPLE_SIZE, row_count)
    sample_positions = (
        np.linspace(0, row_count - 1, num=sample_size, dtype=np.int64)
        if sample_size
        else np.array([], dtype=np.int64)
    )
    sample_parts: list[pd.DataFrame] = []
    numeric_means: dict[str, float] = {}
    numeric_m2: dict[str, float] = {}
    numeric_counts: dict[str, int] = {}
    numeric_min: dict[str, float] = {}
    numeric_max: dict[str, float] = {}
    global_offset = 0
    for frame in _chunks(dataset):
        frame = _clean_chunk(frame)
        numeric = frame.select_dtypes(include="number")
        for column in numeric.columns:
            values = numeric[column].dropna()
            if values.empty:
                continue
            chunk_count = len(values)
            chunk_mean = float(values.mean())
            chunk_m2 = float(values.var(ddof=0) * chunk_count)
            previous_count = numeric_counts.get(column, 0)
            count = previous_count + chunk_count
            previous_mean = numeric_means.get(column, 0.0)
            delta = chunk_mean - previous_mean
            numeric_means[column] = previous_mean + delta * chunk_count / count
            numeric_m2[column] = (
                numeric_m2.get(column, 0.0)
                + chunk_m2
                + delta * delta * previous_count * chunk_count / count
            )
            numeric_counts[column] = count
            numeric_min[column] = min(numeric_min.get(column, float("inf")), float(values.min()))
            numeric_max[column] = max(numeric_max.get(column, float("-inf")), float(values.max()))

        chunk_end = global_offset + len(frame)
        selected = sample_positions[(sample_positions >= global_offset) & (sample_positions < chunk_end)]
        if len(selected):
            sample_parts.append(frame.iloc[selected - global_offset])
        global_offset = chunk_end

    sample = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    statistics = []
    distributions = []
    for column, count in numeric_counts.items():
        mean = numeric_means[column]
        statistics.append(
            {
                "column": column,
                "count": count,
                "mean": mean,
                "std": float(np.sqrt(numeric_m2[column] / (count - 1))) if count > 1 else 0.0,
                "min": numeric_min[column],
                "max": numeric_max[column],
            }
        )
        values = pd.to_numeric(sample[column], errors="coerce").dropna()
        if values.empty:
            continue
        counts, edges = np.histogram(values.to_numpy(), bins=HISTOGRAM_BINS)
        distributions.append(
            {
                "column": column,
                "bins": [
                    {
                        "range": f"{edges[index]:.4g} - {edges[index + 1]:.4g}",
                        "count": int(counts[index]),
                    }
                    for index in range(len(counts))
                ],
            }
        )

    categories = []
    for column in sample.select_dtypes(exclude="number").columns:
        if sample[column].nunique(dropna=True) > 30:
            continue
        counts = sample[column].dropna().astype(str).value_counts().head(10)
        categories.append(
            {
                "column": column,
                "values": [{"value": value, "count": int(count)} for value, count in counts.items()],
            }
        )
    return {
        "row_count": global_offset,
        "sample_size": len(sample),
        "statistics": statistics,
        "numeric_distributions": distributions,
        "categorical_counts": categories,
    }
