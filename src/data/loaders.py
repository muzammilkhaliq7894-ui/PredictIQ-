"""Dataset-specific loaders with explicit parsing rules and no label mixing."""

from pathlib import Path

import pandas as pd


AI4I_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def load_cycle001(path: str | Path) -> pd.DataFrame:
    """Load Cycle 001 telemetry, dropping only completely empty trailing columns."""
    frame = pd.read_csv(
        Path(path),
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        low_memory=False,
    )
    empty_columns = [column for column in frame.columns if frame[column].isna().all()]
    if empty_columns:
        frame = frame.drop(columns=empty_columns)
    frame["Datum/Zeit"] = pd.to_datetime(frame["Datum/Zeit"], errors="raise")
    return frame


def load_ai4i2020(path: str | Path) -> pd.DataFrame:
    """Load AI4I 2020 using the published comma-separated schema."""
    frame = pd.read_csv(Path(path), encoding="utf-8-sig")
    return frame
