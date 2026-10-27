"""Schema and data-quality validation for the two independent EDA inputs."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .loaders import AI4I_COLUMNS


class ValidationError(ValueError):
    """Raised when a dataset violates its required structural contract."""


@dataclass(frozen=True)
class ValidationSummary:
    dataset: str
    rows: int
    columns: int
    missing_values: int
    duplicate_rows: int


def _require_columns(frame: pd.DataFrame, required: list[str], dataset: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValidationError(f"{dataset} is missing required columns: {missing}")


def validate_cycle001(frame: pd.DataFrame) -> ValidationSummary:
    required = ["Datum/Zeit", "Maschinennummer"]
    _require_columns(frame, required, "Cycle 001")
    if not pd.api.types.is_datetime64_any_dtype(frame["Datum/Zeit"]):
        raise ValidationError("Cycle 001 Datum/Zeit must be datetime64")
    if frame["Datum/Zeit"].isna().any():
        raise ValidationError("Cycle 001 contains invalid timestamps")
    if frame["Maschinennummer"].isna().any():
        raise ValidationError("Cycle 001 contains missing machine IDs")
    timestamps = frame["Datum/Zeit"]
    if timestamps.duplicated().any() or (timestamps.diff().dropna() <= pd.Timedelta(0)).any():
        raise ValidationError("Cycle 001 timestamps must be unique and increasing")
    if frame.isna().any().any():
        raise ValidationError("Cycle 001 contains missing values after loading")
    if frame.duplicated().any():
        raise ValidationError("Cycle 001 contains duplicate rows")
    return ValidationSummary(
        "cycle_001", len(frame), len(frame.columns), int(frame.isna().sum().sum()), int(frame.duplicated().sum())
    )


def validate_ai4i2020(frame: pd.DataFrame) -> ValidationSummary:
    _require_columns(frame, AI4I_COLUMNS, "AI4I 2020")
    if list(frame.columns) != AI4I_COLUMNS:
        raise ValidationError("AI4I 2020 columns do not match the expected schema and order")
    if frame.isna().any().any():
        raise ValidationError("AI4I 2020 contains missing values")
    if frame.duplicated().any() or frame["UDI"].duplicated().any():
        raise ValidationError("AI4I 2020 contains duplicate rows or UDI values")
    if not set(frame["Type"].unique()).issubset({"L", "M", "H"}):
        raise ValidationError("AI4I 2020 Type contains unexpected values")
    binary_columns = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
    for column in binary_columns:
        if not set(frame[column].unique()).issubset({0, 1}):
            raise ValidationError(f"AI4I 2020 {column} must be binary")
    if not np.issubdtype(frame["Machine failure"].dtype, np.integer):
        raise ValidationError("AI4I 2020 Machine failure must be integer")
    return ValidationSummary(
        "ai4i2020", len(frame), len(frame.columns), int(frame.isna().sum().sum()), int(frame.duplicated().sum())
    )
