"""Reproducible summary tables and plots for Phase 1 EDA."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def column_profile(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number")
    rows = []
    for column in frame.columns:
        series = frame[column]
        item = {
            "dataset": dataset,
            "column": column,
            "dtype": str(series.dtype),
            "rows": len(series),
            "missing_count": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean() * 100),
            "unique_count": int(series.nunique(dropna=True)),
            "constant": bool(series.nunique(dropna=True) <= 1),
        }
        if column in numeric:
            item.update(
                {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                }
            )
        else:
            item.update({"min": None, "max": None, "mean": None, "std": None})
        rows.append(item)
    return pd.DataFrame(rows)


def numeric_summary(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    summary = frame.select_dtypes(include="number").describe().T.reset_index()
    summary = summary.rename(columns={"index": "column"})
    summary.insert(0, "dataset", dataset)
    return summary


def save_cycle_plots(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    timestamp = frame["Datum/Zeit"]
    selected = [
        "CycleTime.sv_CycleTime.dActval",
        "Heating.sv_Zone15.rActualTemp",
        "InjectionUnit1.sv_rActTorque",
    ]
    for column in selected:
        if column not in frame:
            continue
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.plot(timestamp, frame[column], linewidth=0.8)
        axis.set_title(column)
        axis.set_xlabel("Timestamp")
        axis.set_ylabel("Value")
        figure.autofmt_xdate()
        figure.tight_layout()
        path = output_dir / f"cycle001_{column.replace('.', '_').replace('[', '').replace(']', '')}.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        outputs.append(path)
    return outputs


def save_ai4i_plots(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    figure, axis = plt.subplots(figsize=(6, 4))
    frame["Machine failure"].value_counts().sort_index().plot.bar(ax=axis)
    axis.set_title("AI4I 2020 machine failure counts")
    axis.set_xlabel("Machine failure")
    axis.set_ylabel("Rows")
    figure.tight_layout()
    path = output_dir / "ai4i2020_machine_failure_counts.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    outputs.append(path)

    figure, axis = plt.subplots(figsize=(6, 4))
    frame.groupby("Type")["Machine failure"].mean().sort_index().plot.bar(ax=axis)
    axis.set_title("Failure rate by product type")
    axis.set_xlabel("Type")
    axis.set_ylabel("Failure rate")
    figure.tight_layout()
    path = output_dir / "ai4i2020_failure_rate_by_type.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    outputs.append(path)
    return outputs
