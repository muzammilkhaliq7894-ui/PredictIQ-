"""Generate reproducible cross-pipeline model comparison evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.loaders import load_ai4i2020, load_cycle001
from data.nasa_cmapss import DATASETS as NASA_DATASETS, load_nasa_dataset
from models.artifacts import load_trusted_artifact
from models.baseline import run_ai4i_model_comparison, run_cycle_model_comparison
from models.nasa_rul import get_nasa_rul_model

ROOT = Path(__file__).resolve().parents[1]


def build_comparison_report() -> dict[str, Any]:
    anomaly_model, anomaly_metadata = load_trusted_artifact(
        ROOT / "artifacts", "cycle001_anomaly"
    )
    cycle = run_cycle_model_comparison(
        load_cycle001(ROOT / "docs" / "cycle_001_machine_data.csv"),
        anomaly_model,
        anomaly_metadata,
    )
    ai4i = run_ai4i_model_comparison(
        load_ai4i2020(ROOT / "docs" / "ai4i2020.csv")
    )
    nasa: dict[str, Any] = {}
    for dataset_id in NASA_DATASETS:
        train, test, official_rul = load_nasa_dataset(dataset_id)
        metrics, _ = get_nasa_rul_model(dataset_id)
        nasa[f"nasa_{dataset_id.lower()}"] = {
            "dataset": dataset_id,
            "train_rows": len(train),
            "train_engines": int(train["engine_id"].nunique()),
            "test_rows": len(test),
            "test_engines": int(test["engine_id"].nunique()),
            "official_rul_labels": len(official_rul),
            **metrics,
        }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipelines": {
            "cycle001": cycle,
            "ai4i2020": ai4i,
            **nasa,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "model_comparisons.json",
    )
    args = parser.parse_args()
    report = build_comparison_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote model comparison report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
