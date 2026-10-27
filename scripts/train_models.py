"""Train Phase 3 models and write pipelines plus metadata under artifacts/."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.loaders import load_ai4i2020, load_cycle001
from models.artifacts import train_cycle_artifact, train_failure_artifact

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--contamination", type=float, default=0.01)
    args = parser.parse_args()
    train_cycle_artifact(
        load_cycle001(ROOT / "docs" / "cycle_001_machine_data.csv"),
        args.output,
        contamination=args.contamination,
    )
    train_failure_artifact(load_ai4i2020(ROOT / "docs" / "ai4i2020.csv"), args.output)
    print(f"Wrote trusted model artifacts to {args.output.resolve()}")


if __name__ == "__main__":
    main()
