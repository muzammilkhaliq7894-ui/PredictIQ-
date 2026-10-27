"""Engine-level leakage-conscious baseline for NASA C-MAPSS remaining-life estimates."""

from __future__ import annotations

from functools import lru_cache
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from data.nasa_cmapss import SETTINGS, SENSORS, load_nasa_dataset, validate_dataset_id

FEATURES = ["cycle", *SETTINGS, *SENSORS]
RANDOM_STATE = 42
VALIDATION_SEEDS = (42, 17)
CENSOR_FRACTIONS = (0.65, 0.75, 0.85, 0.9)
_MODEL_CACHE_LOCK = Lock()


def _new_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=120,
        max_features=0.8,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = predicted - actual
    with np.errstate(over="raise", invalid="raise"):
        try:
            nasa_score = float(
                np.where(errors < 0, np.expm1(-errors / 13.0), np.expm1(errors / 10.0)).sum()
            )
        except FloatingPointError as exc:
            raise ValueError("NASA asymmetric score overflowed for the supplied errors") from exc
    if not np.isfinite(nasa_score):
        raise ValueError("NASA asymmetric score is not finite")
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "nasa_score": nasa_score,
    }


def _with_rul_targets(train: pd.DataFrame) -> pd.DataFrame:
    result = train.copy()
    terminal_cycle = result.groupby("engine_id")["cycle"].transform("max")
    result["rul"] = terminal_cycle - result["cycle"]
    return result


def train_and_evaluate_nasa_rul(
    train: pd.DataFrame, test: pd.DataFrame, official_rul: pd.Series
) -> tuple[dict[str, Any], RandomForestRegressor]:
    labeled = _with_rul_targets(train)
    comparison_specs = [
        ("median_rul", "simple_baseline", None, {"strategy": "median training RUL"}),
        (
            "random_forest_current",
            "current",
            "current",
            {"n_estimators": 120, "max_features": 0.8, "min_samples_leaf": 2},
        ),
        (
            "extra_trees_candidate",
            "candidate",
            ExtraTreesRegressor(
                n_estimators=160,
                max_features=0.8,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            {
                "n_estimators": 160,
                "max_features": 0.8,
                "min_samples_leaf": 2,
            },
        ),
        (
            "random_forest_300_candidate",
            "candidate",
            RandomForestRegressor(
                n_estimators=300,
                max_features=0.8,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            {
                "n_estimators": 300,
                "max_features": 0.8,
                "min_samples_leaf": 2,
            },
        ),
    ]
    validation_results: dict[str, list[dict[str, Any]]] = {
        name: [] for name, _, _, _ in comparison_specs
    }

    for seed in VALIDATION_SEEDS:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        fit_indices, validation_indices = next(
            splitter.split(labeled[FEATURES], labeled["rul"], groups=labeled["engine_id"])
        )
        fitting = labeled.iloc[fit_indices]
        held_out = labeled.iloc[validation_indices]

        validation_parts: list[pd.DataFrame] = []
        for _, trajectory in held_out.groupby("engine_id", sort=True):
            terminal_cycle = int(trajectory["cycle"].max())
            for fraction in CENSOR_FRACTIONS:
                cutoff_cycle = max(1, int(terminal_cycle * fraction))
                observed = trajectory.loc[trajectory["cycle"] <= cutoff_cycle].sort_values("cycle").tail(1).copy()
                observed["rul"] = terminal_cycle - int(observed["cycle"].iloc[0])
                validation_parts.append(observed)
        validation = pd.concat(validation_parts, ignore_index=True)
        actual = validation["rul"].to_numpy()

        for name, _, estimator, _ in comparison_specs:
            if name == "median_rul":
                predicted = np.full(len(validation), float(fitting["rul"].median()))
            else:
                model = _new_model() if estimator == "current" else estimator
                assert model is not None
                model.fit(fitting[FEATURES], fitting["rul"])
                predicted = np.maximum(model.predict(validation[FEATURES]), 0)
            validation_results[name].append(
                {"seed": seed, **_metrics(actual, predicted)}
            )

    comparisons = []
    for name, role, _, parameters in comparison_specs:
        split_results = validation_results[name]
        metrics = {
            metric: float(np.mean([result[metric] for result in split_results]))
            for metric in ("mae", "rmse", "r2", "nasa_score")
        }
        metrics.update(
            {
                f"{metric}_std": float(
                    np.std([result[metric] for result in split_results], ddof=0)
                )
                for metric in ("mae", "rmse", "r2", "nasa_score")
            }
        )
        comparisons.append(
            {
                "model": name,
                "role": role,
                "model_version": (
                    "nasa-rul-rf-v1"
                    if role == "current"
                    else "candidate-baseline-v1"
                ),
                "parameters": parameters,
                **metrics,
                "seed_results": split_results,
            }
        )
    current_comparison = next(row for row in comparisons if row["role"] == "current")

    final_model = _new_model()
    final_model.fit(labeled[FEATURES], labeled["rul"])
    test_final = test.sort_values("cycle").groupby("engine_id", sort=True).tail(1).sort_values("engine_id")
    test_prediction = np.maximum(final_model.predict(test_final[FEATURES]), 0)
    actual = official_rul.reindex(test_final["engine_id"]).to_numpy(dtype=float)
    if np.isnan(actual).any():
        raise ValueError("NASA C-MAPSS RUL labels do not match test engine IDs")
    official_test_metrics = _metrics(actual, test_prediction)
    results = {
        "validation": {
            **current_comparison,
            "engine_count": int(held_out["engine_id"].nunique()),
            "row_count": int(len(validation)),
            "split": (
                "Two GroupShuffleSplit runs by engine_id (20% held out, seeds 42 and 17); "
                "four simulated truncation points per held-out engine at 65%, 75%, 85%, and 90% of life."
            ),
        },
        "official_test": {
            **official_test_metrics,
            "engine_count": int(len(test_final)),
            "row_count": int(len(test_final)),
            "split": "Official test split; one prediction at each engine's final observed cycle",
        },
        "comparisons": comparisons,
        "recommendation": (
            "Retain the current Random Forest. Candidate regressors were compared on "
            "engine-held-out simulated truncations across two seeds; official test labels "
            "were excluded from candidate selection."
        ),
        "target_definition": "Training RUL = engine maximum training cycle - current cycle (uncapped).",
        "features": FEATURES,
        "metadata": {
            "model_version": "nasa-rul-rf-v1",
            "model_type": "RandomForestRegressor",
            "model_parameters": {
                "n_estimators": 120,
                "max_features": 0.8,
                "min_samples_leaf": 2,
                "random_state": RANDOM_STATE,
            },
            "random_state": RANDOM_STATE,
            "validation_seeds": list(VALIDATION_SEEDS),
            "validation_fraction": 0.2,
            "validation_split": "GroupShuffleSplit by engine_id",
            "censor_fractions": list(CENSOR_FRACTIONS),
            "feature_names": FEATURES,
            "target": "uncapped maximum training engine cycle minus current cycle",
            "scikit_learn_version": sklearn.__version__,
        },
        "limitations": (
            "Offline benchmark estimate only; not a guarantee of failure prevention or a live-machine warning. "
            "Official test labels are used only for evaluation, never for fitting."
        ),
    }
    return results, final_model


@lru_cache(maxsize=4)
def _cached_nasa_rul_model(dataset_id: str) -> tuple[dict[str, Any], RandomForestRegressor]:
    train, test, official_rul = load_nasa_dataset(dataset_id)
    return train_and_evaluate_nasa_rul(train, test, official_rul)


def get_nasa_rul_model(dataset_id: str) -> tuple[dict[str, Any], RandomForestRegressor]:
    dataset_id = validate_dataset_id(dataset_id)
    # lru_cache may run the wrapped function more than once for simultaneous misses.
    # Keep the expensive per-subset fit single-flight for concurrent API requests.
    with _MODEL_CACHE_LOCK:
        return _cached_nasa_rul_model(dataset_id)
