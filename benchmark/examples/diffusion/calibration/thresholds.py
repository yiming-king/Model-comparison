"""NPE-MCMC threshold calculation for posterior MMD, logML, and PMP errors."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = "benchmark.examples.diffusion.calibration"

from ..config import MODELS, PARAM_DIMS


DEFAULT_CALIBRATION_ROOT = (
    Path(__file__).resolve().parent.parent / "calibration_outputs"
)


GROUP_COLUMNS = (
    "generating_model",
    "generating_model_label",
    "npe_configuration",
    "training_setting",
    "summary_multiplier",
)


@dataclass(frozen=True)
class ThresholdRule:
    metric: str
    value_column: str
    convergence_scope: str
    aggregation: str


THRESHOLD_RULES = (
    ThresholdRule(
        metric="posterior_mmd",
        value_column="posterior_mmd",
        convergence_scope="matching_model",
        aggregation="matching_model_converged_datasets",
    ),
    ThresholdRule(
        metric="absolute_logml_error",
        value_column="absolute_logml_error",
        convergence_scope="matching_model",
        aggregation="matching_model_converged_datasets",
    ),
    ThresholdRule(
        metric="absolute_pmp_error",
        value_column="absolute_pmp_error",
        convergence_scope="all_candidate_models",
        aggregation="all_four_pmp_components_from_all_candidate_converged_datasets",
    ),
)


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "t", "1"}


def _selected_values(group: pd.DataFrame, rule: ThresholdRule) -> np.ndarray:
    if rule.convergence_scope == "matching_model":
        mask = group["candidate_model"].eq(group["generating_model"])
        mask &= group["converged"].map(_as_bool)
    elif rule.convergence_scope == "all_candidate_models":
        mask = group["all_candidate_models_converged"].map(_as_bool)
    else:
        raise ValueError(f"Unknown convergence scope: {rule.convergence_scope}")
    return (
        pd.to_numeric(group.loc[mask, rule.value_column], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )


def _validate_input(frame: pd.DataFrame) -> None:
    required = {
        *GROUP_COLUMNS,
        "candidate_model",
        "model_prior",
        "converged",
        "all_candidate_models_converged",
        *(rule.value_column for rule in THRESHOLD_RULES),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Threshold input is missing columns: {missing}")


def calculate_thresholds(
    frame: pd.DataFrame,
    quantile: float = 0.95,
) -> pd.DataFrame:
    """Calculate all three convergence-filtered calibration thresholds."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between zero and one")
    _validate_input(frame)

    rows = []
    for key, group in frame.groupby(
        list(GROUP_COLUMNS), sort=False, dropna=False
    ):
        metadata = dict(zip(GROUP_COLUMNS, key, strict=True))
        model_priors = {
            f"model_prior_{model}": float(
                group.loc[group["candidate_model"].eq(model), "model_prior"].iloc[0]
            )
            for model in MODELS
        }
        for rule in THRESHOLD_RULES:
            values = _selected_values(group, rule)
            if not len(values):
                raise ValueError(
                    f"No converged values for {metadata} / {rule.metric}"
                )
            rows.append(
                {
                    **metadata,
                    **model_priors,
                    "summary_dimension": PARAM_DIMS[metadata["generating_model"]]
                    * int(metadata["summary_multiplier"]),
                    "metric": rule.metric,
                    "quantile": quantile,
                    "quantile_method": "linear",
                    "threshold": float(
                        np.quantile(values, quantile, method="linear")
                    ),
                    "n_values": len(values),
                    "aggregation": rule.aggregation,
                }
            )
    return pd.DataFrame(rows)


def add_thresholds_to_metrics(
    frame: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the three group-level thresholds to every per-dataset metric row."""
    wide = thresholds.pivot(
        index=["generating_model", "npe_configuration"],
        columns="metric",
        values="threshold",
    ).rename(columns=lambda value: f"{value}_threshold")
    return frame.merge(
        wide.reset_index(),
        on=["generating_model", "npe_configuration"],
        how="left",
        validate="many_to_one",
    )


def calculate_and_save_thresholds(
    input_path: str | Path,
    thresholds_path: str | Path,
    results_path: str | Path,
    quantile: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read per-dataset NPE-MCMC metrics and write both threshold outputs."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"NPE-MCMC metrics not found: {input_path}. "
            "Run pipeline.py metrics first."
        )
    frame = pd.read_csv(input_path, keep_default_na=False)
    thresholds = calculate_thresholds(frame, quantile=quantile)
    results = add_thresholds_to_metrics(frame, thresholds)

    thresholds_path = Path(thresholds_path)
    results_path = Path(results_path)
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds.to_csv(thresholds_path, index=False)
    results.to_csv(results_path, index=False)
    return thresholds, results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate posterior-MMD, absolute-logML-error, and absolute-PMP-error "
            "thresholds from cached per-dataset NPE-MCMC metrics."
        )
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=DEFAULT_CALIBRATION_ROOT,
        help="Directory containing per_dataset_metrics.csv and receiving outputs.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional per_dataset_metrics.csv path; overrides --calibration-root.",
    )
    parser.add_argument(
        "--thresholds-output",
        type=Path,
        help="Optional thresholds.csv path; overrides --calibration-root.",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        help="Optional per_dataset_results.csv path; overrides --calibration-root.",
    )
    parser.add_argument("--quantile", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.calibration_root.resolve()
    input_path = (args.input or root / "per_dataset_metrics.csv").resolve()
    thresholds_path = (
        args.thresholds_output or root / "thresholds.csv"
    ).resolve()
    results_path = (
        args.results_output or root / "per_dataset_results.csv"
    ).resolve()
    thresholds, results = calculate_and_save_thresholds(
        input_path=input_path,
        thresholds_path=thresholds_path,
        results_path=results_path,
        quantile=args.quantile,
    )
    print(f"Input NPE-MCMC metrics: {input_path}")
    print(f"Thresholds: {thresholds_path} ({len(thresholds)} rows)")
    print(f"Metrics with thresholds: {results_path} ({len(results)} rows)")


if __name__ == "__main__":
    main()
