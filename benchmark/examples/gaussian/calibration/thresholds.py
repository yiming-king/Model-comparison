"""NPE-analytical thresholds for the Gaussian case study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CALIBRATION_ROOT, ASSUMED_MODELS


MIN_IMPORTANCE_ESS_RATIO = 0.20
DEFAULT_POSTERIOR_MMD_QUANTILE = 0.95
DEFAULT_SIGNED_ERROR_COVERAGE = 0.90


GROUP_COLUMNS = (
    "configuration",
    "network_tag",
    "summary_label",
    "summary_dim",
    "generating_model",
    "num_dims",
    "num_obs",
    "num_posterior_samples",
    "logml_method",
)


@dataclass(frozen=True)
class ThresholdRule:
    metric: str
    value_column: str
    median_column: str
    selection: str
    aggregation: str
    interval: str


THRESHOLD_RULES = (
    ThresholdRule(
        metric="posterior_mmd",
        value_column="posterior_mmd",
        median_column="posterior_mmd",
        selection="matching_model",
        aggregation="matching_model_well_specified_datasets",
        interval="nonnegative_upper_quantile",
    ),
    ThresholdRule(
        metric="signed_logml_error",
        value_column="signed_logml_error",
        median_column="signed_logml_error",
        selection="matching_model",
        aggregation="matching_model_well_specified_datasets",
        interval="central_reference_interval",
    ),
    ThresholdRule(
        metric="signed_pmp_error",
        value_column="signed_pmp_error",
        median_column="signed_pmp_error",
        selection="all_candidate_models",
        aggregation="all_four_pmp_components_from_well_specified_datasets",
        interval="central_reference_interval",
    ),
)


def _validate_input(frame: pd.DataFrame) -> None:
    required = {
        *GROUP_COLUMNS,
        "candidate_model",
        "dataset_id",
        "importance_ess",
        "num_npe_logml_draws",
        *(rule.value_column for rule in THRESHOLD_RULES),
        *(rule.median_column for rule in THRESHOLD_RULES),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Threshold input is missing columns: {missing}")


def _selected_values(
    group: pd.DataFrame,
    rule: ThresholdRule,
    column: str | None = None,
    *,
    apply_importance_filter: bool = True,
) -> np.ndarray:
    importance_ess = pd.to_numeric(group["importance_ess"], errors="coerce")
    num_draws = pd.to_numeric(group["num_npe_logml_draws"], errors="coerce")
    importance_ok = (
        importance_ess.notna()
        & num_draws.gt(0.0)
        & importance_ess.div(num_draws).ge(MIN_IMPORTANCE_ESS_RATIO)
    )
    if rule.selection == "matching_model":
        mask = group["candidate_model"].eq(group["generating_model"])
    elif rule.selection == "all_candidate_models":
        mask = pd.Series(True, index=group.index)
    else:
        raise ValueError(f"Unknown threshold selection: {rule.selection}")
    if apply_importance_filter:
        mask &= importance_ok
    return (
        pd.to_numeric(group.loc[mask, column or rule.value_column], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )


def calculate_thresholds(
    frame: pd.DataFrame,
    quantile: float = DEFAULT_POSTERIOR_MMD_QUANTILE,
    signed_error_coverage: float = DEFAULT_SIGNED_ERROR_COVERAGE,
) -> pd.DataFrame:
    """Calculate raw reference thresholds without MCMC convergence filters.

    ``quantile`` is the one-sided upper quantile for non-negative posterior MMD.
    ``signed_error_coverage`` is the central coverage for signed logML/PMP errors.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between zero and one")
    if not 0.0 < signed_error_coverage < 1.0:
        raise ValueError("signed_error_coverage must be between zero and one")
    _validate_input(frame)
    unknown = sorted(set(frame["generating_model"]).difference(ASSUMED_MODELS))
    if unknown:
        raise ValueError(f"Unknown generating models: {unknown}")

    rows = []
    for key, group in frame.groupby(list(GROUP_COLUMNS), sort=False, dropna=False):
        metadata = dict(zip(GROUP_COLUMNS, key, strict=True))
        for rule in THRESHOLD_RULES:
            values_before_ess_filter = _selected_values(
                group, rule, apply_importance_filter=False
            )
            values = _selected_values(group, rule)
            medians = _selected_values(group, rule, rule.median_column)
            if not len(values) or not len(medians):
                raise ValueError(
                    f"No calibration values for {metadata} / {rule.metric}"
                )
            if rule.interval == "nonnegative_upper_quantile":
                coverage = quantile
                lower_quantile = 0.0
                upper_quantile = coverage
                lower_threshold = 0.0
            elif rule.interval == "central_reference_interval":
                coverage = signed_error_coverage
                tail_probability = (1.0 - coverage) / 2.0
                lower_quantile = tail_probability
                upper_quantile = 1.0 - tail_probability
                lower_threshold = float(
                    np.quantile(values, lower_quantile, method="linear")
                )
            else:
                raise ValueError(f"Unknown interval rule: {rule.interval}")
            upper_threshold = float(
                np.quantile(values, upper_quantile, method="linear")
            )
            rows.append(
                {
                    **metadata,
                    "metric": rule.metric,
                    "quantile": float(coverage),
                    "lower_quantile": float(lower_quantile),
                    "upper_quantile": float(upper_quantile),
                    "quantile_method": "linear",
                    "lower_threshold": lower_threshold,
                    "threshold": upper_threshold,
                    "degenerate_interval": bool(
                        lower_threshold == upper_threshold
                    ),
                    "median": float(np.median(medians)),
                    "n_values": int(len(values)),
                    "n_values_before_ess_filter": int(
                        len(values_before_ess_filter)
                    ),
                    "n_dropped_low_importance_ess": int(
                        len(values_before_ess_filter) - len(values)
                    ),
                    "aggregation": rule.aggregation,
                    "gold_standard": "analytical",
                    "min_importance_ess_ratio": MIN_IMPORTANCE_ESS_RATIO,
                }
            )
    return pd.DataFrame(rows)


def add_thresholds_to_metrics(
    frame: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """Attach one model/configuration threshold set to every metric row."""
    join_columns = list(GROUP_COLUMNS)
    pivot_values = ["lower_threshold", "threshold", "median"]
    if "degenerate_interval" in thresholds:
        pivot_values.append("degenerate_interval")
    wide = thresholds.pivot(
        index=join_columns,
        columns="metric",
        values=pivot_values,
    )
    wide.columns = [f"{metric}_{statistic}" for statistic, metric in wide.columns]
    return frame.merge(
        wide.reset_index(),
        on=join_columns,
        how="left",
        validate="many_to_one",
    )


def calculate_and_save_thresholds(
    metrics_path: str | Path,
    thresholds_path: str | Path,
    results_path: str | Path,
    quantile: float = DEFAULT_POSTERIOR_MMD_QUANTILE,
    signed_error_coverage: float = DEFAULT_SIGNED_ERROR_COVERAGE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(metrics_path, keep_default_na=False)
    thresholds = calculate_thresholds(
        frame,
        quantile=quantile,
        signed_error_coverage=signed_error_coverage,
    )
    results = add_thresholds_to_metrics(frame, thresholds)
    thresholds_path = Path(thresholds_path)
    results_path = Path(results_path)
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds.to_csv(thresholds_path, index=False)
    results.to_csv(results_path, index=False)
    return thresholds, results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", default="20d_10n")
    parser.add_argument("--calibration-root", type=Path, default=CALIBRATION_ROOT)
    parser.add_argument(
        "--quantile",
        type=float,
        default=DEFAULT_POSTERIOR_MMD_QUANTILE,
        help="One-sided upper quantile for posterior MMD (default: 0.95).",
    )
    parser.add_argument(
        "--signed-error-coverage",
        type=float,
        default=DEFAULT_SIGNED_ERROR_COVERAGE,
        help="Central coverage for signed logML/PMP errors (default: 0.90).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.calibration_root / args.configuration
    thresholds, results = calculate_and_save_thresholds(
        root / "per_dataset_metrics.csv",
        root / "thresholds.csv",
        root / "per_dataset_results.csv",
        quantile=args.quantile,
        signed_error_coverage=args.signed_error_coverage,
    )
    print(f"Thresholds: {root / 'thresholds.csv'} ({len(thresholds)} rows)")
    print(f"Metrics with thresholds: {root / 'per_dataset_results.csv'} ({len(results)} rows)")


if __name__ == "__main__":
    main()
