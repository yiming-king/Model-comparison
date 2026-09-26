"""Shared signed PMP error calibration for direct classifiers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["generating_model", "dataset_id"]


def load_indirect_gold(
    path: str | Path,
    dataset_index: pd.DataFrame,
    models: tuple[str, ...],
    gold_column: str,
) -> pd.DataFrame:
    """Align saved indirect gold PMPs with the benchmark's exact IDs and order."""
    source = pd.read_csv(path)
    columns = [*KEYS, "candidate_model", gold_column]
    if set(columns) - set(source):
        raise ValueError(f"Indirect gold is missing columns: {set(columns) - set(source)}")
    if source[columns].isna().any().any():
        raise ValueError("Indirect gold contains missing dataset keys or probabilities")
    groups = source.groupby(columns[:-1], sort=False)[gold_column]
    consistency = groups.agg(["min", "max"])
    if (consistency["max"] - consistency["min"] > 1e-12).any():
        raise ValueError("Indirect gold PMP differs across summary networks")
    unique = groups.first().reset_index()
    expected = dataset_index[KEYS].merge(
        pd.DataFrame({"candidate_model": models}), how="cross"
    )
    if len(unique) != len(expected) or set(map(tuple, unique[columns[:-1]].to_numpy())) != set(
        map(tuple, expected[columns[:-1]].to_numpy())
    ):
        raise ValueError("Indirect gold and saved benchmark dataset IDs differ")
    gold = expected.merge(unique, on=columns[:-1], how="left", validate="one_to_one")
    values = gold[gold_column].to_numpy(dtype=float).reshape(-1, len(models))
    if not np.isfinite(values).all() or (values < -1e-8).any() or not np.allclose(
        values.sum(axis=1), 1.0, rtol=0, atol=1e-6
    ):
        raise ValueError("Indirect gold PMP vectors must be finite probabilities")
    return gold


def calibrate_direct_pmp_signed(
    metrics: pd.DataFrame,
    models: tuple[str, ...],
    gold_column: str,
    coverage: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return signed component errors and their central reference intervals."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    required = ["loss_function", *KEYS, "candidate_model", gold_column, "direct_pmp"]
    if set(required) - set(metrics) or metrics.empty:
        raise ValueError(f"Missing or empty direct PMP metrics: {set(required) - set(metrics)}")
    row_keys = required[:4]
    if metrics[row_keys].isna().any().any() or metrics.duplicated(row_keys).any():
        raise ValueError("Direct PMP component keys must be unique and nonempty")
    probabilities = metrics[[gold_column, "direct_pmp"]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < -1e-8).any() or (
        probabilities > 1 + 1e-8
    ).any():
        raise ValueError("Direct and gold PMPs must be finite probabilities")
    component_sets = metrics.groupby(row_keys[:-1], sort=False)["candidate_model"].agg(set)
    if not component_sets.map(lambda value: value == set(models)).all():
        raise ValueError("Each benchmark dataset needs all four PMP components")
    sums = metrics.groupby(row_keys[:-1], sort=False)[[gold_column, "direct_pmp"]].sum()
    if not np.allclose(sums.to_numpy(), 1, rtol=0, atol=1e-6):
        raise ValueError("Direct and gold PMP vectors must each sum to one")

    errors = metrics[row_keys].copy()
    errors["signed_pmp_error"] = metrics["direct_pmp"] - metrics[gold_column]
    lower_q = (1 - coverage) / 2
    rows = []
    for (loss, candidate), group in errors.groupby(
        ["loss_function", "candidate_model"], sort=False
    ):
        if set(group["generating_model"]) != set(models):
            raise ValueError("Each direct loss requires all four generating models")
        samples = group["signed_pmp_error"].to_numpy(dtype=float)
        low, median, high = np.quantile(
            samples, [lower_q, 0.5, 1 - lower_q], method="linear"
        )
        rows.append(
            {
                "loss_function": loss,
                "network_tag": f"direct_{loss}",
                "candidate_model": candidate,
                "metric": "signed_pmp_error",
                "quantile": coverage,
                "lower_quantile": lower_q,
                "upper_quantile": 1 - lower_q,
                "quantile_method": "linear",
                "lower_threshold": low,
                "threshold": high,
                "median": median,
                "n_values": len(samples),
                "aggregation": "signed_error_per_candidate_across_benchmark_datasets",
            }
        )
    return errors, pd.DataFrame(rows)
