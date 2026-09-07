from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/yimingzang/Documents/Project/benchmark2")
WORK_DIR = ROOT / ".codex_spreadsheet_work" / "diffusion_100_update"
DIFFUSION_DIR = ROOT / "benchmark" / "examples" / "diffusion"
NPE_RESULT_DIR = DIFFUSION_DIR / "results" / "NPE_results"
POSTERIOR_DIR = DIFFUSION_DIR / "results" / "posterior_diagnostics"

SUMMARIES = {
    "S=D": "S1D",
    "S=2D": "S2D",
    "S=4D": "S4D",
    "S=6D": "S6D",
}
UPDATED_SUMMARIES = ("S=D", "S=2D", "S=4D")
DIAGNOSTICS = ("l2", "linf", "density", "mmd")
MODELS = ("m0", "m1", "m2", "m3")
SOURCE_GROUPS = ("empirical", "simulated")
METRICS = (
    (
        "log10_ml_error",
        "signed_logml_error",
        "ESS-filtered raw NPE-MCMC central 90% interval [q5, q95]",
    ),
    (
        "pmp_error",
        "signed_pmp_error",
        "ESS-filtered raw NPE-MCMC central 90% interval [q5, q95]",
    ),
    (
        "posterior_mmd",
        "posterior_mmd",
        "ESS-filtered raw NPE-MCMC interval [0, q95]",
    ),
)


def row_key(row: list[object]) -> tuple[object, ...]:
    return tuple(row[1:6])


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    keep = np.isfinite(scores)
    labels = labels[keep]
    scores = scores[keep]
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    rank_sum_pos = float(ranks[labels].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def load_threshold_csv(
    threshold_path: Path,
    cache_suffix: str,
    summaries: tuple[str, ...],
) -> dict[tuple[str, str, str], tuple[float, float]]:
    frame = pd.read_csv(threshold_path)
    lookup: dict[tuple[str, str, str], tuple[float, float]] = {}
    for summary in summaries:
        configuration = f"{SUMMARIES[summary]}_{cache_suffix}"
        for model in MODELS:
            for metric_name, source_metric, _definition in METRICS:
                selected = frame.loc[
                    frame["generating_model"].eq(model)
                    & frame["npe_configuration"].eq(configuration)
                    & frame["metric"].eq(source_metric)
                ]
                if len(selected) != 1:
                    raise ValueError(
                        f"Expected one threshold for {configuration}/{model}/{source_metric}; "
                        f"found {len(selected)}"
                    )
                lower = float(selected["lower_threshold"].iloc[0])
                upper = float(selected["threshold"].iloc[0])
                if metric_name == "log10_ml_error":
                    lower /= math.log(10.0)
                    upper /= math.log(10.0)
                lookup[(model, summary, metric_name)] = (lower, upper)
    return lookup


def derive_old_thresholds(
    per_dataset_metrics_path: Path,
    cache_suffix: str,
) -> dict[tuple[str, str, str], tuple[float, float]]:
    calibration = pd.read_csv(per_dataset_metrics_path)
    thresholds: dict[tuple[str, str, str], tuple[float, float]] = {}
    for model in MODELS:
        for summary, token in SUMMARIES.items():
            configuration = f"{token}_{cache_suffix}"
            base = calibration.loc[
                calibration["npe_configuration"].eq(configuration)
                & calibration["generating_model"].eq(model)
            ].copy()
            ess_ratio = pd.to_numeric(base["importance_ess"], errors="coerce") / pd.to_numeric(
                base["num_npe_logml_draws"], errors="coerce"
            )
            base = base.loc[ess_ratio.ge(0.20)]
            matching = base.loc[base["candidate_model"].eq(model)]
            if matching.empty or base.empty:
                raise ValueError(f"No ESS-filtered calibration rows for {model}/{summary}")
            logml = pd.to_numeric(matching["signed_logml_error"], errors="coerce").dropna()
            posterior_mmd = pd.to_numeric(matching["posterior_mmd"], errors="coerce").dropna()
            pmp = pd.to_numeric(base["signed_pmp_error"], errors="coerce").dropna()
            thresholds[(model, summary, "log10_ml_error")] = (
                float(logml.quantile(0.05) / math.log(10.0)),
                float(logml.quantile(0.95) / math.log(10.0)),
            )
            thresholds[(model, summary, "pmp_error")] = (
                float(pmp.quantile(0.05)),
                float(pmp.quantile(0.95)),
            )
            thresholds[(model, summary, "posterior_mmd")] = (
                0.0,
                float(posterior_mmd.quantile(0.95)),
            )
    return thresholds


def compute_rows(
    cache_suffix: str,
    thresholds: dict[tuple[str, str, str], tuple[float, float]],
    summaries: tuple[str, ...],
) -> list[list[object]]:
    cached: dict[tuple[str, str], pd.DataFrame] = {}
    posterior_maps: dict[str, pd.Series] = {}
    for summary in summaries:
        token = SUMMARIES[summary]
        posterior_path = POSTERIOR_DIR / f"npe_{token}_{cache_suffix}_all_observed_posterior.csv"
        posterior = pd.read_csv(posterior_path)
        posterior_maps[summary] = posterior.set_index(["dataset", "id", "model"])["observed_mmd"]
        for diagnostic in DIAGNOSTICS:
            result_path = NPE_RESULT_DIR / f"npe_{token}_{cache_suffix}_all_observed_diagnostic_{diagnostic}.csv"
            cached[(summary, diagnostic)] = pd.read_csv(result_path)

    rows: list[list[object]] = []
    for source_group in SOURCE_GROUPS:
        for metric_name, _source_metric, definition in METRICS:
            for model in MODELS:
                for summary in summaries:
                    lower, upper = thresholds[(model, summary, metric_name)]
                    for diagnostic in DIAGNOSTICS:
                        all_rows = cached[(summary, diagnostic)]
                        source_filter = (
                            all_rows["dataset"].eq("empirical")
                            if source_group == "empirical"
                            else all_rows["dataset"].ne("empirical")
                        )
                        frame = all_rows.loc[source_filter].reset_index(drop=True)
                        scores = pd.to_numeric(frame[f"rho_{model}"], errors="coerce").to_numpy(float)
                        if metric_name == "log10_ml_error":
                            errors = (
                                pd.to_numeric(frame[f"signed_logml_error_{model}"], errors="coerce")
                                .to_numpy(float)
                                / math.log(10.0)
                            )
                        elif metric_name == "pmp_error":
                            errors = pd.to_numeric(
                                frame[f"signed_pmp_error_{model}"], errors="coerce"
                            ).to_numpy(float)
                        else:
                            index = pd.MultiIndex.from_arrays(
                                [
                                    frame["dataset"].to_numpy(),
                                    frame["id"].to_numpy(),
                                    np.repeat(model, len(frame)),
                                ]
                            )
                            errors = posterior_maps[summary].reindex(index).to_numpy(float)
                        if not np.isfinite(scores).all() or not np.isfinite(errors).all():
                            raise ValueError(
                                f"Missing evaluation values for {cache_suffix}/{source_group}/"
                                f"{metric_name}/{model}/{summary}/{diagnostic}"
                            )
                        error_positive = (errors < lower) | (errors > upper)
                        diagnostic_positive = scores > 1.0
                        tp = int((diagnostic_positive & error_positive).sum())
                        fp = int((diagnostic_positive & ~error_positive).sum())
                        fn = int((~diagnostic_positive & error_positive).sum())
                        tn = int((~diagnostic_positive & ~error_positive).sum())
                        n = int(len(frame))
                        fnr = safe_rate(fn, fn + tp)
                        fpr = safe_rate(fp, fp + tn)
                        f1 = safe_rate(2 * tp, 2 * tp + fp + fn)
                        auc = auc_rank(error_positive, scores)
                        rows.append(
                            [
                                "Diffusion",
                                source_group,
                                diagnostic,
                                model,
                                summary,
                                metric_name,
                                definition,
                                lower,
                                upper,
                                tp,
                                fp,
                                fn,
                                tn,
                                n,
                                "N/A" if fnr is None else fnr,
                                "N/A" if fpr is None else fpr,
                                "N/A" if f1 is None else f1,
                                "N/A" if auc is None else auc,
                            ]
                        )
    return rows


def values_equal(left: object, right: object, tolerance: float = 1e-12) -> bool:
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        return (
            isinstance(right, (int, float))
            and not isinstance(right, bool)
            and math.isfinite(float(left))
            and math.isfinite(float(right))
            and abs(float(left) - float(right)) <= tolerance
        )
    return left == right


def validate_current_against_old(
    current_rows: list[list[object]],
    old_rows: list[list[object]],
) -> dict[str, object]:
    current = {
        row_key(row): row
        for row in current_rows
        if row and row[0] == "Diffusion"
    }
    computed = {row_key(row): row for row in old_rows}
    if current.keys() != computed.keys():
        missing = sorted(set(computed) - set(current))[:5]
        extra = sorted(set(current) - set(computed))[:5]
        raise AssertionError(f"Current/old keys differ; missing={missing}, extra={extra}")
    mismatch_count = 0
    max_abs_difference = 0.0
    samples: list[dict[str, object]] = []
    for key in current:
        left = current[key]
        right = computed[key]
        for column_index in range(7, 18):
            if not values_equal(left[column_index], right[column_index]):
                mismatch_count += 1
                if len(samples) < 10:
                    samples.append(
                        {
                            "key": key,
                            "column_index": column_index,
                            "current": left[column_index],
                            "computed": right[column_index],
                        }
                    )
            elif isinstance(left[column_index], (int, float)) and isinstance(
                right[column_index], (int, float)
            ):
                max_abs_difference = max(
                    max_abs_difference,
                    abs(float(left[column_index]) - float(right[column_index])),
                )
    if mismatch_count:
        raise AssertionError(f"Old pipeline validation failed: {samples}")
    return {
        "validated_rows": len(current),
        "mismatch_count": mismatch_count,
        "max_abs_difference": max_abs_difference,
    }


def build_payload(config: dict[str, object]) -> dict[str, object]:
    current_path = WORK_DIR / f"{config['workbook']}.current.json"
    current_dump = json.loads(current_path.read_text(encoding="utf-8"))
    current_values: list[list[object]] = current_dump["sourceValues"]
    current_rows = current_values[1:]

    old_thresholds = derive_old_thresholds(
        Path(config["old_per_dataset_metrics"]),
        str(config["cache_suffix"]),
    )
    old_rows = compute_rows(
        str(config["cache_suffix"]),
        old_thresholds,
        tuple(SUMMARIES),
    )
    validation = validate_current_against_old(current_rows, old_rows)

    new_thresholds = load_threshold_csv(
        Path(config["new_threshold_path"]),
        str(config["cache_suffix"]),
        UPDATED_SUMMARIES,
    )
    new_rows = compute_rows(
        str(config["cache_suffix"]),
        new_thresholds,
        UPDATED_SUMMARIES,
    )
    new_lookup = {row_key(row): row for row in new_rows}

    replacements: list[dict[str, object]] = []
    changed_rows = 0
    changed_cells = 0
    changed_by_metric = {metric[0]: 0 for metric in METRICS}
    preserved_s6d_rows = 0
    for zero_based_index, row in enumerate(current_rows):
        if not row or row[0] != "Diffusion":
            continue
        if row[4] == "S=6D":
            preserved_s6d_rows += 1
            continue
        key = row_key(row)
        replacement = new_lookup.get(key)
        if replacement is None:
            raise KeyError(f"Missing new result for {key}")
        replacement_values = replacement[7:18]
        differences = sum(
            not values_equal(left, right)
            for left, right in zip(row[7:18], replacement_values)
        )
        if differences:
            changed_rows += 1
            changed_cells += differences
            changed_by_metric[str(row[5])] += 1
        replacements.append(
            {
                "excelRow": zero_based_index + 2,
                "key": list(key),
                "valuesHtoR": replacement_values,
            }
        )

    if len(replacements) != 288:
        raise AssertionError(f"Expected 288 updated Diffusion rows, found {len(replacements)}")
    if preserved_s6d_rows != 96:
        raise AssertionError(f"Expected 96 preserved S=6D rows, found {preserved_s6d_rows}")

    threshold_frame = pd.read_csv(Path(config["new_threshold_path"]))
    n_values = {
        metric: sorted(map(int, group["n_values"].unique()))
        for metric, group in threshold_frame.groupby("metric")
    }
    return {
        "workbook": config["workbook"],
        "cache_suffix": config["cache_suffix"],
        "new_threshold_path": str(config["new_threshold_path"]),
        "fallback_s6d_threshold_path": str(config["fallback_s6d_threshold_path"]),
        "updated_summaries": list(UPDATED_SUMMARIES),
        "preserved_summary": "S=6D",
        "replacement_count": len(replacements),
        "preserved_s6d_rows": preserved_s6d_rows,
        "changed_rows": changed_rows,
        "changed_cells": changed_cells,
        "changed_by_metric": changed_by_metric,
        "n_values": n_values,
        "old_pipeline_validation": validation,
        "replacements": replacements,
    }


def main() -> None:
    configs = [
        {
            "workbook": "FNR-FPR tables without MMD rerun1.xlsx",
            "cache_suffix": "noMMD_rerun1",
            "new_threshold_path": DIFFUSION_DIR
            / "calibration_outputs_100_noMMD_rerun1"
            / "thresholds.csv",
            "fallback_s6d_threshold_path": DIFFUSION_DIR
            / "calibration_outputs_rerun1"
            / "thresholds.csv",
            "old_per_dataset_metrics": DIFFUSION_DIR
            / "calibration_outputs_rerun1"
            / "per_dataset_metrics.csv",
        },
        {
            "workbook": "FNR-FPR tables without MMD.xlsx",
            "cache_suffix": "noMMD",
            "new_threshold_path": DIFFUSION_DIR
            / "calibration_outputs_100_noMMD"
            / "thresholds.csv",
            "fallback_s6d_threshold_path": DIFFUSION_DIR
            / "calibration_outputs"
            / "thresholds.csv",
            "old_per_dataset_metrics": DIFFUSION_DIR
            / "calibration_outputs"
            / "per_dataset_metrics.csv",
        },
    ]
    report: dict[str, object] = {}
    for config in configs:
        payload = build_payload(config)
        output_path = WORK_DIR / f"{config['workbook']}.payload.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        report[str(config["workbook"])] = {
            key: payload[key]
            for key in [
                "replacement_count",
                "preserved_s6d_rows",
                "changed_rows",
                "changed_cells",
                "changed_by_metric",
                "n_values",
                "old_pipeline_validation",
            ]
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
