from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/yimingzang/Documents/Project/benchmark2")
WORK_DIR = ROOT / ".codex_spreadsheet_work" / "diffusion_m0_m2"
DIFFUSION_DIR = ROOT / "benchmark" / "examples" / "diffusion"
NPE_RESULT_DIR = DIFFUSION_DIR / "results" / "NPE_results"
POSTERIOR_DIR = DIFFUSION_DIR / "results" / "posterior_diagnostics"
PLOTS_DIR = DIFFUSION_DIR / "results" / "plots"

SUMMARIES = {
    "S=D": "S1D",
    "S=2D": "S2D",
    "S=4D": "S4D",
    "S=6D": "S6D",
}
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


def load_thresholds(threshold_path: Path, cache_suffix: str) -> dict[tuple[str, str, str], tuple[float, float]]:
    frame = pd.read_csv(threshold_path)
    lookup: dict[tuple[str, str, str], tuple[float, float]] = {}
    for summary, token in SUMMARIES.items():
        configuration = f"{token}_{cache_suffix}"
        for model in MODELS:
            for metric_name, threshold_metric, _definition in METRICS:
                source_metric = {
                    "log10_ml_error": "signed_logml_error",
                    "pmp_error": "signed_pmp_error",
                    "posterior_mmd": "posterior_mmd",
                }[metric_name]
                selected = frame.loc[
                    frame["generating_model"].eq(model)
                    & frame["npe_configuration"].eq(configuration)
                    & frame["metric"].eq(source_metric)
                ]
                if len(selected) != 1:
                    raise ValueError(
                        f"Expected one threshold row for {model}, {summary}, {metric_name}; found {len(selected)}"
                    )
                lower = float(selected["lower_threshold"].iloc[0])
                upper = float(selected["threshold"].iloc[0])
                if metric_name == "log10_ml_error":
                    lower /= math.log(10.0)
                    upper /= math.log(10.0)
                lookup[(model, summary, metric_name)] = (lower, upper)
    return lookup


def compute_run(cache_suffix: str, per_dataset_metrics_path: Path) -> list[list[object]]:
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
                raise ValueError(f"No ESS-filtered calibration rows for {model}, {summary}")
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

    cached: dict[tuple[str, str], pd.DataFrame] = {}
    posterior_maps: dict[str, pd.Series] = {}
    for summary, token in SUMMARIES.items():
        posterior_path = POSTERIOR_DIR / f"npe_{token}_{cache_suffix}_all_observed_posterior.csv"
        posterior = pd.read_csv(posterior_path)
        posterior_maps[summary] = posterior.set_index(["dataset", "id", "model"])["observed_mmd"]
        for diagnostic in DIAGNOSTICS:
            result_path = NPE_RESULT_DIR / f"npe_{token}_{cache_suffix}_all_observed_diagnostic_{diagnostic}.csv"
            cached[(summary, diagnostic)] = pd.read_csv(result_path)

    rows: list[list[object]] = []
    for source_group in SOURCE_GROUPS:
        for metric_name, _raw_metric, definition in METRICS:
            for model in MODELS:
                for summary in SUMMARIES:
                    lower, upper = thresholds[(model, summary, metric_name)]
                    for diagnostic in DIAGNOSTICS:
                        frame = cached[(summary, diagnostic)]
                        source_filter = (
                            frame["dataset"].eq("empirical")
                            if source_group == "empirical"
                            else frame["dataset"].ne("empirical")
                        )
                        frame = frame.loc[source_filter].reset_index(drop=True)
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
                                f"Missing values for {cache_suffix}, {source_group}, {metric_name}, {model}, {summary}, {diagnostic}"
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
                        fnr_cell = "N/A" if fnr is None else fnr
                        fpr_cell = "N/A" if fpr is None else fpr
                        f1_cell = "N/A" if f1 is None else f1
                        auc_cell = "N/A" if auc is None else auc
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
                                fnr_cell,
                                fpr_cell,
                                f1_cell,
                                auc_cell,
                            ]
                        )
    return rows


def key(row: list[object]) -> tuple[object, ...]:
    return tuple(row[:7])


def validate_existing(computed: list[list[object]], dump_path: Path) -> dict[str, object]:
    with dump_path.open("r", encoding="utf-8") as handle:
        dump = json.load(handle)
    existing_rows = [
        row
        for row in dump["sheets"]["Source rates"]["values"][1:]
        if row[0] == "Diffusion" and row[3] in {"m1", "m3"}
    ]
    computed_rows = [row for row in computed if row[3] in {"m1", "m3"}]
    existing = {key(row): row for row in existing_rows}
    candidate = {key(row): row for row in computed_rows}
    if existing.keys() != candidate.keys():
        raise AssertionError(
            f"Existing/computed keys differ: existing={len(existing)}, computed={len(candidate)}"
        )

    mismatches: list[dict[str, object]] = []
    max_abs_error = 0.0
    for row_key in existing:
        left = existing[row_key]
        right = candidate[row_key]
        for column_index, (left_value, right_value) in enumerate(zip(left[7:], right[7:]), start=7):
            if left_value is None or right_value is None:
                equal = left_value is None and right_value is None
                difference = 0.0 if equal else math.inf
            elif isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                difference = abs(float(left_value) - float(right_value))
                equal = difference <= 1e-12
                max_abs_error = max(max_abs_error, difference)
            else:
                equal = left_value == right_value
                difference = 0.0 if equal else math.inf
            if not equal:
                mismatches.append(
                    {
                        "key": row_key,
                        "column_index": column_index,
                        "existing": left_value,
                        "computed": right_value,
                        "difference": difference,
                    }
                )
                if len(mismatches) >= 20:
                    break
        if len(mismatches) >= 20:
            break
    if mismatches:
        raise AssertionError(f"Existing m1/m3 validation failed: {mismatches[:5]}")
    return {
        "validated_rows": len(existing),
        "max_abs_error": max_abs_error,
        "mismatches": 0,
    }


def main() -> None:
    configs = [
        {
            "slug": "rerun1",
            "cache_suffix": "noMMD_rerun1",
            "threshold_path": DIFFUSION_DIR / "calibration_outputs_rerun1" / "thresholds.csv",
            "classification_root": PLOTS_DIR / "summary_diagnostics_noMMD_rerun1_calibrated_thresholds" / "all",
            "per_dataset_metrics_path": DIFFUSION_DIR / "calibration_outputs_rerun1" / "per_dataset_metrics.csv",
            "workbook": "FNR-FPR tables without MMD rerun1.xlsx",
        },
        {
            "slug": "without_mmd",
            "cache_suffix": "noMMD",
            "threshold_path": DIFFUSION_DIR / "calibration_outputs" / "thresholds.csv",
            "classification_root": PLOTS_DIR / "summary_diagnostics_noMMD_calibrated_thresholds" / "all",
            "per_dataset_metrics_path": DIFFUSION_DIR / "calibration_outputs" / "per_dataset_metrics.csv",
            "workbook": "FNR-FPR tables without MMD.xlsx",
        },
    ]
    report = {}
    for config in configs:
        computed = compute_run(config["cache_suffix"], config["per_dataset_metrics_path"])
        validation = validate_existing(
            computed,
            WORK_DIR / f"{config['workbook']}.dump.json",
        )
        new_rows = [row for row in computed if row[3] in {"m0", "m2"}]
        payload = {
            "workbook": config["workbook"],
            "cache_suffix": config["cache_suffix"],
            "threshold_path": str(config["threshold_path"]),
            "classification_root": str(config["classification_root"]),
            "per_dataset_metrics_path": str(config["per_dataset_metrics_path"]),
            "validation": validation,
            "new_rows": new_rows,
        }
        output_path = WORK_DIR / f"computed_{config['slug']}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        report[config["slug"]] = {
            "validation": validation,
            "new_rows": len(new_rows),
            "output": str(output_path),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
