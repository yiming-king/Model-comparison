from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path("/Users/yimingzang/Documents/Project/benchmark2")
sys.path.insert(0, str(ROOT))

from benchmark.examples.gaussian.analysis.summary_dimension_comparison import (
    DIAGNOSTICS,
    load_comparison_data,
)
from benchmark.examples.gaussian.config import ASSUMED_MODELS, CALIBRATION_ROOT


WORK_DIR = ROOT / ".codex_spreadsheet_work" / "gaussian_interval_20260831"
SUMMARY_ORDER = ("S=D", "S=2D", "S=4D")
METRICS = (
    ("log10_logml_error", "signed_logml_error"),
    ("pmp_error", "pmp_error"),
    ("posterior_mmd", "posterior_mmd"),
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


def raw_threshold_lookup() -> dict[tuple[str, str, str], tuple[float, float]]:
    thresholds = pd.read_csv(CALIBRATION_ROOT / "20d_10n" / "thresholds.csv")
    lookup: dict[tuple[str, str, str], tuple[float, float]] = {}
    error_metric = {
        "signed_logml_error": "signed_logml_error",
        "signed_pmp_error": "pmp_error",
        "posterior_mmd": "posterior_mmd",
    }
    for row in thresholds.itertuples(index=False):
        lookup[(row.network_tag, row.generating_model, error_metric[row.metric])] = (
            float(row.lower_threshold),
            float(row.threshold),
        )
    return lookup


def compute_rows() -> list[list[object]]:
    data = load_comparison_data()
    raw_thresholds = raw_threshold_lookup()
    summary_to_network = {"S=D": "20d_10n", "S=2D": "40d_10n", "S=4D": "80d_10n"}
    rows: list[list[object]] = []
    for analysis_metric, workbook_metric in METRICS:
        for model in ASSUMED_MODELS:
            for summary in SUMMARY_ORDER:
                for diagnostic in DIAGNOSTICS:
                    panel = data.loc[
                        data["error_metric"].eq(analysis_metric)
                        & data["assumed_model"].eq(model)
                        & data["summary"].eq(summary)
                        & data["diagnostic"].eq(diagnostic)
                    ].copy()
                    if len(panel) != 600:
                        raise ValueError(
                            f"Expected 600 Gaussian rows for {analysis_metric}/{model}/{summary}/{diagnostic}; found {len(panel)}"
                        )
                    values = pd.to_numeric(panel["error_value"], errors="coerce")
                    lower = pd.to_numeric(panel["error_lower_threshold"], errors="coerce")
                    upper = pd.to_numeric(panel["error_upper_threshold"], errors="coerce")
                    rho = pd.to_numeric(panel["rho"], errors="coerce")
                    keep = values.notna() & lower.notna() & upper.notna() & rho.notna()
                    values = values.loc[keep].to_numpy(float)
                    lower = lower.loc[keep].to_numpy(float)
                    upper = upper.loc[keep].to_numpy(float)
                    rho = rho.loc[keep].to_numpy(float)
                    error_positive = (values < lower) | (values > upper)
                    diagnostic_positive = rho > 1.0
                    tp = int(np.sum(error_positive & diagnostic_positive))
                    fp = int(np.sum(~error_positive & diagnostic_positive))
                    fn = int(np.sum(error_positive & ~diagnostic_positive))
                    tn = int(np.sum(~error_positive & ~diagnostic_positive))
                    network_tag = summary_to_network[summary]
                    raw_lower, raw_upper = raw_thresholds[(network_tag, model, workbook_metric)]
                    definition = (
                        "raw NPE-analytical [0, q95]"
                        if workbook_metric == "posterior_mmd"
                        else "raw NPE-analytical central 90% interval [q5, q95]"
                    )
                    rows.append(
                        [
                            "Gaussian",
                            "simulated",
                            diagnostic,
                            model,
                            summary,
                            workbook_metric,
                            definition,
                            raw_lower,
                            raw_upper,
                            tp,
                            fp,
                            fn,
                            tn,
                            tp + fp + fn + tn,
                            safe_rate(fn, fn + tp),
                            safe_rate(fp, fp + tn),
                            safe_rate(2 * tp, 2 * tp + fp + fn),
                            auc_rank(error_positive, rho),
                        ]
                    )
    return rows


if __name__ == "__main__":
    output = {
        "definition": "Gaussian signed logML/PMP central 90% interval [q5, q95]",
        "rows": compute_rows(),
    }
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "computed_gaussian_90.json").write_text(
        json.dumps(output, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"rows": len(output["rows"]), "path": str(WORK_DIR / "computed_gaussian_90.json")}))
